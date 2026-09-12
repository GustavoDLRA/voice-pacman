#!/usr/bin/env python3
"""
Live microphone transcription using sounddevice + faster-whisper directly
(bypasses RealtimeSTT/PyAudio, which struggled with this machine's
PipeWire + ALSA setup).

Captures audio at the mic's NATIVE sample rate/channel count, downmixes to
mono, resamples to 16kHz in Python, and transcribes on a background thread
so a slow transcription pass never blocks audio capture. Segments that
look like Whisper hallucinations (low confidence / likely non-speech) are
filtered out before printing.

Usage:
    python list_mics_sd.py                      # find your mic's index first
    python live_transcribe_sd.py --device 7
    python live_transcribe_sd.py --device 7 --language es
    python live_transcribe_sd.py --device 7 --chunk-duration 3
    python live_transcribe_sd.py --device 7 --compute-device cpu
    python live_transcribe_sd.py --device 7 --show-dropped   # debug hallucination filter

Press Ctrl+C to stop. Saves a timestamped transcript on exit.
"""

import argparse
import datetime
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd


TARGET_SR = 16000


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return audio
    try:
        from samplerate import resample
        return resample(audio, TARGET_SR / orig_sr, "sinc_fastest").astype(np.float32)
    except ImportError:
        # Fallback: linear interpolation. Lower quality than samplerate's
        # sinc resampler but needs no extra dependency.
        duration = len(audio) / orig_sr
        n_target = int(duration * TARGET_SR)
        x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0, duration, num=n_target, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Live mic transcription with sounddevice + faster-whisper")
    parser.add_argument("--device", type=int, default=None,
                         help="sounddevice input device index (see list_mics_sd.py). Default: system default.")
    parser.add_argument("--language", default=None, help="Force a language code (en, es). Default: auto-detect.")
    parser.add_argument("--model", default="large-v3-turbo", help="faster-whisper model (default: large-v3-turbo)")
    parser.add_argument("--compute-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default=None, help="Override compute type, e.g. int8, int8_float16, float16")
    parser.add_argument("--chunk-duration", type=float, default=4.0,
                         help="Seconds of audio per transcription chunk (default: 4.0)")
    parser.add_argument("--max-no-speech-prob", type=float, default=0.6,
                         help="Drop segments whose no_speech_prob exceeds this (default: 0.6)")
    parser.add_argument("--min-avg-logprob", type=float, default=-1.0,
                         help="Drop segments whose avg_logprob is below this (default: -1.0)")
    parser.add_argument("--show-dropped", action="store_true",
                         help="Print segments filtered out as likely hallucinations, for tuning the thresholds.")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed.\nRun: pip install faster-whisper")

    devinfo = sd.query_devices(args.device, "input")
    samplerate = int(devinfo["default_samplerate"])
    channels = min(devinfo["max_input_channels"], 2) or 1

    # Note: RTX 50-series (Blackwell/sm_120) GPUs crash on INT8 compute types
    # (CUBLAS_STATUS_NOT_SUPPORTED) in CTranslate2 versions before the fix in
    # v4.6.3 disabled INT8 there — float16 is the safe default on GPU.
    compute_type = args.compute_type or ("float16" if args.compute_device == "cuda" else "int8")

    print(f"Input device: {devinfo['name']!r} | native_sr={samplerate}Hz channels={channels}")
    print(f"Loading model '{args.model}' on {args.compute_device} ({compute_type})...")
    model = WhisperModel(args.model, device=args.compute_device, compute_type=compute_type)
    print("Model loaded. Speak now. Press Ctrl+C to stop.\n")

    frames_per_chunk = int(samplerate * args.chunk_duration)

    # Small raw blocks straight from the audio callback.
    raw_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
    # Assembled chunks ready to transcribe. Kept small on purpose: if
    # transcription falls behind real time we drop old chunks (below)
    # rather than let a backlog grow, so live output stays near real-time.
    chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)

    transcript_lines = []
    transcript_lock = threading.Lock()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        try:
            raw_queue.put_nowait(indata.copy())
        except queue.Full:
            pass  # drop if we fall drastically behind; keeps latency bounded

    def chunker_thread_fn():
        """Only accumulates raw blocks into fixed-size chunks — kept
        deliberately cheap so it's never busy long enough to itself become
        a bottleneck between the audio callback and the transcriber."""
        buffer = []
        total = 0
        while not stop_event.is_set():
            try:
                block = raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            buffer.append(block)
            total += len(block)
            if total < frames_per_chunk:
                continue

            audio_data = np.concatenate(buffer)[:frames_per_chunk]
            buffer = []
            total = 0

            try:
                chunk_queue.put_nowait(audio_data)
            except queue.Full:
                # Transcription is falling behind; drop the oldest pending
                # chunk so we keep tracking near-live audio instead of an
                # ever-growing backlog.
                try:
                    chunk_queue.get_nowait()
                except queue.Empty:
                    pass
                chunk_queue.put_nowait(audio_data)

    def transcriber_thread_fn():
        while not stop_event.is_set():
            try:
                audio_data = chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if channels > 1:
                audio_data = audio_data.reshape(-1, channels).mean(axis=1)
            audio_data = audio_data.astype(np.float32).flatten()
            audio_data = resample_to_16k(audio_data, samplerate)

            segments, info = model.transcribe(
                audio_data,
                language=args.language,
                beam_size=5,
                vad_filter=True,
                without_timestamps=True,
            )

            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue

                # Whisper-family models can hallucinate plausible-sounding
                # phrases on silence/noise. no_speech_prob and avg_logprob
                # are the model's own confidence signals for catching that.
                looks_hallucinated = (
                    segment.no_speech_prob > args.max_no_speech_prob
                    or segment.avg_logprob < args.min_avg_logprob
                )
                if looks_hallucinated:
                    if args.show_dropped:
                        print(f"  [dropped: no_speech={segment.no_speech_prob:.2f} "
                              f"avg_logprob={segment.avg_logprob:.2f}] {text}")
                    continue

                stamp = datetime.datetime.now().strftime("%H:%M:%S")
                lang_tag = f"({info.language}) " if args.language is None else ""
                print(f"[{stamp}] {lang_tag}{text}")
                with transcript_lock:
                    transcript_lines.append(f"[{stamp}] {text}")

    chunker_thread = threading.Thread(target=chunker_thread_fn, daemon=True)
    transcriber_thread = threading.Thread(target=transcriber_thread_fn, daemon=True)

    try:
        with sd.InputStream(
            device=args.device,
            samplerate=samplerate,
            channels=channels,
            callback=audio_callback,
            blocksize=int(samplerate * 0.2),
            latency="high",
        ):
            chunker_thread.start()
            transcriber_thread.start()
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        chunker_thread.join(timeout=2)
        transcriber_thread.join(timeout=5)
        if transcript_lines:
            out_path = f"live_transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(transcript_lines) + "\n")
            print(f"\nSaved transcript to: {out_path}")
        else:
            print("\nNo speech captured.")


if __name__ == "__main__":
    main()
