#!/usr/bin/env python3
"""
Live microphone transcription using sounddevice + faster-whisper directly
(bypasses RealtimeSTT/PyAudio, which struggle with this machine's
PipeWire + ALSA setup).

Captures audio at the mic's NATIVE sample rate/channel count (avoids the
"Invalid sample rate" error you get asking ALSA for 16kHz directly),
downmixes to mono, resamples to 16kHz in Python, and transcribes each
chunk with faster-whisper. Uses faster-whisper's built-in VAD filter to
skip silence rather than a separate voice-activity library.

Usage:
    python list_mics_sd.py                      # find your mic's index first
    python live_transcribe_sd.py --device 7
    python live_transcribe_sd.py --device 7 --language es
    python live_transcribe_sd.py --device 7 --chunk-duration 3
    python live_transcribe_sd.py --device 7 --compute-device cpu

Press Ctrl+C to stop. Saves a timestamped transcript on exit.
"""

import argparse
import datetime
import queue
import sys

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
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed.\nRun: pip install faster-whisper")

    devinfo = sd.query_devices(args.device, "input")
    samplerate = int(devinfo["default_samplerate"])
    channels = min(devinfo["max_input_channels"], 2) or 1

    compute_type = args.compute_type or ("int8_float16" if args.compute_device == "cuda" else "int8")

    print(f"Input device: {devinfo['name']!r} | native_sr={samplerate}Hz channels={channels}")
    print(f"Loading model '{args.model}' on {args.compute_device} ({compute_type})...")
    model = WhisperModel(args.model, device=args.compute_device, compute_type=compute_type)
    print("Model loaded. Speak now. Press Ctrl+C to stop.\n")

    frames_per_chunk = int(samplerate * args.chunk_duration)
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
    transcript_lines = []

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass  # drop if we fall behind rather than blocking the audio thread

    buffer = []

    try:
        with sd.InputStream(
            device=args.device,
            samplerate=samplerate,
            channels=channels,
            callback=audio_callback,
            blocksize=int(samplerate * 0.5),
            latency="high",
        ):
            while True:
                block = audio_queue.get()
                buffer.append(block)
                total_frames = sum(len(b) for b in buffer)
                if total_frames < frames_per_chunk:
                    continue

                audio_data = np.concatenate(buffer)[:frames_per_chunk]
                buffer = []

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
                    stamp = datetime.datetime.now().strftime("%H:%M:%S")
                    lang_tag = f"({info.language})" if args.language is None else ""
                    print(f"[{stamp}] {lang_tag} {text}")
                    transcript_lines.append(f"[{stamp}] {text}")
    except KeyboardInterrupt:
        pass
    finally:
        if transcript_lines:
            out_path = f"live_transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(transcript_lines) + "\n")
            print(f"\nSaved transcript to: {out_path}")
        else:
            print("\nNo speech captured.")


if __name__ == "__main__":
    main()
