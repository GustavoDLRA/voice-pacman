"""
voice_control.py

Background microphone capture + live transcription, refactored from
voice_first_tests/live_transcribe_sd_choice.py so it can run embedded
inside the Pacman game process instead of as a standalone script.

VoiceController only ever produces plain text via get_text() -- it knows
nothing about Pacman, directions, or pygame. The word->direction mapping
is a separate layer on top of this.
"""


import os
import sys
import sysconfig

def _ensure_cuda_libs_on_path():
    site_packages = sysconfig.get_paths()["purelib"]
    cuda_lib_dirs = [
        os.path.join(site_packages, "nvidia", "cublas", "lib"),
        os.path.join(site_packages, "nvidia", "cudnn", "lib"),
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if all (d in current for d in cuda_lib_dirs):
        return # already set -- this is the re-exec'd process
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(cuda_lib_dirs + [current])
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_cuda_libs_on_path()




import queue
import threading

import numpy as np
import sounddevice as sd

TARGET_SR = 16000  # Whisper model sample rate

def _resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return audio
    try:
        from samplerate import resample
        return resample(audio, TARGET_SR / orig_sr, "sinc_fastest").astype(np.float32)
    except ImportError:
        duration = len(audio) / orig_sr
        n_target = int(duration * TARGET_SR)
        x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0, duration, num=n_target, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

class VoiceController:
    def __init__(self, device=None, language="en", model="large-v3-turbo", 
                 compute_device="cuda", compute_type=None, chunk_duration=4.0,
                 max_no_speech_prob = 0.6, min_avg_logprob=-1.0):

        self.device = device
        self.language = language
        self.model_name = model
        self.compute_device = compute_device
        self.compute_type = compute_type or ("float16" if compute_device == "cuda" else "int8")
        self.chunk_duration = chunk_duration
        self.max_no_speech_prob = max_no_speech_prob
        self.min_avg_logprob = min_avg_logprob

        self.model = None
        self.model_ready = threading.Event()
        self.error = None

        self._raw_queue = queue.Queue(maxsize=400)
        self._chunk_queue = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()

        self._latest_text = None
        self._latest_lock = threading.Lock()

        self._stream = None
        self._threads = []
        self._samplerate = None
        self._channels = None
        self._frames_per_chunk = None

    # --- lifecycle --------------------------------------------------------------------------------------

    def start(self):
        threading.Thread(target=self._startup, daemon=True).start()

    def stop(self):
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        for t in self._threads:
            t.join(timeout=2)

    def is_ready(self):
        return self.model_ready.is_set()

    def get_text(self):
        """Non-blocking. Returns the most recent recognized phrase (str) or
        None, and clears it so the same phrase is never read twice."""
        with self._latest_lock:
            text, self._latest_text = self._latest_text, None
        return text

    # --- internals --------------------------------------------------------------------------------------

    def _startup(self):
        try:
            print(f"[voice] loading '{self.model_name}' on {self.compute_device} ({self.compute_type})...")
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_name, device=self.compute_device,
                                    compute_type=self.compute_type)
            self.model_ready.set()
            print("[voice] model ready.")
        except Exception as e:
            self.error = f"Failed to load speech model: {e}"
            return

        if self._stop_event.is_set():
            return # stop() was called while we were still loading the model

        try:
            devinfo = sd.query_devices(self.device, "input")
            self._samplerate = int(devinfo["default_samplerate"])
            self._channels = min(devinfo["max_input_channels"], 2) or 1
            self._frames_per_chunk = int(self._samplerate * self.chunk_duration)

            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self._samplerate,
                channels=self._channels,
                callback=self._audio_callback,
                blocksize=int(self._samplerate * 0.2),
                latency="high",
            )
            self._stream.start()
            print(f"[voice] listening on device {devinfo['name']!r} ({self._samplerate}Hz, {self._channels}ch)")
        except Exception as e:
            self.error = f"Failed to open audio input device {self.device!r}: {e}"
            return

        chunker = threading.Thread(target=self._chunker_loop, daemon=True)
        transcriber = threading.Thread(target=self._transcriber_loop, daemon=True)
        chunker.start()
        transcriber.start()
        self._threads = [chunker, transcriber]

    def _audio_callback(self, indata, frames, time_info, status):
        try:
            self._raw_queue.put_nowait(indata.copy())
        except queue.Full:
            pass # fell behind; drop rather than block the audio driver

    def _chunker_loop(self):
        buffer = []
        total = 0
        while not self._stop_event.is_set():
            try:
                block = self._raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            buffer.append(block)
            total += len(block)
            if total < self._frames_per_chunk:
                continue
            audio_data = np.concatenate(buffer)[:self._frames_per_chunk]
            buffer, total = [], 0
            try:
                self._chunk_queue.put_nowait(audio_data)
            except queue.Full:
                try:
                    self._chunk_queue.get_nowait()  # drop oldest chunk
                except queue.Empty:
                    pass
                self._chunk_queue.put_nowait(audio_data)

    def _transcriber_loop(self):
        self.model_ready.wait()
        while not self._stop_event.is_set():
            if self.error:
                return
            try:
                audio_data = self._chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._channels > 1:
                audio_data = audio_data.reshape(-1, self._channels).mean(axis=1)
            audio_data = audio_data.astype(np.float32).flatten()
            audio_data = _resample_to_16k(audio_data, self._samplerate)

            segments, info = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                without_timestamps=True,
            )

            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                looks_hallucinated = (
                    segment.no_speech_prob > self.max_no_speech_prob or
                    segment.avg_logprob < self.min_avg_logprob
                )
                if looks_hallucinated:
                    continue
                with self._latest_lock:
                    self._latest_text = text

if __name__ == "__main__":
    # Quick standalone check that the refactor didn't break anything --
    # same idea as the old script, just driven through the class now.
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--model", type=str, default="large-v3-turbo")
    parser.add_argument("--compute-device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    vc = VoiceController(device=args.device, language=args.language, 
                         model=args.model, compute_device=args.compute_device)

    vc.start()
    print("Loading model...")
    try:
        while True:
            if vc.error:
                print(vc.error)
                break
            if vc.is_ready():
                text = vc.get_text()
                if text:
                    print(f"heard: {text}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        vc.stop()

