#!/usr/bin/env python3
"""
Live microphone transcription using RealtimeSTT (built on faster-whisper).

Shows a fast, rough preview as you speak, then replaces it with a more
accurate final line once you pause. Auto-detects English vs. Spanish
per utterance by default, so you can freely switch languages mid-session.

Usage:
    python live_transcribe.py
    python live_transcribe.py --language es       # force Spanish
    python live_transcribe.py --language en        # force English
    python live_transcribe.py --device cpu         # no GPU
    python live_transcribe.py --no-preview         # only show final text

Press Ctrl+C to stop. A timestamped transcript is saved to
live_transcript_<timestamp>.txt when you exit.
"""

import argparse
import datetime
import sys


def main():
    parser = argparse.ArgumentParser(description="Live mic transcription with RealtimeSTT")
    parser.add_argument(
        "--language",
        default="",
        help="Force a language code (en, es). Default: auto-detect per utterance.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for the final (accurate) model. Default: cuda.",
    )
    parser.add_argument(
        "--final-model",
        default="large-v3-turbo",
        help="Model used for the accurate final transcript (default: large-v3-turbo).",
    )
    parser.add_argument(
        "--preview-model",
        default="base",
        help="Smaller/faster model used for the live rough preview (default: base).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the fast rolling preview; only print finalized lines.",
    )
    args = parser.parse_args()

    try:
        from RealtimeSTT import AudioToTextRecorder
    except ImportError:
        sys.exit(
            "RealtimeSTT is not installed.\n"
            "Run: pip install \"realtimestt[recommended]\""
        )

    compute_type = "int8_float16" if args.device == "cuda" else "int8"
    transcript_lines = []

    def on_preview(text):
        # Overwrite the same terminal line so the preview doesn't spam the console.
        print(f"\r... {text}", end="", flush=True)

    def on_final(text):
        text = text.strip()
        if not text:
            return
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\r[{stamp}] {text}" + " " * 20)  # pad to clear leftover preview chars
        transcript_lines.append(f"[{stamp}] {text}")

    recorder_kwargs = dict(
        model=args.final_model,
        language=args.language,
        device=args.device,
        compute_type=compute_type,
        enable_realtime_transcription=not args.no_preview,
        realtime_model_type=args.preview_model,
        on_realtime_transcription_update=None if args.no_preview else on_preview,
        post_speech_silence_duration=0.6,
    )

    print(f"Loading models (preview={args.preview_model}, final={args.final_model} on {args.device})...")
    print("Speak now. Press Ctrl+C to stop.\n")

    try:
        with AudioToTextRecorder(**recorder_kwargs) as recorder:
            while True:
                recorder.text(on_final)
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
