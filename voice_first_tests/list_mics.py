#!/usr/bin/env python3
"""
Lists audio input devices PyAudio/PortAudio can see, so you can find the
correct index for --input-device-index in live_transcribe.py.

Usage:
    python list_mics.py
"""

import pyaudio

p = pyaudio.PyAudio()

print(f"PortAudio default input device index: {p.get_default_input_device_info()['index']} "
      f"({p.get_default_input_device_info()['name']})\n")

print("All devices with at least one input channel:\n")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get("maxInputChannels", 0) > 0:
        print(
            f"  index {i}: {info['name']!r}  "
            f"(inputs={info['maxInputChannels']}, "
            f"default_sample_rate={int(info['defaultSampleRate'])})"
        )

p.terminate()
