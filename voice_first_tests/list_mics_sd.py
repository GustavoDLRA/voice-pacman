#!/usr/bin/env python3
"""
Lists audio devices as `sounddevice` (PortAudio) sees them.

Note: these indices are numbered independently from PyAudio's list
(list_mics.py) — always use the index from the tool matching the
script you're running.

Usage:
    python list_mics_sd.py
"""

import sounddevice as sd

print(sd.query_devices())
print()

try:
    default_in = sd.default.device[0]
    info = sd.query_devices(default_in, "input")
    print(f"Default input device: index {default_in} -> {info['name']!r}")
except Exception as e:
    print(f"Could not resolve default input device: {e}")
