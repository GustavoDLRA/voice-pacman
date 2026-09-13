import time
from voice_control import VoiceController
from voice_commands import parse_direction

vc = VoiceController(device=0, language="es")
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
                direction = parse_direction(text, vc.language)
                print(f"heard: {text!r} -> direction: {direction}")
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    vc.stop()