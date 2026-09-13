"""
voice_commands.py

Maps recognized speech text (from VoiceController.get_text()) to a
Pacman direction constant. Deliberately separate from voice_control.py --
that module only ever produces plain text; this is the only place that
knows text can mean "move this way".
"""

import string

from constants import UP, DOWN, LEFT, RIGHT

COMMANDS = {
    "en": {
        "up": UP,
        "down": DOWN,
        "left": LEFT,
        "right": RIGHT,
    },
    "es": {
        "arriba": UP,
        "abajo": DOWN,
        "izquierda": LEFT,
        "derecha": RIGHT,
    },
}

_STRIP_CHARS = string.punctuation + "¡¿"

def parse_direction(text, language):
    """Return a direction constant (UP/DOWN/LEFT/RIGHT) if `text` contains
    a recognized command word for `language`, else None. Matches whole
    words only (so "cup" never matches "up"), returning the first command
    word found reading left to right."""

    if not text:
        return None
    commands = COMMANDS.get(language, {})
    if not commands:
        return None
    normalized = text.lower().translate(str.maketrans("","", _STRIP_CHARS))
    for word in normalized.split():
        if word in commands:
            return commands[word]
    return None

if __name__ == "__main__":
    tests = [
        ("Left.", "en", LEFT),
        ("Turn right!", "en", RIGHT),
        ("up up up", "en", UP),
        ("", "en", None),
        ("Arriba!", "es", UP),
        ("Ve a la izquierda", "es", LEFT),
        ("derecha", "es", RIGHT),
        ("¡Arriba!", "es", UP),
        ("Hola, hola, hola.", "es", None),
    ]
    for text, lang, expected in tests:
        result = parse_direction(text, lang)
        status = "OK" if result == expected else "FAIL"
        print(f"[{status}] parse_direction({text!r}, {lang!r}) -> {result} (expected {expected})")