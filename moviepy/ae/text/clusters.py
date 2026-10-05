"""Group common combining marks and emoji sequences for title reveals.

This compact grouping covers marks, variation selectors, ZWJ sequences and
regional-indicator flags. It is not a full Unicode UAX #29 implementation.
"""

import unicodedata


def clusters(text):
    """Return reveal units without splitting common composed characters."""
    result = []
    regional = 0
    for character in text:
        code = ord(character)
        flag = 0x1F1E6 <= code <= 0x1F1FF
        continuation = unicodedata.category(character).startswith("M")
        continuation |= character == "\u200d" or 0x1F3FB <= code <= 0x1F3FF
        continuation |= bool(result and result[-1].endswith("\u200d"))
        continuation |= flag and regional % 2 == 1
        continuation |= character == "\n" and bool(result and result[-1] == "\r")
        if result and continuation:
            result[-1] += character
        else:
            result.append(character)
        regional = regional + 1 if flag else 0
    return tuple(result)
