"""ISO 6346 container numbers: 3-letter owner code + equipment category (U/J/Z) +
6-digit serial + check digit. The check digit catches most single-character typos."""

from __future__ import annotations

import re

CONTAINER_RE = re.compile(r"\b([A-Z]{3}[UJZ])\s?(\d{6})\s?-?\s?(\d)\b")


def _letter_values() -> dict[str, int]:
    values, v = {}, 10
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if v % 11 == 0:  # multiples of 11 are skipped
            v += 1
        values[letter] = v
        v += 1
    return values


_VALUES = _letter_values()


def check_digit(first10: str) -> int:
    """Check digit for the first 10 characters (owner code + category + serial)."""
    if not re.fullmatch(r"[A-Z]{4}\d{6}", first10):
        raise ValueError(f"Not a container prefix: {first10!r}")
    total = sum((_VALUES[c] if c.isalpha() else int(c)) * (2**i) for i, c in enumerate(first10))
    return total % 11 % 10


def is_valid(number: str) -> bool:
    n = re.sub(r"[\s-]", "", number.upper())
    return bool(re.fullmatch(r"[A-Z]{3}[UJZ]\d{7}", n)) and check_digit(n[:10]) == int(n[10])


# OCR confusions: digits read as letters in the serial, letters read as digits in the owner.
FUZZY_RE = re.compile(r"\b([A-Z0158]{3}[UJZ])\s?([0-9OILSB]{6})\s?-?\s?([0-9OILSB])\b")
_TO_DIGIT = str.maketrans("OILSB", "01158")
_TO_LETTER = str.maketrans("0158", "OISB")


def find_containers(text: str, repair: bool = False) -> list[str]:
    """All container numbers in ``text``, normalised to 11 characters, in order.

    With ``repair`` (OCR text), near-misses such as ``QZXU26lOO42`` are read as
    ``QZXU2610042``, but only when the repaired number passes the check digit; a
    well-formed number with a wrong check digit is kept as-is so validation flags it.
    """
    seen: list[str] = []
    pattern = FUZZY_RE if repair else CONTAINER_RE
    for owner, serial, digit in pattern.findall(text.upper()):
        n = f"{owner}{serial}{digit}"
        if repair and not re.fullmatch(r"[A-Z]{3}[UJZ]\d{7}", n):
            fixed = owner.translate(_TO_LETTER) + f"{serial}{digit}".translate(_TO_DIGIT)
            if not is_valid(fixed):
                continue
            n = fixed
        if n not in seen:
            seen.append(n)
    return seen
