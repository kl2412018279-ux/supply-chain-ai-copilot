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


def find_containers(text: str) -> list[str]:
    """All container numbers in ``text``, normalised to 11 characters, in order."""
    seen: list[str] = []
    for owner, serial, digit in CONTAINER_RE.findall(text.upper()):
        n = f"{owner}{serial}{digit}"
        if n not in seen:
            seen.append(n)
    return seen
