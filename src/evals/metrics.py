"""Answer-quality metrics for the copilot eval.

- **Tool selection**: did the agent call the tools the question needs?
- **Numeric faithfulness** (groundedness): every RM amount, percentage and number with a
  unit in the answer must be backed by a number the agent's tools returned (or the user
  typed), after normalising formatting and rounding to the precision shown. This is the
  check that catches invented prices, rates and volumes.
- **must_include**: required facts / labels are present in the answer.
- Small helpers for latency percentiles.

Everything here is pure Python so it can be unit-tested without the agent.
"""

from __future__ import annotations

import bisect
import itertools
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.evals.dataset import REGEX_PREFIX

# A number as written in prose: optional sign, optional thousands separators, decimals.
_DIGITS = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
# Sign only when it is not part of a word or a date ("2026-12-10", "Euro-5").
_SIGNED = r"(?:(?<![\w.,])(?P<sign>[-−])|(?<![\w.,\-−]))" + _DIGITS

RM_RE = re.compile(
    r"\b(?:RM|MYR)\s?(?P<sign>[-−])?" + _DIGITS + r"(?:\s?(?P<suffix>k|m|mil|million)\b)?"
)
PERCENT_RE = re.compile(_SIGNED + r"\s?%")
UNIT_RE = re.compile(
    _SIGNED + r"\s?(?P<unit>TEU|km|kg|tonnes?|tons?|L|litres?|liters?|mm|h|hrs?|hours?|d|days?|"
    r"weeks?|containers?|boxes|pallets?|deliveries|jobs)\b",
    re.IGNORECASE,
)
# Any number inside tool text (snippets, labels, explanations, even field names such as
# ``due_within_48h``): deliberately loose, because it only widens what counts as support.
_LOOSE_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_SUFFIX = {"k": 1e3, "m": 1e6, "mil": 1e6, "million": 1e6}
# Bound on pairwise derivations per dict so a wide tool output stays cheap to index.
_MAX_PAIRWISE = 40


@dataclass(frozen=True)
class NumericClaim:
    """A number the answer states, with the precision it was shown at."""

    text: str
    value: float
    decimals: int
    kind: str  # "rm" | "percent" | "unit"
    scale: float = 1.0

    @property
    def tolerance(self) -> float:
        """Half a unit of the last shown digit: RM1,234.50 matches 1234.495..1234.505."""
        return 0.5 * 10 ** (-self.decimals) * self.scale + 1e-9


def parse_number(text: str) -> tuple[float, int]:
    """``"1,234.50"`` -> ``(1234.5, 2)``: value and number of decimals shown."""
    raw = text.replace(",", "").replace("−", "-").strip()
    decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
    return float(raw), decimals


def normalise_amount(text: str) -> float:
    """Turn a formatted figure into a float: ``"RM1,234.50"`` / ``"RM 1.2m"`` / ``"91.2%"``."""
    m = RM_RE.search(text) or PERCENT_RE.search(text) or UNIT_RE.search(text)
    if m is None:
        m = re.search(_SIGNED, text)
    if m is None:
        raise ValueError(f"no number in {text!r}")
    value, _ = parse_number(m.group("num"))
    if m.groupdict().get("sign"):
        value = -value
    suffix = (m.groupdict().get("suffix") or "").lower()
    return value * _SUFFIX.get(suffix, 1.0)


def extract_claims(answer: str) -> list[NumericClaim]:
    """RM amounts, percentages and numbers with units stated in ``answer``.

    Bare numbers (ranks, years, form numbers, "fit 87/100") are not claims: they are
    rarely the figures a user acts on, and flagging them would drown the signal.
    """
    claims: list[NumericClaim] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in (("rm", RM_RE), ("percent", PERCENT_RE), ("unit", UNIT_RE)):
        for m in pattern.finditer(answer):
            start, end = m.span()
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            value, decimals = parse_number(m.group("num"))
            if m.groupdict().get("sign"):
                value = -value
            scale = _SUFFIX.get((m.groupdict().get("suffix") or "").lower(), 1.0)
            claims.append(NumericClaim(m.group(0).strip(), value * scale, decimals, kind, scale))
    return claims


def _numbers_in_text(text: str) -> list[float]:
    out: list[float] = []
    for m in _LOOSE_NUMBER_RE.finditer(text):
        raw = m.group(0)
        out.append(float(raw.replace(",", "")))
        if "," in raw:  # "1,2" in a list could also be two numbers
            out.extend(float(p) for p in re.split(r"[,.]", raw) if p)
    return out


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def collect_numbers(obj: Any) -> list[float]:
    """Every number in a tool payload: numeric values, numbers inside strings and keys."""
    out: list[float] = []
    if _is_number(obj):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(_numbers_in_text(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_numbers_in_text(str(k)))
            out.extend(collect_numbers(v))
    elif isinstance(obj, list | tuple):
        for v in obj:
            out.extend(collect_numbers(v))
    return out


def derived_numbers(obj: Any) -> list[float]:
    """Simple arithmetic an honest answer may show without the tool returning it.

    - sums / differences of two numeric fields of the same record (revenue - cost), for
      magnitudes of at least 1 (differences of rates stored as fractions only add noise);
    - totals of a numeric column over a list of records, overall and grouped by any text
      column (quote line items summed per cost category).
    """
    out: list[float] = []
    if isinstance(obj, dict):
        nums = [float(v) for v in obj.values() if _is_number(v) and abs(v) >= 1]
        nums = nums[:_MAX_PAIRWISE]
        for a, b in itertools.combinations(nums, 2):
            out.extend((a + b, a - b, b - a))
        for v in obj.values():
            out.extend(derived_numbers(v))
    elif isinstance(obj, list | tuple):
        rows = [r for r in obj if isinstance(r, dict)]
        if rows:
            num_keys = {k for r in rows for k, v in r.items() if _is_number(v)}
            text_keys = {k for r in rows for k, v in r.items() if isinstance(v, str)}
            for n in num_keys:
                out.append(sum(float(r[n]) for r in rows if _is_number(r.get(n))))
                for t in text_keys:
                    groups: dict[str, float] = {}
                    for r in rows:
                        if _is_number(r.get(n)) and isinstance(r.get(t), str):
                            groups[r[t]] = groups.get(r[t], 0.0) + float(r[n])
                    out.extend(groups.values())
        for v in obj:
            out.extend(derived_numbers(v))
    return out


def _sorted_abs(values: Iterable[float]) -> list[float]:
    # Sign is presentation: "2 day(s) over" is backed by days_left=-2.
    return sorted({abs(v) for v in values})


def _contains(values: list[float], value: float, tolerance: float) -> bool:
    lo = bisect.bisect_left(values, value - tolerance)
    return lo < len(values) and values[lo] <= value + tolerance


class NumberPool:
    """Sorted index of supporting numbers with tolerance lookups.

    ``raw`` are numbers the tools returned; ``derived`` are the simple aggregates from
    :func:`derived_numbers`. A percentage may also match a raw fraction (0.912 -> 91.2%).
    """

    def __init__(self, raw: Iterable[float], derived: Iterable[float] = ()):
        self.raw = _sorted_abs(raw)
        self.derived = _sorted_abs(derived)

    def __len__(self) -> int:
        return len(self.raw) + len(self.derived)

    def supports(self, claim: NumericClaim) -> bool:
        value = abs(claim.value)
        if _contains(self.raw, value, claim.tolerance):
            return True
        if _contains(self.derived, value, claim.tolerance):
            return True
        # Tools store rates as fractions: on_time_rate 0.912 is shown as "91.2%".
        return claim.kind == "percent" and _contains(self.raw, value / 100, claim.tolerance / 100)


def build_pool(tool_payloads: Iterable[Any], question: str = "") -> NumberPool:
    """Numbers the agent was given: tool outputs and arguments, plus the user's question."""
    raw: list[float] = _numbers_in_text(question)
    derived: list[float] = []
    for payload in tool_payloads:
        raw.extend(collect_numbers(payload))
        derived.extend(derived_numbers(payload))
    return NumberPool(raw, derived)


def unsupported_numbers(answer: str, tool_payloads: Iterable[Any], question: str = "") -> list[str]:
    """Figures in ``answer`` that no tool output (or the question) supports."""
    pool = build_pool(tool_payloads, question)
    return [c.text for c in extract_claims(answer) if not pool.supports(c)]


@dataclass(frozen=True)
class ToolScore:
    precision: float
    recall: float
    passed: bool  # every expected tool was called
    missing: list[str]
    unexpected: list[str]


def tool_score(expected: Iterable[str], called: Iterable[str]) -> ToolScore:
    """Order-insensitive set comparison of expected vs called tool names.

    A tool called several times (one forecast per commodity) counts once. With nothing
    expected and nothing called, precision and recall are both 1.
    """
    exp, got = set(expected), set(called)
    hit = exp & got
    recall = len(hit) / len(exp) if exp else 1.0
    precision = len(hit) / len(got) if got else (1.0 if not exp else 0.0)
    return ToolScore(
        precision=round(precision, 4),
        recall=round(recall, 4),
        passed=exp <= got,
        missing=sorted(exp - got),
        unexpected=sorted(got - exp),
    )


def missing_includes(answer: str, patterns: Iterable[str]) -> list[str]:
    """``must_include`` entries not found in ``answer`` (substring or ``re:`` regex)."""
    missing = []
    lowered = answer.lower()
    for p in patterns:
        if p.startswith(REGEX_PREFIX):
            found = re.search(p[len(REGEX_PREFIX) :], answer, re.IGNORECASE) is not None
        else:
            found = p.lower() in lowered
        if not found:
            missing.append(p)
    return missing


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in 0..100); 0.0 for no values."""
    data = sorted(values)
    if not data:
        return 0.0
    k = (len(data) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)
