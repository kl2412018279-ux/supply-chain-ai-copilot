"""Golden questions for the copilot answer-quality eval.

One JSON object per line in ``data/evals/copilot_golden.jsonl``::

    {"id": "quote-01", "category": "quote", "question": "...",
     "expected_tools": ["quote_container_job"], "must_include": ["Pasir Gudang", "re:RM\\s?\\d"],
     "expect_blocked": false, "notes": "..."}

``must_include`` entries are case-insensitive substrings, or regular expressions when
prefixed with ``re:``. ``expected_tools`` is order-insensitive.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import PROJECT_ROOT

GOLDEN_PATH = PROJECT_ROOT / "data" / "evals" / "copilot_golden.jsonl"
REGEX_PREFIX = "re:"


class GoldenCase(BaseModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_tools: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    expect_blocked: bool = False
    notes: str = ""

    @field_validator("expected_tools")
    @classmethod
    def known_tools(cls, v: list[str]) -> list[str]:
        from src.agent.tools import TOOLS_BY_NAME  # noqa: PLC0415

        unknown = [t for t in v if t not in TOOLS_BY_NAME]
        if unknown:
            raise ValueError(f"unknown tool(s): {', '.join(unknown)}")
        return v

    @field_validator("must_include")
    @classmethod
    def valid_patterns(cls, v: list[str]) -> list[str]:
        for item in v:
            if item.startswith(REGEX_PREFIX):
                re.compile(item[len(REGEX_PREFIX) :])
        return v

    @model_validator(mode="after")
    def blocked_cases_expect_nothing(self) -> GoldenCase:
        if self.expect_blocked and (self.expected_tools or self.must_include):
            raise ValueError("a case expected to be blocked cannot expect tools or content")
        return self


def load_golden(path: str | Path | None = None, category: str | None = None) -> list[GoldenCase]:
    """Load (and validate) the golden set, optionally filtered to one category."""
    path = Path(path or GOLDEN_PATH)
    cases: list[GoldenCase] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            cases.append(GoldenCase(**json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path.name}:{lineno}: {exc}") from exc
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate case ids")
    if category:
        cases = [c for c in cases if c.category == category]
    return cases
