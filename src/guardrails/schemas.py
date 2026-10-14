"""Pydantic schemas that every copilot response must satisfy."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.guardrails.validators import GuardrailCategory


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: Any = None


class Source(BaseModel):
    source: str
    page: int | None = None
    snippet: str
    score: float


class CopilotResponse(BaseModel):
    answer: str = Field(min_length=1, max_length=8000)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    cached: bool = False
    blocked: bool = False
    guardrail: GuardrailCategory = GuardrailCategory.ok
    agent: str = "rule-based"
    latency_ms: float = 0.0

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("answer must not be blank")
        return v
