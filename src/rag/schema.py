"""Shared data models for the retrieval layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    source: str
    chunk_index: int
    text: str
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source}#{self.chunk_index}"


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
