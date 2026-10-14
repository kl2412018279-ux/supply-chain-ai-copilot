"""Second-stage rerankers applied to hybrid-search candidates."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

from src.config import get_settings
from src.rag.schema import SearchResult
from src.rag.vector_store import tokenize

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]: ...


class LexicalReranker:
    """Cheap fallback: query-term coverage blended with the first-stage fusion score."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        terms = set(tokenize(query))
        if not results:
            return []
        max_score = max(r.score for r in results) or 1.0
        rescored = []
        for r in results:
            coverage = len(terms & set(tokenize(r.chunk.text))) / len(terms) if terms else 0.0
            score = 0.6 * coverage + 0.4 * (r.score / max_score)
            rescored.append(r.model_copy(update={"score": round(score, 4)}))
        return sorted(rescored, key=lambda r: r.score, reverse=True)[:top_k]


class FlashRankReranker:
    """Cross-encoder reranking with FlashRank (ms-marco TinyBERT, ~4MB, CPU-only)."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        from flashrank import Ranker  # noqa: PLC0415

        self._ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        from flashrank import RerankRequest  # noqa: PLC0415

        if not results:
            return []
        passages = [{"id": i, "text": r.chunk.text} for i, r in enumerate(results)]
        ranked = self._ranker.rerank(RerankRequest(query=query, passages=passages))
        return [
            results[p["id"]].model_copy(update={"score": round(float(p["score"]), 4)})
            for p in ranked[:top_k]
        ]


@lru_cache
def get_reranker() -> Reranker:
    if get_settings().reranker_backend == "flashrank":
        try:
            return FlashRankReranker()
        except Exception as exc:
            logger.warning("FlashRank unavailable (%s); using LexicalReranker.", exc)
    return LexicalReranker()
