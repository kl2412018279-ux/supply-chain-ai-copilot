"""Hybrid retrieval: dense + keyword search fused with Reciprocal Rank Fusion, then reranked.

Dense vectors capture meaning ("carbon reduction goals"), keyword search catches
exact tokens that embeddings blur (SKU codes, clause numbers like "4.2").
"""

from __future__ import annotations

from functools import lru_cache

from src.rag.embeddings import Embedder, get_embedder
from src.rag.reranker import Reranker, get_reranker
from src.rag.schema import SearchResult
from src.rag.vector_store import VectorStore, get_vector_store

RRF_K = 60


def reciprocal_rank_fusion(dense: list, sparse: list, k: int = RRF_K) -> list[SearchResult]:
    fused: dict[str, SearchResult] = {}
    for kind, results in (("vector_rank", dense), ("keyword_rank", sparse)):
        for rank, (chunk, _score) in enumerate(results, start=1):
            entry = fused.get(chunk.key) or SearchResult(chunk=chunk, score=0.0)
            entry.score += 1.0 / (k + rank)
            setattr(entry, kind, rank)
            fused[chunk.key] = entry
    return sorted(fused.values(), key=lambda r: r.score, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        candidates: int = 20,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, k: int = 4, rerank: bool = True) -> list[SearchResult]:
        dense = self.store.vector_search(self.embedder.embed_query(query), self.candidates)
        sparse = self.store.keyword_search(query, self.candidates)
        fused = reciprocal_rank_fusion(dense, sparse)
        if rerank and self.reranker is not None:
            return self.reranker.rerank(query, fused[: self.candidates], k)
        return fused[:k]


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(get_vector_store(), get_embedder(), get_reranker())
