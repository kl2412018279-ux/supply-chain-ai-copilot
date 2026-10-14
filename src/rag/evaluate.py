"""Retrieval evaluation: compare dense-only, keyword-only, hybrid and hybrid+rerank.

Each labelled question names a phrase that must appear in a retrieved chunk.
Reports hit@k and MRR so retrieval changes are measured, not guessed.

    python -m src.rag.evaluate --k 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.config import get_settings
from src.rag.embeddings import Embedder, get_embedder
from src.rag.ingest_pdf import ingest_directory
from src.rag.reranker import Reranker, get_reranker
from src.rag.retriever import HybridRetriever, reciprocal_rank_fusion
from src.rag.vector_store import InMemoryVectorStore

# (question, phrase that a correct chunk contains) - for the bundled sample policy.
EVAL_SET: list[tuple[str, str]] = [
    ("Can lithium battery packs go on passenger planes?", "UN3480"),
    ("Which product is classified as dangerous goods?", "UN3480"),
    ("When is air freight allowed?", "Regional Logistics Director"),
    ("What is the net-zero target year?", "2045"),
    ("How much must scope 1 and 2 emissions fall by 2030?", "50 percent by 2030"),
    ("What is the emission factor for ocean shipping?", "ocean freight 0.016"),
    ("Long shipments in Europe should use which transport?", "intermodal"),
    ("What share of spend must be with suppliers that have SBTi targets?", "80 percent"),
    ("Renewable electricity requirement for suppliers", "renewable electricity"),
    ("Which labour code of conduct must suppliers follow?", "Responsible Business Alliance"),
    ("Maximum weekly working hours for supplier staff", "60 hours"),
    ("Packaging plastic-free requirement", "plastic-free"),
    ("How much recycled aluminium is in SKU-105?", "post-consumer recycled aluminium"),
    ("Take-back programme collection target for monitors", "40 percent collection"),
    ("How is safety stock calculated?", "1.65 standard deviations"),
    ("Excess inventory write-off escalation threshold", "2 percent"),
    ("How often does the board review sustainability progress?", "every quarter"),
    ("Which disclosure standards are used for ESG reporting?", "IFRS S2"),
    ("How can employees report policy breaches?", "ethics hotline"),
    ("Who reviews ML demand forecasts before purchase orders?", "category planners"),
]


@dataclass
class EvalResult:
    method: str
    hit_rate: float
    mrr: float


def _score(ranked_texts: list[list[str]], k: int) -> tuple[float, float]:
    hits, rr = 0, 0.0
    for (_, phrase), texts in zip(EVAL_SET, ranked_texts, strict=True):
        for rank, text in enumerate(texts[:k], start=1):
            if phrase.lower() in " ".join(text.split()).lower():
                hits += 1
                rr += 1 / rank
                break
    return hits / len(EVAL_SET), rr / len(EVAL_SET)


def evaluate(
    store: InMemoryVectorStore, embedder: Embedder, reranker: Reranker, k: int = 3
) -> list[EvalResult]:
    retriever = HybridRetriever(store, embedder, reranker)

    def dense(q: str) -> list[str]:
        return [c.text for c, _ in store.vector_search(embedder.embed_query(q), k)]

    def keyword(q: str) -> list[str]:
        return [c.text for c, _ in store.keyword_search(q, k)]

    def hybrid(q: str) -> list[str]:
        fused = reciprocal_rank_fusion(
            store.vector_search(embedder.embed_query(q), 20), store.keyword_search(q, 20)
        )
        return [r.chunk.text for r in fused]

    def hybrid_rerank(q: str) -> list[str]:
        return [r.chunk.text for r in retriever.search(q, k=k)]

    methods = {
        "dense only": dense,
        "keyword only (BM25)": keyword,
        "hybrid (RRF)": hybrid,
        "hybrid + rerank": hybrid_rerank,
    }
    results = []
    for name, fn in methods.items():
        hit, mrr = _score([fn(q) for q, _ in EVAL_SET], k)
        results.append(EvalResult(name, round(hit, 3), round(mrr, 3)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    store = InMemoryVectorStore()
    embedder = get_embedder()
    ingest_directory(get_settings().documents_dir, store, embedder)
    reranker = get_reranker()
    print(f"Embedder: {type(embedder).__name__} | reranker: {type(reranker).__name__}")
    print(f"{'method':<22} hit@{args.k:<4} MRR@{args.k}")
    for r in evaluate(store, embedder, reranker, args.k):
        print(f"{r.method:<22} {r.hit_rate:<8.2f} {r.mrr:.2f}")


if __name__ == "__main__":
    main()
