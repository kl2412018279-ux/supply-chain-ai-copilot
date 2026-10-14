"""Retrieval evaluation: compare dense-only, keyword-only, hybrid and hybrid+rerank.

Each labelled question names a phrase that must appear in a retrieved chunk
(questions are written for ``data/raw/operations_manual.pdf``).
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

# (question, phrase that a correct chunk contains) - for the bundled operations manual.
EVAL_SET: list[tuple[str, str]] = [
    ("Which customs form is used for imports?", "Form K1"),
    ("What form do we use for transshipment between customs areas?", "Form K8"),
    ("What documents support a customs declaration?", "commercial invoice, packing list"),
    ("Can we pay import duties on behalf of a forwarder?", "signed indemnity"),
    ("How old can a safety data sheet be?", "no older than five years"),
    ("What licence do drivers need to carry dangerous goods?", "GDL"),
    ("How far apart must flammable liquids and oxidisers be stored?", "at least 3 metres"),
    (
        "How quickly must DG containers be collected from the terminal?",
        "within 24 hours of discharge",
    ),
    ("What temperature should crude palm oil be heated to?", "between 50 and 55 degrees"),
    ("Which certificate is required for palm oil exports?", "MSPO"),
    ("When is the palm oil peak season?", "September to November"),
    ("Do we handle plastic waste or scrap?", "does not handle plastic waste"),
    ("What is the maximum gross weight of a container on our trailers?", "25,000 kg"),
    ("Which destinations can use rail?", "Ipoh and Prai"),
    ("How are shipments to Sabah and Sarawak moved?", "coastal shipping"),
    ("What is the difference between demurrage and detention?", "Detention is charged"),
    ("How many free days of port storage are typical at Port Klang?", "3 days for port storage"),
    ("How much more does dangerous goods storage cost?", "1.8 times"),
    ("How long is a quotation valid?", "valid for 7 days"),
    ("What is the fleet emissions reduction target?", "30 percent by 2030"),
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
