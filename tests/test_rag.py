import importlib.util

import numpy as np
import pytest

from src.rag.ingest_pdf import chunk_text, extract_pages, ingest_bytes
from src.rag.reranker import LexicalReranker
from src.rag.retriever import HybridRetriever, reciprocal_rank_fusion
from src.rag.schema import Chunk
from src.rag.vector_store import InMemoryVectorStore
from tests.conftest import ROOT, SAMPLE_PDF


def _make_pdf(text: str) -> bytes:
    spec = importlib.util.spec_from_file_location("mkpdf", ROOT / "scripts" / "make_sample_pdf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.text_to_pdf(text)


def test_chunk_text_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} about logistics." for i in range(200))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # consecutive chunks share some words thanks to the overlap
    assert set(chunks[0].split()[-3:]) & set(chunks[1].split()[:12])


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)


def test_extract_pages_from_pdf():
    pages = extract_pages(_make_pdf("Hello ESG world.\nScope 3 emissions."), "x.pdf")
    assert pages[0][0] == 1
    assert "Scope 3" in pages[0][1]


def test_extract_pages_rejects_unknown_type():
    with pytest.raises(ValueError):
        extract_pages(b"data", "file.docx")


def test_reciprocal_rank_fusion_rewards_agreement():
    a, b, c = (Chunk(source="d", chunk_index=i, text=t) for i, t in enumerate("abc"))
    fused = reciprocal_rank_fusion([(a, 0.9), (b, 0.8)], [(b, 5.0), (c, 1.0)])
    assert fused[0].chunk.key == b.key  # in both lists
    assert fused[0].vector_rank == 2 and fused[0].keyword_rank == 1


def test_reingesting_replaces_chunks(embedder):
    store = InMemoryVectorStore()
    ingest_bytes(b"Rail freight policy. " * 50, "policy.txt", store, embedder)
    first = store.count()
    ingest_bytes(b"Rail freight policy. " * 50, "policy.txt", store, embedder)
    assert store.count() == first
    assert store.delete_source("policy.txt") == first
    assert store.count() == 0


def test_hybrid_search_finds_exact_policy_clause(embedder):
    store = InMemoryVectorStore()
    ingest_bytes(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name, store, embedder)
    retriever = HybridRetriever(store, embedder, LexicalReranker())
    results = retriever.search("K8 transshipment form", k=3)
    assert results
    assert "K8" in results[0].chunk.text
    assert results[0].chunk.page is not None


def test_in_memory_store_validates_lengths(embedder):
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.add([Chunk(source="s", chunk_index=0, text="t")], np.zeros((2, 384)))


def test_retrieval_evaluation_hybrid_beats_dense(embedder):
    from src.rag.evaluate import evaluate

    store = InMemoryVectorStore()
    ingest_bytes(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name, store, embedder)
    results = {r.method: r for r in evaluate(store, embedder, LexicalReranker(), k=3)}
    assert len(results) == 4
    assert results["hybrid (RRF)"].hit_rate >= results["dense only"].hit_rate
    assert results["hybrid + rerank"].mrr > 0.5
