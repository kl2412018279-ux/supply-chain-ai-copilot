"""Integration tests against real PostgreSQL + pgvector and Redis.

Run with e.g.::

    TEST_DATABASE_URL=postgresql://admin:password123@localhost:5432/supply_chain \
    TEST_REDIS_URL=redis://localhost:6379/15 pytest -m integration
"""

import os
import uuid

import pytest

from src.cache.semantic_cache import RedisCacheBackend, SemanticCache
from src.rag.ingest_pdf import ingest_bytes
from src.rag.reranker import LexicalReranker
from src.rag.retriever import HybridRetriever
from tests.conftest import SAMPLE_PDF

DB_URL = os.getenv("TEST_DATABASE_URL", "")
REDIS_URL = os.getenv("TEST_REDIS_URL", "")
pytestmark = pytest.mark.integration


@pytest.fixture
def pg_store():
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    from src.rag.vector_store import PgVectorStore

    store = PgVectorStore(DB_URL, dim=384)
    yield store
    for source in store.sources():
        if source.startswith("it-"):
            store.delete_source(source)


def test_pgvector_hybrid_search(pg_store, embedder):
    source = f"it-{uuid.uuid4().hex[:8]}.pdf"
    n = ingest_bytes(SAMPLE_PDF.read_bytes(), source, pg_store, embedder)
    assert pg_store.sources()[source] == n

    # re-ingesting is idempotent
    ingest_bytes(SAMPLE_PDF.read_bytes(), source, pg_store, embedder)
    assert pg_store.sources()[source] == n

    keyword = pg_store.keyword_search("UN3480 lithium", 3)
    assert any("UN3480" in c.text for c, _ in keyword)

    results = HybridRetriever(pg_store, embedder, LexicalReranker()).search(
        "dangerous goods UN3480", k=2
    )
    assert "UN3480" in results[0].chunk.text
    assert pg_store.delete_source(source) == n


def test_postgres_sales_repository(sales_df):
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    from src.forecasting.sales_repository import CsvSalesRepository, PostgresSalesRepository

    repo = PostgresSalesRepository(DB_URL)
    repo.load_dataframe(sales_df)
    summary = repo.summary("sku-101", start=None, end=None)
    assert summary.by_sku[0].sku == "SKU-101"

    csv_repo = CsvSalesRepository.__new__(CsvSalesRepository)
    csv_repo.df = sales_df
    assert summary.by_sku[0].total_units == csv_repo.summary("SKU-101").by_sku[0].total_units


def test_redis_semantic_cache(embedder):
    if not REDIS_URL:
        pytest.skip("TEST_REDIS_URL not set")
    backend = RedisCacheBackend(REDIS_URL)
    backend.clear()
    cache = SemanticCache(embedder, backend, threshold=0.9, ttl_seconds=60)
    cache.store("What is our ESG score?", {"answer": "A-"})
    assert cache.lookup("what is our esg score")[0] == {"answer": "A-"}
    assert cache.lookup("Forecast demand for SKU-101") is None
    backend.clear()
