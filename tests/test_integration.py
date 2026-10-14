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

    keyword = pg_store.keyword_search("flexitank heating pads", 3)
    assert any("flexitank" in c.text.lower() for c, _ in keyword)

    results = HybridRetriever(pg_store, embedder, LexicalReranker()).search(
        "K8 transshipment form", k=2
    )
    assert "K8" in results[0].chunk.text
    assert pg_store.delete_source(source) == n


def test_postgres_volume_repository(volumes_df):
    if not DB_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    from src.forecasting.volume_repository import CsvVolumeRepository, PostgresVolumeRepository

    repo = PostgresVolumeRepository(DB_URL)
    repo.load_dataframe(volumes_df)
    summary = repo.summary("Palm Oil", start=None, end=None)
    assert summary.by_commodity[0].commodity == "palm-oil"

    csv_repo = CsvVolumeRepository.__new__(CsvVolumeRepository)
    csv_repo.df = volumes_df
    expected = csv_repo.summary("palm-oil").by_commodity[0].total_teu
    assert summary.by_commodity[0].total_teu == expected


def test_redis_semantic_cache(embedder):
    if not REDIS_URL:
        pytest.skip("TEST_REDIS_URL not set")
    backend = RedisCacheBackend(REDIS_URL)
    backend.clear()
    cache = SemanticCache(embedder, backend, threshold=0.9, ttl_seconds=60)
    cache.store("What is our ESG score?", {"answer": "A-"})
    assert cache.lookup("what is our esg score")[0] == {"answer": "A-"}
    assert cache.lookup("Forecast chemicals volume") is None
    backend.clear()
