import time

from src.cache.semantic_cache import InMemoryCacheBackend, SemanticCache, entity_fingerprint


def _cache(embedder, threshold=0.9, ttl=60):
    return SemanticCache(embedder, InMemoryCacheBackend(), threshold=threshold, ttl_seconds=ttl)


def test_hit_for_same_question_with_different_formatting(embedder):
    cache = _cache(embedder)
    cache.store("What is our ESG score?", {"answer": "A-"})
    hit = cache.lookup("  what is our ESG score  ")
    assert hit is not None and hit[0] == {"answer": "A-"}
    assert cache.stats()["hits"] == 1


def test_miss_for_unrelated_question(embedder):
    cache = _cache(embedder)
    cache.store("What is our ESG score?", {"answer": "A-"})
    assert cache.lookup("Forecast demand for the laptop stand") is None


def test_different_sku_or_number_never_hits(embedder):
    cache = _cache(embedder, threshold=0.5)
    cache.store("Forecast demand for SKU-101 for 14 days", {"answer": "650"})
    assert cache.lookup("Forecast demand for SKU-102 for 14 days") is None
    assert cache.lookup("Forecast demand for SKU-101 for 30 days") is None
    assert cache.lookup("forecast demand for sku 101 for 14 days") is not None


def test_entity_fingerprint():
    assert entity_fingerprint("Ship 500 kg 300km for SKU_104") == "300|500|sku-104"
    assert entity_fingerprint("no numbers") == ""


def test_entries_expire(embedder):
    cache = _cache(embedder, ttl=0)
    cache.store("What is our ESG score?", {"answer": "A-"})
    time.sleep(0.01)
    assert cache.lookup("What is our ESG score?") is None


def test_in_memory_backend_is_bounded(embedder):
    backend = InMemoryCacheBackend(max_entries=3)
    for i in range(5):
        backend.put(embedder.embed_query(str(i)), {"i": i}, ttl=60)
    assert [p["i"] for _, p in backend.entries()] == [2, 3, 4]
