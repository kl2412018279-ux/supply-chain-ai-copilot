import time

from src.cache.semantic_cache import InMemoryCacheBackend, SemanticCache, entity_fingerprint


def _cache(embedder, threshold=0.9, ttl=60):
    return SemanticCache(embedder, InMemoryCacheBackend(), threshold=threshold, ttl_seconds=ttl)


def test_hit_for_same_question_with_different_formatting(embedder):
    cache = _cache(embedder)
    cache.store("What is our on-time rate?", {"answer": "A-"})
    hit = cache.lookup("  what is our on-time rate  ")
    assert hit is not None and hit[0] == {"answer": "A-"}
    assert cache.stats()["hits"] == 1


def test_miss_for_unrelated_question(embedder):
    cache = _cache(embedder)
    cache.store("What is our on-time rate?", {"answer": "A-"})
    assert cache.lookup("Forecast demand for the laptop stand") is None


def test_different_entity_or_number_never_hits(embedder):
    cache = _cache(embedder, threshold=0.5)
    cache.store("Quote a 40ft palm oil container to Penang", {"answer": "RM4,000"})
    assert cache.lookup("Quote a 40ft palm oil container to Ipoh") is None
    assert cache.lookup("Quote a 20ft palm oil container to Penang") is None
    assert cache.lookup("Quote a 40ft chemicals container to Penang") is None
    assert cache.lookup("quote a 40ft CPO container to prai") is not None  # same entities


def test_entity_fingerprint():
    fp = entity_fingerprint("Ship 500 kg of resin 300km to KL by rail")
    assert fp == "300|500|cargo:plastics|dest:kuala-lumpur|rail"
    assert entity_fingerprint("no numbers") == ""
    assert entity_fingerprint("safest carrier to Penang") != entity_fingerprint(
        "cheapest carrier to Penang"
    )


def test_entries_expire(embedder):
    cache = _cache(embedder, ttl=0)
    cache.store("What is our on-time rate?", {"answer": "A-"})
    time.sleep(0.01)
    assert cache.lookup("What is our on-time rate?") is None


def test_in_memory_backend_is_bounded(embedder):
    backend = InMemoryCacheBackend(max_entries=3)
    for i in range(5):
        backend.put(embedder.embed_query(str(i)), {"i": i}, ttl=60)
    assert [p["i"] for _, p in backend.entries()] == [2, 3, 4]
