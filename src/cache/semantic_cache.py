"""Semantic cache in front of the LLM.

A question is embedded and compared (cosine similarity) against previously answered
questions; above ``threshold`` the stored answer is returned without an LLM call.
Backends: Redis (shared across API replicas, TTL-based expiry) or in-process memory.

Paraphrases embed close together, but so do "quote 40ft to Penang" and "quote 40ft to Ipoh".
To avoid serving a wrong answer, a hit additionally requires the *entity fingerprint*
(every number, destination, commodity and transport mode in the question) to match exactly.

For large caches swap the linear scan for a Redis Stack / RediSearch vector index.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from src.config import get_settings
from src.rag.embeddings import Embedder, get_embedder

logger = logging.getLogger(__name__)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z]+")


@lru_cache
def _entity_vocabulary() -> dict[str, str]:
    """Words that change the answer: destinations, commodities, modes and ports."""
    from src.costing.rates import get_rate_card  # noqa: PLC0415
    from src.forecasting.data import COMMODITIES, COMMODITY_ALIASES  # noqa: PLC0415

    vocab: dict[str, str] = {}
    try:
        for key, dest in get_rate_card().destinations.items():
            for phrase in [key.replace("-", " "), dest["label"], *dest.get("aliases", [])]:
                for word in _WORD_RE.findall(phrase.lower()):
                    vocab[word] = f"dest:{key}"
    except (OSError, KeyError, ValueError):
        logger.warning("Rate card unavailable; cache fingerprints ignore destinations.")
    for commodity in COMMODITIES:
        for word in _WORD_RE.findall(commodity):
            vocab[word] = f"cargo:{commodity}"
    for alias, commodity in COMMODITY_ALIASES.items():
        if " " not in alias:
            vocab[alias] = f"cargo:{commodity}"
    try:
        for key, carrier in get_rate_card().data["carriers"].items():
            if key.startswith("_"):
                continue
            for word in _WORD_RE.findall(carrier["label"].lower()):
                if word not in {"transport", "haulage", "shipping", "line", "own", "fleet"}:
                    vocab.setdefault(word, f"carrier:{key}")
    except (OSError, KeyError, ValueError):
        pass
    months = ["january", "february", "march", "april", "may", "june", "july", "august",
              "september", "october", "november", "december"]  # fmt: skip
    for i, month in enumerate(months, start=1):
        vocab[month] = f"month:{i}"
    for word in ("truck", "rail", "sea", "air", "westport", "northport", "inspection",
                 "cheapest", "fastest", "safest", "reliable"):  # fmt: skip
        vocab[word] = word
    for generic in ("oil", "port", "the", "goods"):  # too ambiguous on their own
        vocab.pop(generic, None)
    return vocab


def entity_fingerprint(query: str) -> str:
    """Canonical string of the numbers, places, cargo types and modes in a query.

    Two questions only share a cached answer when these all match, so "quote 40ft to
    Penang" never returns the answer for "quote 40ft to Ipoh".
    """
    vocab = _entity_vocabulary()
    text = query.lower()
    entities = set(_NUMBER_RE.findall(text))
    entities |= {vocab[w] for w in _WORD_RE.findall(text) if w in vocab}
    return "|".join(sorted(entities))


class CacheBackend(Protocol):
    def entries(self) -> list[tuple[np.ndarray, dict[str, Any]]]: ...

    def put(self, embedding: np.ndarray, payload: dict[str, Any], ttl: int) -> None: ...

    def clear(self) -> None: ...


class InMemoryCacheBackend:
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[np.ndarray, dict[str, Any], float]] = OrderedDict()
        self._lock = threading.Lock()

    def entries(self):
        now = time.time()
        with self._lock:
            expired = [k for k, (_, _, exp) in self._data.items() if exp < now]
            for k in expired:
                del self._data[k]
            return [(e, p) for e, p, _ in self._data.values()]

    def put(self, embedding, payload, ttl):
        with self._lock:
            self._data[uuid.uuid4().hex] = (embedding, payload, time.time() + ttl)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


class RedisCacheBackend:
    PREFIX = "semcache:"

    def __init__(self, url: str):
        import redis  # noqa: PLC0415

        self._redis = redis.Redis.from_url(url)
        self._redis.ping()

    def entries(self):
        out = []
        keys = list(self._redis.scan_iter(f"{self.PREFIX}*", count=500))
        if not keys:
            return out
        pipe = self._redis.pipeline()
        for k in keys:
            pipe.hmget(k, "embedding", "payload")
        for emb, payload in pipe.execute():
            if emb is not None and payload is not None:
                out.append((np.frombuffer(emb, dtype=np.float32), json.loads(payload)))
        return out

    def put(self, embedding, payload, ttl):
        key = f"{self.PREFIX}{uuid.uuid4().hex}"
        self._redis.hset(
            key,
            mapping={
                "embedding": np.asarray(embedding, dtype=np.float32).tobytes(),
                "payload": json.dumps(payload, default=str),
            },
        )
        self._redis.expire(key, ttl)

    def clear(self):
        for k in self._redis.scan_iter(f"{self.PREFIX}*", count=500):
            self._redis.delete(k)


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        backend: CacheBackend,
        threshold: float = 0.92,
        ttl_seconds: int = 86400,
    ):
        self.embedder = embedder
        self.backend = backend
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.lower().split()).rstrip("?!. ")

    def lookup(self, query: str) -> tuple[dict[str, Any], float] | None:
        q = self.embedder.embed_query(self._normalize_query(query))
        fingerprint = entity_fingerprint(query)
        best: tuple[dict[str, Any], float] | None = None
        for emb, entry in self.backend.entries():
            if entry.get("fingerprint") != fingerprint:
                continue
            payload = entry["response"]
            sim = float(np.dot(q, emb))
            if sim >= self.threshold and (best is None or sim > best[1]):
                best = (payload, sim)
        if best:
            self.hits += 1
        else:
            self.misses += 1
        return best

    def store(self, query: str, payload: dict[str, Any]) -> None:
        emb = self.embedder.embed_query(self._normalize_query(query))
        entry = {"fingerprint": entity_fingerprint(query), "response": payload}
        self.backend.put(emb, entry, self.ttl_seconds)

    def clear(self) -> None:
        self.backend.clear()

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


@lru_cache
def get_semantic_cache() -> SemanticCache:
    settings = get_settings()
    backend: CacheBackend = InMemoryCacheBackend()
    if settings.redis_url:
        try:
            backend = RedisCacheBackend(settings.redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s); using in-memory semantic cache.", exc)
    return SemanticCache(
        get_embedder(),
        backend,
        threshold=settings.cache_similarity_threshold,
        ttl_seconds=settings.cache_ttl_seconds,
    )
