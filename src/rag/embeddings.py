"""Text embedders.

* ``SentenceTransformerEmbedder`` - real semantic embeddings (BAAI/bge-small-en-v1.5,
  384-d) computed locally on CPU; no API calls, $0.
* ``HashingEmbedder`` - deterministic feature-hashing fallback (words + char
  trigrams). Used offline, in CI, and when sentence-transformers is not installed.
"""

from __future__ import annotations

import logging
import re
import zlib
from functools import lru_cache
from typing import Protocol

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        grams = [f"w:{t}" for t in tokens]
        for t in tokens:
            padded = f"#{t}#"
            grams.extend(f"c:{padded[i : i + 3]}" for i in range(len(padded) - 2))
        for g in grams:
            h = zlib.crc32(g.encode())
            weight = 2.0 if g.startswith("w:") else 0.5
            vec[h % self.dim] += weight if (h >> 16) & 1 else -weight
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._embed(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        get_dim = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self._query_prefix + text])[0]


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.embedding_backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:  # ImportError, or model download failure offline
            logger.warning("sentence-transformers unavailable (%s); using HashingEmbedder.", exc)
    return HashingEmbedder(settings.embedding_dim)
