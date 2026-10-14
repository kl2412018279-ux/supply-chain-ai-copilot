"""Vector stores with both dense (embedding) and sparse (keyword) search.

``PgVectorStore`` keeps embeddings *and* a full-text ``tsvector`` in a single
PostgreSQL table (pgvector HNSW index + GIN index). ``InMemoryVectorStore`` is a
drop-in replacement using NumPy cosine similarity and ``rank_bm25``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from functools import lru_cache
from typing import Protocol

import numpy as np

from src.config import get_settings
from src.rag.schema import Chunk

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> int: ...

    def vector_search(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...

    def delete_source(self, source: str) -> int: ...

    def sources(self) -> dict[str, int]: ...

    def count(self) -> int: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._bm25 = None
        self._lock = threading.Lock()

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return 0
        with self._lock:
            self._chunks.extend(chunks)
            emb = np.asarray(embeddings, dtype=np.float32)
            self._embeddings = (
                emb if self._embeddings is None else np.vstack([self._embeddings, emb])
            )
            self._bm25 = None
        return len(chunks)

    def vector_search(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if self._embeddings is None or not self._chunks:
            return []
        sims = self._embeddings @ np.asarray(embedding, dtype=np.float32)
        top = np.argsort(-sims)[:k]
        return [(self._chunks[i], float(sims[i])) for i in top]

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not self._chunks or not terms:
            return []
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415

            self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks])
        scores = self._bm25.get_scores(terms)
        top = np.argsort(-scores)[:k]
        return [(self._chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    def delete_source(self, source: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.source != source]
            removed = len(self._chunks) - len(keep)
            if removed:
                self._chunks = [self._chunks[i] for i in keep]
                self._embeddings = self._embeddings[keep] if keep else None
                self._bm25 = None
        return removed

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self._chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def count(self) -> int:
        return len(self._chunks)


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS document_chunks (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page        INTEGER,
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding   vector({dim}) NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, chunk_index)
);
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS document_chunks_tsv_idx
    ON document_chunks USING gin (content_tsv);
"""


class PgVectorStore:
    def __init__(self, dsn: str, dim: int = 384):
        import psycopg  # noqa: PLC0415
        from pgvector.psycopg import register_vector  # noqa: PLC0415

        self.dim = dim
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._lock = threading.Lock()
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            self._conn.execute(SCHEMA_SQL.format(dim=self.dim))

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        rows = [
            (c.source, c.chunk_index, c.page, c.text, json.dumps(c.metadata), np.asarray(e))
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self._lock, self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO document_chunks
                    (source, chunk_index, page, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (source, chunk_index) DO UPDATE
                SET page = EXCLUDED.page, content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding
                """,
                rows,
            )
        return len(rows)

    @staticmethod
    def _to_chunk(row: tuple) -> Chunk:
        source, chunk_index, page, content, metadata = row[:5]
        return Chunk(
            source=source, chunk_index=chunk_index, page=page, text=content, metadata=metadata or {}
        )

    def vector_search(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT source, chunk_index, page, content, metadata,
                       1 - (embedding <=> %(q)s) AS similarity
                FROM document_chunks ORDER BY embedding <=> %(q)s LIMIT %(k)s
                """,
                {"q": np.asarray(embedding, dtype=np.float32), "k": k},
            ).fetchall()
        return [(self._to_chunk(r), float(r[5])) for r in rows]

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        # OR-semantics so natural-language questions still match; ts_rank_cd orders them.
        tsquery = " | ".join(dict.fromkeys(terms))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT source, chunk_index, page, content, metadata,
                       ts_rank_cd(content_tsv, q) AS rank
                FROM document_chunks, to_tsquery('english', %(q)s) AS q
                WHERE content_tsv @@ q ORDER BY rank DESC LIMIT %(k)s
                """,
                {"q": tsquery, "k": k},
            ).fetchall()
        return [(self._to_chunk(r), float(r[5])) for r in rows]

    def delete_source(self, source: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM document_chunks WHERE source = %s", (source,))
            return cur.rowcount

    def sources(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT source, count(*) FROM document_chunks GROUP BY source ORDER BY source"
            ).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM document_chunks").fetchone()[0])


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.database_url:
        try:
            from src.rag.embeddings import get_embedder  # noqa: PLC0415

            return PgVectorStore(settings.database_url, dim=get_embedder().dim)
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s); using in-memory vector store.", exc)
    return InMemoryVectorStore()
