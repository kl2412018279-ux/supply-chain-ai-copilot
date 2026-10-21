"""Persist confirmed extractions (PostgreSQL JSONB, or in-memory offline)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Protocol

from src.config import get_settings
from src.db import PgConnection
from src.extraction.extract import ShippingDocument

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS shipping_documents (
    id              BIGSERIAL PRIMARY KEY,
    doc_type        TEXT NOT NULL,
    document_number TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    data            JSONB NOT NULL,
    issues          JSONB NOT NULL DEFAULT '[]'::jsonb,
    saved_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_type, document_number)
);
ALTER TABLE shipping_documents ENABLE ROW LEVEL SECURITY;
"""


class DocumentStore(Protocol):
    def save(self, doc: ShippingDocument, file_name: str, issues: list[dict]) -> dict: ...

    def list(self, limit: int = 100) -> list[dict]: ...


def _record(doc: ShippingDocument, file_name: str, issues: list[dict]) -> dict[str, Any]:
    if not doc.document_number:
        raise ValueError("Cannot save a document without a document number.")
    return {
        "doc_type": doc.doc_type.value,
        "document_number": doc.document_number,
        "file_name": file_name,
        "data": doc.model_dump(mode="json"),
        "issues": issues,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    def save(self, doc, file_name, issues):
        rec = _record(doc, file_name, issues)
        with self._lock:  # re-saving the same document replaces it
            self._rows[(rec["doc_type"], rec["document_number"])] = rec
        return rec

    def list(self, limit=100):
        return sorted(self._rows.values(), key=lambda r: r["saved_at"], reverse=True)[:limit]


class PostgresDocumentStore:
    def __init__(self, dsn: str):

        self._conn = PgConnection(dsn)
        self._conn.execute(SCHEMA_SQL)
        self._lock = threading.Lock()

    def save(self, doc, file_name, issues):
        rec = _record(doc, file_name, issues)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO shipping_documents (doc_type, document_number, file_name, data, issues)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (doc_type, document_number) DO UPDATE
                SET file_name = EXCLUDED.file_name, data = EXCLUDED.data,
                    issues = EXCLUDED.issues, saved_at = now()
                """,
                (rec["doc_type"], rec["document_number"], file_name,
                 json.dumps(rec["data"]), json.dumps(issues)),
            )  # fmt: skip
        return rec

    def list(self, limit=100):
        with self._lock:
            rows = self._conn.execute(
                """SELECT doc_type, document_number, file_name, data, issues, saved_at
                   FROM shipping_documents ORDER BY saved_at DESC LIMIT %s""",
                (limit,),
            ).fetchall()
        keys = ["doc_type", "document_number", "file_name", "data", "issues", "saved_at"]
        out = [dict(zip(keys, r, strict=True)) for r in rows]
        for r in out:
            r["saved_at"] = r["saved_at"].isoformat(timespec="seconds")
        return out


@lru_cache
def get_document_store() -> DocumentStore:
    dsn = get_settings().database_url
    if dsn:
        try:
            return PostgresDocumentStore(dsn)
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s); keeping documents in memory.", exc)
    return InMemoryDocumentStore()
