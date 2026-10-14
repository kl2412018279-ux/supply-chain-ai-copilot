"""PostgreSQL connection that survives dropped connections.

Each store keeps one long-lived connection. On a hosted database (Supabase,
Neon, RDS) that connection can be cut by a network blip, a pooler restart or an
idle timeout; a plain psycopg connection then fails on every later query until
the app restarts. This wrapper reopens it and retries the statement once.

Server-side prepared statements are disabled so the same code also works behind
a transaction-mode pooler (PgBouncer / Supavisor on port 6543).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PgConnection:
    def __init__(self, dsn: str, on_connect: Callable[[Any], None] | None = None):
        self._dsn = dsn
        self._on_connect = on_connect
        self._conn = self._open()

    def _open(self):
        import psycopg  # noqa: PLC0415

        conn = psycopg.connect(
            self._dsn,
            autocommit=True,
            prepare_threshold=None,
            connect_timeout=10,
            keepalives=1,
            keepalives_idle=30,
        )
        if self._on_connect:
            self._on_connect(conn)
        return conn

    def _alive(self) -> bool:
        return not (self._conn.closed or self._conn.broken)

    def _reconnect(self) -> None:
        logger.warning("Database connection lost; reconnecting.")
        with contextlib.suppress(Exception):  # already broken
            self._conn.close()
        self._conn = self._open()

    def _retry(self, fn: Callable[[], Any]) -> Any:
        import psycopg  # noqa: PLC0415

        if not self._alive():
            self._reconnect()
        try:
            return fn()
        except (psycopg.OperationalError, psycopg.InterfaceError):
            if self._alive():
                raise  # the statement failed, not the connection
            self._reconnect()
            return fn()

    def execute(self, query: Any, params: Any = None):
        return self._retry(lambda: self._conn.execute(query, params))

    def cursor(self):
        if not self._alive():
            self._reconnect()
        return self._conn.cursor()

    def close(self) -> None:
        self._conn.close()

    @property
    def raw(self):
        return self._conn
