"""Observability: LangSmith tracing for LLM/agent steps + lightweight in-process metrics.

When ``LANGSMITH_API_KEY`` is set, every LangGraph node, LLM call (prompt, tokens,
cost, latency) and tool call is traced to LangSmith automatically, and the
``traceable`` decorator adds the surrounding pipeline (guardrails, cache) as a
parent span. Without a key everything still works and metrics stay local.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from src.config import get_settings

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def configure_tracing() -> bool:
    settings = get_settings()
    if not settings.langsmith_api_key:
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    return True


def traceable(name: str, run_type: str = "chain") -> Callable[[F], F]:
    """LangSmith ``@traceable`` when available and enabled, otherwise a no-op."""

    def decorator(fn: F) -> F:
        if not get_settings().langsmith_api_key:
            return fn
        try:
            from langsmith import traceable as ls_traceable  # noqa: PLC0415
        except ImportError:
            return fn
        return ls_traceable(name=name, run_type=run_type)(fn)  # type: ignore[return-value]

    return decorator


class Metrics:
    """Thread-safe counters and latency stats exposed at ``GET /metrics``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = {}
        self._latency: dict[str, list[float]] = {}

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            series = self._latency.setdefault(name, [])
            series.append(ms)
            del series[:-1000]  # keep a bounded window

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latency = {}
            for name, series in self._latency.items():
                ordered = sorted(series)
                latency[name] = {
                    "count": len(ordered),
                    "p50_ms": round(ordered[len(ordered) // 2], 2),
                    "p95_ms": round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 2),
                }
            return {"counters": dict(self.counters), "latency": latency}


metrics = Metrics()


def timed(name: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                metrics.observe(name, (time.perf_counter() - start) * 1000)

        return wrapper  # type: ignore[return-value]

    return decorator
