"""End-to-end request pipeline::

input guardrails -> semantic cache -> agent (LangGraph / rule-based)
                 -> output redaction -> Pydantic-validated CopilotResponse
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from pydantic import ValidationError

from src.agent.workflow import Agent, build_agent
from src.cache.semantic_cache import SemanticCache, get_semantic_cache
from src.config import get_settings
from src.guardrails.schemas import CopilotResponse
from src.guardrails.validators import check_input, redact
from src.observability import configure_tracing, metrics, traceable

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "Sorry, I couldn't produce a reliable answer. Please rephrase the question."


class SupplyChainCopilot:
    def __init__(self, agent: Agent, cache: SemanticCache | None = None):
        self.agent = agent
        self.cache = cache

    @traceable(name="copilot.ask")
    def ask(self, query: str, use_cache: bool = True) -> CopilotResponse:
        start = time.perf_counter()
        metrics.incr("requests")

        def done(response: CopilotResponse) -> CopilotResponse:
            response.latency_ms = round((time.perf_counter() - start) * 1000, 2)
            metrics.observe("copilot.ask", response.latency_ms)
            return response

        guard = check_input(query)
        if not guard.allowed:
            metrics.incr(f"blocked.{guard.category.value}")
            return done(
                CopilotResponse(
                    answer=guard.reason, blocked=True, guardrail=guard.category, agent="guardrail"
                )
            )

        if use_cache and self.cache is not None:
            hit = self.cache.lookup(query)
            if hit is not None:
                metrics.incr("cache.hit")
                cached = CopilotResponse.model_validate(hit[0])
                cached.cached = True
                return done(cached)
            metrics.incr("cache.miss")

        try:
            output = self.agent.run(query)
            response = CopilotResponse(
                answer=redact(output.answer) or FALLBACK_ANSWER,
                tool_calls=output.tool_calls,
                sources=output.sources,
                agent=output.agent,
            )
        except ValidationError:
            logger.exception("Agent output failed schema validation")
            metrics.incr("errors.validation")
            return done(CopilotResponse(answer=FALLBACK_ANSWER, agent=self.agent.name))
        except Exception:
            logger.exception("Agent run failed")
            metrics.incr("errors.agent")
            return done(
                CopilotResponse(
                    answer="The copilot hit an internal error while answering. Please retry.",
                    agent=self.agent.name,
                )
            )

        if use_cache and self.cache is not None and response.tool_calls:
            self.cache.store(query, response.model_dump(mode="json", exclude={"latency_ms"}))
        return done(response)


@lru_cache
def get_copilot() -> SupplyChainCopilot:
    configure_tracing()
    settings = get_settings()
    cache = get_semantic_cache() if settings.cache_enabled else None
    return SupplyChainCopilot(build_agent(settings), cache)
