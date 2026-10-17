"""Optional LLM-as-judge: relevance and helpfulness of an answer on a 1-5 scale.

Uses the chat model configured for the copilot (``LLM_PROVIDER`` / API key) at
temperature 0 and asks for strict JSON. When no provider is configured the judge is
simply unavailable and the eval reports deterministic metrics only.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You grade answers from a freight-operations assistant for a Port Klang 3PL.

Score the ANSWER to the QUESTION on two 1-5 scales:
- relevance: 5 = directly answers what was asked; 3 = partly on point; 1 = off topic.
- helpfulness: 5 = a dispatcher could act on it (key figures, next step, caveats);
  3 = correct but thin or cluttered; 1 = useless or misleading.
Judge only what is written. Do not reward length. Do not check arithmetic.

Reply with JSON only, no prose:
{"relevance": <1-5>, "helpfulness": <1-5>, "reason": "<one short sentence>"}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class JudgeScore:
    relevance: int
    helpfulness: int
    reason: str = ""


def parse_judgement(text: str) -> JudgeScore | None:
    """Pull the JSON verdict out of the judge's reply; ``None`` if it is unusable."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        data: dict[str, Any] = json.loads(m.group(0))
        rel, helpful = int(data["relevance"]), int(data["helpfulness"])
    except (ValueError, KeyError, TypeError):
        return None
    if not (1 <= rel <= 5 and 1 <= helpful <= 5):
        return None
    return JudgeScore(rel, helpful, str(data.get("reason", ""))[:300])


class LLMJudge:
    """Scores answers with the configured chat model."""

    def __init__(self, llm):
        self.llm = llm

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LLMJudge | None:
        from src.agent.workflow import build_llm  # noqa: PLC0415

        settings = replace(settings or get_settings(), llm_temperature=0.0)
        llm = build_llm(settings)
        return cls(llm) if llm is not None else None

    def score(self, question: str, answer: str) -> JudgeScore | None:
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

        try:
            reply = self.llm.invoke(
                [
                    SystemMessage(content=JUDGE_PROMPT),
                    HumanMessage(content=f"QUESTION:\n{question}\n\nANSWER:\n{answer}"),
                ]
            )
        except Exception:  # provider outage / rate limit: skip this case, keep the run
            logger.exception("Judge call failed")
            return None
        content = reply.content
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        return parse_judgement(str(content))
