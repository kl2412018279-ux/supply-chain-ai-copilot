"""Input and output guardrails for the copilot.

Input:  length limits, prompt-injection / jailbreak detection, and a topical
        allow-list so the assistant refuses non-supply-chain questions.
Output: secret / PII redaction and strict Pydantic schema validation of the
        final response (see ``src.guardrails.schemas``).

Rules are deterministic and fast (<1 ms) so they run before any token is spent.
They can be complemented by NeMo Guardrails / Guardrails.ai policies.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel

MAX_QUERY_CHARS = 2000


class GuardrailCategory(str, Enum):
    ok = "ok"
    empty = "empty"
    too_long = "too_long"
    prompt_injection = "prompt_injection"
    off_topic = "off_topic"


class GuardrailResult(BaseModel):
    allowed: bool
    category: GuardrailCategory
    reason: str = ""


INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"\bdisregard\s+(all\s+|the\s+|your\s+)?(previous|prior|above|system)\b",
    r"\bforget\s+(all\s+|your\s+|the\s+)?(instructions|rules|guidelines)",
    r"\b(reveal|show|print|repeat|leak)\b.{0,30}\b(system\s+prompt|instructions|hidden\s+prompt)",
    r"\byou\s+are\s+now\b",
    r"\b(act|pretend|roleplay)\s+as\s+(an?\s+)?(unrestricted|jailbroken|dan|evil)",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"\bdo\s+anything\s+now\b",
    r"<\s*/?\s*(system|assistant|im_start|im_end)\s*>",
    r"\b(drop|truncate|delete\s+from|alter)\s+table\b",
    r"\b(api[_\s-]?key|password|credentials?|secret\s+key)s?\b.{0,20}\b(give|show|reveal|print|what)",
    r"\b(give|show|reveal|print|what)\b.{0,30}\b(api[_\s-]?key|password|credentials?)\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

DOMAIN_TERMS = {
    # supply chain & operations
    "supply",
    "chain",
    "demand",
    "forecast",
    "forecasting",
    "predict",
    "prediction",
    "sales",
    "sku",
    "skus",
    "inventory",
    "stock",
    "safety",
    "warehouse",
    "logistics",
    "shipping",
    "ship",
    "shipment",
    "shipped",
    "freight",
    "transport",
    "transportation",
    "truck",
    "rail",
    "air",
    "sea",
    "ocean",
    "procurement",
    "purchase",
    "order",
    "orders",
    "supplier",
    "suppliers",
    "vendor",
    "lead",
    "replenishment",
    "distribution",
    "delivery",
    "units",
    "product",
    "products",
    "manufacturing",
    "production",
    "capacity",
    "planning",
    "reorder",
    "backorder",
    "sourcing",
    "packaging",
    "pallet",
    "container",
    "route",
    "km",
    "kg",
    "tonnes",
    "tons",
    # compliance, claims & commercial terms
    "waste",
    "policy",
    "compliance",
    "audit",
    "report",
    "reporting",
    "insurance",
    "insured",
    "liability",
    "claim",
    "claims",
    "damage",
    "damaged",
    "margin",
    "transit",
    "iso",
    "gps",
    "tracking",
    "fatigue",
    "licence",
    "license",
    "training",
    "certification",
    "dangerous",
    "goods",
    # freight forwarding & port operations (Malaysia)
    "forwarder", "forwarding", "haulage", "haulier", "prime", "mover", "trailer",
    "teu", "teus", "containers", "20ft", "40ft", "ft", "port", "klang", "westport", "northport",
    "terminal", "customs", "clearance", "declaration", "k1", "k2", "k8", "ucustoms", "permit",
    "permits", "duty", "duties", "quote", "quotation", "price", "pricing", "cost", "costs",
    "rm", "ringgit", "rate", "rates", "toll", "tolls", "diesel", "fuel", "maintenance", "driver",
    "demurrage", "detention", "storage", "pallets", "palm", "oil", "cpo", "mspo", "mpob",
    "flexitank", "chemical", "chemicals", "sds", "msds", "un", "plastic", "plastics", "resin",
    "spend", "budget", "revenue", "volume", "volumes", "malaysia", "penang", "johor", "sabah",
    "sarawak", "ipoh", "kuantan", "melaka", "shah", "alam", "kl", "kuching", "kinabalu",
    "carrier", "carriers", "hauliers", "subcontractor", "scorecard", "delay", "delayed",
    "delays", "late", "eta", "risk", "congestion", "monsoon", "bill", "lading", "invoice",
    "recommend", "routes", "reliable", "on-time",
}  # fmt: skip
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|help|what can you do)\b[\s!?.]*$", re.IGNORECASE
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def check_input(query: str) -> GuardrailResult:
    text = (query or "").strip()
    if not text:
        return GuardrailResult(
            allowed=False, category=GuardrailCategory.empty, reason="Empty query."
        )
    if len(text) > MAX_QUERY_CHARS:
        return GuardrailResult(
            allowed=False,
            category=GuardrailCategory.too_long,
            reason=f"Query exceeds {MAX_QUERY_CHARS} characters.",
        )
    for pattern in _INJECTION_RE:
        if pattern.search(text):
            return GuardrailResult(
                allowed=False,
                category=GuardrailCategory.prompt_injection,
                reason="The request looks like a prompt-injection attempt and was blocked.",
            )
    if _GREETING_RE.match(text):
        return GuardrailResult(allowed=True, category=GuardrailCategory.ok)
    words = set(_WORD_RE.findall(text.lower()))
    if not words & DOMAIN_TERMS:
        return GuardrailResult(
            allowed=False,
            category=GuardrailCategory.off_topic,
            reason=(
                "I can only help with freight, logistics, customs, pricing, carrier "
                "performance and compliance questions."
            ),
        )
    return GuardrailResult(allowed=True, category=GuardrailCategory.ok)


_REDACTIONS = [
    (re.compile(r"\b(sk|gsk|sk-ant|lsv2_pt)[-_][A-Za-z0-9_\-]{16,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"postgres(ql)?://[^\s:@/]+:[^\s@/]+@"), "postgresql://[REDACTED]@"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED_CARD]"),
]


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text
