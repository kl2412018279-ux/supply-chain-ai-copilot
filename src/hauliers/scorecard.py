"""Haulier scorecard: delivery performance (SQL) + compliance evidence (documents) -> grade.

Performance comes from analytical SQL over the delivery log, run on PostgreSQL when
``DATABASE_URL`` is set and on an in-memory SQLite copy of the CSV otherwise (the
same SQL works on both). Compliance evidence (DG training, driver hours, fleet
maintenance, tracking, cargo insurance, ISO certification) is extracted from each
haulier's policy document: every criterion that is met cites the exact sentence and file.

Composite score (0-1)::

    0.45 x on-time     (70% -> 0, 100% -> 1)
  + 0.25 x quality     (damage rate 3% -> 0, 0% -> 1)
  + 0.20 x compliance  (criteria met / criteria checked)
  + 0.10 x price       (rate-card price factor 1.20 -> 0, 0.85 -> 1)
"""

from __future__ import annotations

import logging
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.config import PROJECT_ROOT, get_settings
from src.costing.rates import get_rate_card
from src.db import PgConnection

logger = logging.getLogger(__name__)

DELIVERIES_CSV = PROJECT_ROOT / "data" / "raw" / "haulier_deliveries.csv"
POLICY_DOCS: dict[str, Path] = {
    "muara-fleet": PROJECT_ROOT / "data" / "raw" / "operations_manual.pdf",
    **{
        key: PROJECT_ROOT / "data" / "raw" / f"haulier_policy_{key}.md"
        for key in ("hijau", "cempaka", "raillink", "nusantara", "borneo-express")
    },
}
WEIGHTS = {"on_time": 0.45, "quality": 0.25, "compliance": 0.20, "price": 0.10}

# criterion -> (label, regex that counts as evidence)
COMPLIANCE_CRITERIA: dict[str, tuple[str, str]] = {
    "dg_training": (
        "Dangerous-goods training / licensing",
        r"(dangerous goods|\bdg) (training|endorsement|licen[cs]e)"
        r"|(training|certification|endorsement).{0,40}dangerous goods",
    ),
    "driver_hours": (
        "Driver working hours & fatigue management",
        # A promise to "look into" fatigue is not a rule: require a limit or a rest rule.
        r"(maximum|limit\w*|no more than|not exceed)\w*.{0,40}\b(hours|driving)"
        r"|rest (hours|periods|breaks?)|fatigue management",
    ),
    "fleet_maintenance": (
        "Modern fleet (Euro 5/6) & preventive maintenance",
        r"\beuro ?(5|6|v|vi)\b|preventive maintenance|planned maintenance",
    ),
    "tracking": (
        "GPS / real-time tracking",
        r"\bgps\b|real[- ]time (tracking|visibility|position)|\bais\b",
    ),
    "cargo_insurance": (
        "Goods-in-transit / cargo liability insurance",
        r"goods[- ]in[- ]transit (insurance|cover)|cargo (liability|insurance)"
        r"|liability insurance",
    ),
    "certification": (
        "Certified management system (ISO 9001 / 45001 / 39001)",
        r"iso ?(9001|45001|39001)",
    ),
}

PERFORMANCE_SQL = """
SELECT haulier,
       COUNT(*)                                              AS deliveries,
       SUM(containers)                                       AS containers,
       AVG(CAST(on_time AS REAL))                            AS on_time_rate,
       AVG(CASE WHEN on_time = 0 THEN late_hours END)        AS avg_late_hours,
       AVG(CAST(damaged AS REAL))                            AS damage_rate,
       SUM(claim_rm)                                         AS claims_rm
FROM haulier_deliveries
WHERE (:start IS NULL OR date >= :start)
GROUP BY haulier
"""

MONTHLY_SQL = """
SELECT SUBSTR(CAST(date AS TEXT), 1, 7) AS month,
       AVG(CAST(on_time AS REAL))       AS on_time_rate,
       COUNT(*)                         AS deliveries
FROM haulier_deliveries
WHERE haulier = :haulier
GROUP BY SUBSTR(CAST(date AS TEXT), 1, 7)
ORDER BY month
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS haulier_deliveries (
    delivery_id TEXT PRIMARY KEY,
    date        DATE    NOT NULL,
    haulier     TEXT    NOT NULL,
    mode        TEXT    NOT NULL,
    destination TEXT    NOT NULL,
    commodity   TEXT    NOT NULL,
    containers  INTEGER NOT NULL,
    late_hours  REAL    NOT NULL,
    on_time     INTEGER NOT NULL,
    damaged     INTEGER NOT NULL,
    claim_rm    REAL    NOT NULL
);
ALTER TABLE haulier_deliveries ENABLE ROW LEVEL SECURITY;
"""


class ComplianceEvidence(BaseModel):
    criterion: str
    label: str
    met: bool
    evidence: str | None = None
    source: str | None = None


class HaulierScore(BaseModel):
    haulier: str
    label: str
    modes: list[str]
    fleet: str
    deliveries: int
    containers: int
    on_time_rate: float
    avg_late_hours: float
    damage_rate: float
    claims_rm: float
    compliance_score: float
    compliance: list[ComplianceEvidence]
    price_factor: float
    component_scores: dict[str, float]
    score: float
    grade: str
    rank: int = 0
    summary: str = ""


# ------------------------------------------------------------------------ SQL backend


class DeliveryLog:
    """Runs the scorecard SQL on PostgreSQL or an in-memory SQLite copy of the CSV."""

    def __init__(self, df: pd.DataFrame | None = None, dsn: str | None = None):
        self._pg = None
        if dsn:
            self._pg = PgConnection(dsn)
            self._pg.execute(SCHEMA_SQL)
            count = self._pg.execute("SELECT count(*) FROM haulier_deliveries").fetchone()[0]
            if count == 0:
                self._load_pg(df if df is not None else pd.read_csv(DELIVERIES_CSV))
        else:
            self._sqlite = sqlite3.connect(":memory:", check_same_thread=False)
            frame = df if df is not None else pd.read_csv(DELIVERIES_CSV)
            frame.to_sql("haulier_deliveries", self._sqlite, index=False)

    def _load_pg(self, df: pd.DataFrame) -> None:
        cols = list(df.columns)
        rows = [tuple(r) for r in df[cols].itertuples(index=False)]
        placeholders = ", ".join(["%s"] * len(cols))
        with self._pg.cursor() as cur:
            cur.executemany(
                f"INSERT INTO haulier_deliveries ({', '.join(cols)}) VALUES ({placeholders}) "
                "ON CONFLICT (delivery_id) DO NOTHING",
                rows,
            )

    def query(self, sql: str, params: dict[str, Any]) -> pd.DataFrame:
        if self._pg is not None:
            pg_sql = re.sub(r":(\w+)", r"%(\1)s", sql)
            pg_sql = pg_sql.replace("%(start)s IS NULL", "%(start)s::date IS NULL")
            cur = self._pg.execute(pg_sql, params)
            return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
        return pd.read_sql_query(sql, self._sqlite, params=params)


@lru_cache
def get_delivery_log() -> DeliveryLog:
    dsn = get_settings().database_url
    if dsn:
        try:
            return DeliveryLog(dsn=dsn)
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s); using SQLite for delivery logs.", exc)
    return DeliveryLog()


# --------------------------------------------------------------- compliance evidence


def _document_text(path: Path) -> str:
    from src.rag.ingest_pdf import extract_pages  # noqa: PLC0415

    pages = extract_pages(path.read_bytes(), path.name)
    return " ".join(" ".join(text.split()) for _, text in pages)


@lru_cache
def compliance_evidence(haulier: str) -> tuple[ComplianceEvidence, ...]:
    path = POLICY_DOCS.get(haulier)
    text = _document_text(path) if path and path.exists() else ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    found = []
    for key, (label, pattern) in COMPLIANCE_CRITERIA.items():
        regex = re.compile(pattern, re.IGNORECASE)
        hit = next((s.strip() for s in sentences if regex.search(s)), None)
        found.append(
            ComplianceEvidence(
                criterion=key,
                label=label,
                met=hit is not None,
                evidence=hit,
                source=path.name if hit and path else None,
            )
        )
    return tuple(found)


# ------------------------------------------------------------------------- scoring


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _grade(score: float) -> str:
    return "A" if score >= 0.8 else "B" if score >= 0.65 else "C" if score >= 0.5 else "D"


def _summary(s: HaulierScore) -> str:
    strengths, gaps = [], []
    if s.on_time_rate >= 0.92:
        strengths.append(f"reliable ({s.on_time_rate:.0%} on time)")
    elif s.on_time_rate < 0.85:
        gaps.append(
            f"punctuality ({s.on_time_rate:.0%} on time, "
            f"late by {s.avg_late_hours:.0f} h on average)"
        )
    if s.damage_rate <= 0.008:
        strengths.append(f"low damage rate ({s.damage_rate:.1%})")
    elif s.damage_rate >= 0.015:
        gaps.append(f"damage ({s.damage_rate:.1%} of jobs, RM{s.claims_rm:,.0f} in claims)")
    missing = [e.label.lower() for e in s.compliance if not e.met]
    if s.compliance_score >= 0.8:
        strengths.append("strong compliance evidence")
    elif missing:
        gaps.append("no evidence of " + ", ".join(missing[:3]))
    if s.price_factor <= 0.9:
        strengths.append("cheapest rates")
    text = f"{s.label}: grade {s.grade} ({s.score:.2f}), rank {s.rank}."
    if strengths:
        text += " Strengths: " + "; ".join(strengths) + "."
    if gaps:
        text += " Watch: " + "; ".join(gaps) + "."
    return text


def build_scorecard(log: DeliveryLog | None = None, since: str | None = None) -> list[HaulierScore]:
    """Score and rank every carrier in the rate card that has delivery history."""
    log = log or get_delivery_log()
    carriers = {k: v for k, v in get_rate_card().data["carriers"].items() if not k.startswith("_")}
    perf = log.query(PERFORMANCE_SQL, {"start": since}).set_index("haulier")
    scores: list[HaulierScore] = []
    for key, carrier in carriers.items():
        if key not in perf.index:
            continue
        row = perf.loc[key]
        evidence = list(compliance_evidence(key))
        compliance = sum(e.met for e in evidence) / len(evidence)
        components = {
            "on_time": _clip((float(row["on_time_rate"]) - 0.70) / 0.30),
            "quality": _clip(1 - float(row["damage_rate"]) / 0.03),
            "compliance": compliance,
            "price": _clip((1.20 - carrier["price_factor"]) / 0.35),
        }
        score = sum(WEIGHTS[k] * v for k, v in components.items())
        scores.append(
            HaulierScore(
                haulier=key,
                label=carrier["label"],
                modes=carrier["modes"],
                fleet=carrier.get("fleet", ""),
                deliveries=int(row["deliveries"]),
                containers=int(row["containers"]),
                on_time_rate=round(float(row["on_time_rate"]), 4),
                avg_late_hours=round(float(row["avg_late_hours"] or 0), 1),
                damage_rate=round(float(row["damage_rate"]), 4),
                claims_rm=round(float(row["claims_rm"]), 2),
                compliance_score=round(compliance, 3),
                compliance=evidence,
                price_factor=carrier["price_factor"],
                component_scores={k: round(v, 3) for k, v in components.items()},
                score=round(score, 3),
                grade=_grade(score),
            )
        )
    scores.sort(key=lambda s: s.score, reverse=True)
    for i, s in enumerate(scores, start=1):
        s.rank = i
        s.summary = _summary(s)
    return scores


def monthly_on_time(haulier: str, log: DeliveryLog | None = None) -> pd.DataFrame:
    return (log or get_delivery_log()).query(MONTHLY_SQL, {"haulier": haulier})


def on_time_rates() -> dict[str, float]:
    """Historical on-time rate per carrier (used by the route planner)."""
    perf = get_delivery_log().query(PERFORMANCE_SQL, {"start": None})
    return dict(zip(perf["haulier"], perf["on_time_rate"].astype(float), strict=True))


def damage_rates() -> dict[str, float]:
    """Historical cargo damage rate per carrier (the route planner's safety criterion)."""
    perf = get_delivery_log().query(PERFORMANCE_SQL, {"start": None})
    return dict(zip(perf["haulier"], perf["damage_rate"].astype(float), strict=True))


SUMMARY_PROMPT = """You are a logistics procurement analyst. Write a 3-sentence assessment of
the haulier below for an operations manager: overall verdict, the main strength, and the
main risk with one concrete action. Use only these facts; do not invent numbers.

{facts}"""


def llm_summary(score: HaulierScore, llm) -> str:
    """Natural-language assessment via a LangChain prompt template (falls back to rules)."""
    from langchain_core.prompts import ChatPromptTemplate  # noqa: PLC0415

    facts = score.model_dump(exclude={"summary"}, mode="json")
    chain = ChatPromptTemplate.from_template(SUMMARY_PROMPT) | llm
    content = chain.invoke({"facts": facts}).content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content).strip() or score.summary
