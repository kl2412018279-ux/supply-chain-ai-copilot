import pandas as pd
import pytest

from src.hauliers.data import generate_deliveries
from src.hauliers.scorecard import (
    COMPLIANCE_CRITERIA,
    POLICY_DOCS,
    WEIGHTS,
    DeliveryLog,
    build_scorecard,
    compliance_evidence,
    damage_rates,
    monthly_on_time,
    on_time_rates,
)


@pytest.fixture(scope="module")
def log():
    return DeliveryLog(df=generate_deliveries(start="2026-01-01", end="2026-06-30"))


def _by_key(scores):
    return {s.haulier: s for s in scores}


def test_generated_logs_are_consistent():
    df = generate_deliveries(start="2026-01-01", end="2026-01-31")
    assert set(df["mode"]) <= {"truck", "rail", "sea"}
    assert ((df["on_time"] == 1) == (df["late_hours"] <= 2)).all()
    assert (df.loc[df["damaged"] == 0, "claim_rm"] == 0).all()


def test_sql_performance_matches_pandas(log):
    df = generate_deliveries(start="2026-01-01", end="2026-06-30")
    scores = _by_key(build_scorecard(log))
    expected = df.groupby("haulier")["on_time"].mean()
    for key, rate in expected.items():
        assert scores[key].on_time_rate == pytest.approx(rate, abs=1e-4)
        assert scores[key].deliveries == int((df["haulier"] == key).sum())


def test_ranking_separates_good_and_poor_hauliers(log):
    scores = build_scorecard(log)
    assert [s.rank for s in scores] == list(range(1, len(scores) + 1))
    assert scores == sorted(scores, key=lambda s: s.score, reverse=True)
    by_key = _by_key(scores)
    assert {scores[0].haulier, scores[1].haulier} == {"muara-fleet", "hijau"}
    assert by_key["muara-fleet"].grade == "A" and by_key["hijau"].grade in {"A", "B"}
    assert by_key["cempaka"].grade == "D"
    assert "punctuality" in by_key["cempaka"].summary or "damage" in by_key["cempaka"].summary


def _met(haulier):
    return {e.criterion for e in compliance_evidence(haulier) if e.met}


def test_compliance_evidence_cites_sentences():
    hijau = {e.criterion: e for e in compliance_evidence("hijau")}
    assert set(hijau) == set(COMPLIANCE_CRITERIA)
    assert all(e.met for e in hijau.values())
    assert "RM 500,000" in hijau["cargo_insurance"].evidence
    assert "maximum of 10 hours" in hijau["driver_hours"].evidence
    assert hijau["certification"].source == "haulier_policy_hijau.md"


def test_hauliers_meet_different_compliance_subsets():
    assert _met("cempaka") == {"tracking"}  # weakest: GPS units only
    assert _met("raillink") == {"dg_training", "fleet_maintenance", "tracking", "certification"}
    assert _met("nusantara") == {"dg_training", "driver_hours", "tracking", "cargo_insurance"}
    assert _met("borneo-express") == {"dg_training", "cargo_insurance"}


def test_muara_evidence_comes_from_operations_manual():
    muara = {e.criterion: e for e in compliance_evidence("muara-fleet")}
    assert all(e.met for e in muara.values())
    assert {e.source for e in muara.values()} == {"operations_manual.pdf"}
    assert "goods-in-transit insurance" in muara["cargo_insurance"].evidence
    assert "ISO 9001" in muara["certification"].evidence


def test_near_misses_are_not_evidence():
    # "arrange their own cargo cover" is not insurance; monthly inspection is not a Euro 5/6
    # fleet with preventive maintenance; a planned certification audit is not a certificate.
    assert "cargo_insurance" not in _met("cempaka")
    assert "fleet_maintenance" not in _met("cempaka")
    assert "certification" not in _met("borneo-express")


def test_policy_documents_exist_and_weights_sum_to_one():
    assert all(path.exists() for path in POLICY_DOCS.values())
    assert WEIGHTS == {"on_time": 0.45, "quality": 0.25, "compliance": 0.20, "price": 0.10}
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_scorecard_exposes_compliance_fields(log):
    s = _by_key(build_scorecard(log))["raillink"]
    assert s.compliance_score == pytest.approx(4 / 6, abs=1e-3)
    assert set(s.component_scores) == {"on_time", "quality", "compliance", "price"}


def test_monthly_trend_and_on_time_rates(log):
    trend = monthly_on_time("hijau", log)
    assert list(trend["month"]) == [f"2026-0{m}" for m in range(1, 7)]
    assert trend["on_time_rate"].between(0, 1).all()
    rates = on_time_rates()
    assert set(rates) >= {"hijau", "cempaka", "raillink"}
    damage = damage_rates()
    assert set(damage) == set(rates)
    assert damage["cempaka"] > damage["hijau"]


def test_llm_summary_uses_prompt_template(log):
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    from src.hauliers.scorecard import llm_summary

    seen = {}

    def fake_llm(prompt_value):
        seen["prompt"] = prompt_value.to_string()
        return AIMessage(content="Solid partner. Keep it.")

    score = build_scorecard(log)[0]
    assert llm_summary(score, RunnableLambda(fake_llm)) == "Solid partner. Keep it."
    assert "do not invent numbers" in seen["prompt"] and score.label in seen["prompt"]


@pytest.mark.integration
def test_postgres_backend_matches_sqlite(log):
    import os

    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set")
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS haulier_deliveries")
    df = generate_deliveries(start="2026-01-01", end="2026-06-30")
    pg = _by_key(build_scorecard(DeliveryLog(df=df, dsn=dsn)))
    sq = _by_key(build_scorecard(log))
    for key in sq:
        assert pg[key].on_time_rate == pytest.approx(sq[key].on_time_rate, abs=1e-4)
        assert pg[key].claims_rm == pytest.approx(sq[key].claims_rm, abs=0.01)
    assert isinstance(monthly_on_time("hijau", DeliveryLog(dsn=dsn)), pd.DataFrame)
