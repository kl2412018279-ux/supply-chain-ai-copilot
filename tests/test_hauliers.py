import pandas as pd
import pytest

from src.hauliers.data import generate_deliveries
from src.hauliers.scorecard import (
    DeliveryLog,
    build_scorecard,
    esg_evidence,
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
    assert by_key["hijau"].grade == "A"
    assert by_key["cempaka"].grade == "D"
    assert "punctuality" in by_key["cempaka"].summary or "damage" in by_key["cempaka"].summary


def test_esg_evidence_cites_sentences():
    hijau = {e.criterion: e for e in esg_evidence("hijau")}
    assert all(e.met for e in hijau.values())
    assert "40 percent by 2030" in hijau["emissions_target"].evidence
    assert hijau["certification"].source == "haulier_esg_hijau.md"
    assert not any(e.met for e in esg_evidence("cempaka"))


def test_plan_to_set_target_is_not_a_target():
    borneo = {e.criterion: e for e in esg_evidence("borneo-express")}
    assert not borneo["emissions_target"].met  # "plan to publish a reduction target in 2027"
    assert not borneo["clean_fleet"].met  # low-sulphur fuel is compliance, not a cleaner fleet


def test_measurement_detected_in_either_word_order():
    nusantara = {e.criterion: e for e in esg_evidence("nusantara")}
    assert nusantara["measurement"].met  # "...CO2 emissions per voyage are recorded and reported"


def test_monthly_trend_and_on_time_rates(log):
    trend = monthly_on_time("hijau", log)
    assert list(trend["month"]) == [f"2026-0{m}" for m in range(1, 7)]
    assert trend["on_time_rate"].between(0, 1).all()
    rates = on_time_rates()
    assert set(rates) >= {"hijau", "cempaka", "raillink"}


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
