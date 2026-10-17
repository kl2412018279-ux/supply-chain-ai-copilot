import json

import pytest

from src.evals.dataset import GOLDEN_PATH, GoldenCase, load_golden
from src.evals.judge import parse_judgement
from src.evals.metrics import (
    build_pool,
    derived_numbers,
    extract_claims,
    missing_includes,
    normalise_amount,
    parse_number,
    percentile,
    tool_score,
    unsupported_numbers,
)
from src.evals.run import gate, main, run_eval, summarise

REQUIRED_CATEGORIES = {
    "quote",
    "mode_comparison",
    "route_recommendation",
    "delay_risk",
    "free_time",
    "margin_leakage",
    "forecast",
    "spend",
    "scorecard",
    "documents",
    "adversarial",
}


# ------------------------------------------------------------------ number handling


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("RM1,234.50", 1234.5),
        ("RM 1,234.5", 1234.5),
        ("**RM18,760.00**", 18760.0),
        ("RM1.2m", 1_200_000.0),
        ("RM84.3k", 84_300.0),
        ("91.2%", 91.2),
        ("ipoh · rail -1.7%", -1.7),
        ("≈ 670 L", 670.0),
        ("3.2 TEU/day", 3.2),
    ],
)
def test_normalise_amount(text, value):
    assert normalise_amount(text) == pytest.approx(value)


def test_parse_number_keeps_shown_precision():
    assert parse_number("1,234.50") == (1234.5, 2)
    assert parse_number("670") == (670.0, 0)


def test_extract_claims_finds_money_percent_and_units_only():
    answer = (
        "**Price: RM10,546.32** (15% margin), 290 km, diesel ≈ 670 L, 87% on time, "
        "2 d door to door, 14 TEU. Form K8, rank 1, fit 77/100, ETA 2026-12-10, 20ft box."
    )
    claims = {c.text: c.kind for c in extract_claims(answer)}
    assert claims == {
        "RM10,546.32": "rm",
        "15%": "percent",
        "290 km": "unit",
        "670 L": "unit",
        "87%": "percent",
        "2 d": "unit",
        "14 TEU": "unit",
    }


def test_rounding_to_shown_precision():
    payload = {"price_rm": 1234.504, "diesel_litres": 669.6, "on_time_rate": 0.9123}
    ok = "RM1,234.50 or RM1,235 or RM1.2k, 670 L, 91.2% or 91% on time"
    assert unsupported_numbers(ok, [payload]) == []
    assert unsupported_numbers("RM1,236 and 92.5%", [payload]) == ["RM1,236", "92.5%"]


def test_detects_hallucinated_numbers():
    payload = {"price_rm": 7995.2, "margin_rm": 1042.85, "transit_days": {"truck": 1}}
    answer = "Price RM7,995.20, margin RM1,042.85, 1 d by truck. Rail would be RM5,100.00 (2 d)."
    assert unsupported_numbers(answer, [payload]) == ["RM5,100.00", "2 d"]


def test_numbers_inside_tool_text_and_question_count_as_support():
    docs = [{"snippet": "Typical free time is 3 days; insurance of RM 300,000 per box."}]
    answer = "Free time is 3 days and cover is RM300,000; you asked about 18 tonnes."
    assert unsupported_numbers(answer, docs, question="18 tonnes of resin") == []


def test_derived_sums_are_supported():
    quote = {
        "cost_rm": 100.0,
        "price_rm": 115.0,
        "line_items": [
            {"category": "truck", "amount_rm": 40.0},
            {"category": "truck", "amount_rm": 25.5},
            {"category": "fuel", "amount_rm": 34.5},
        ],
    }
    derived = derived_numbers(quote)
    assert 65.5 in derived  # truck lines grouped
    assert 15.0 in derived  # price - cost
    assert unsupported_numbers("truck RM65.50, margin RM15.00", [quote]) == []


def test_percent_matches_fraction_but_not_derived_fraction_noise():
    pool = build_pool([{"a": 0.92, "b": 0.90}])
    [claim] = extract_claims("92%")
    assert pool.supports(claim)
    [noise] = extract_claims("2%")  # 0.92 - 0.90 is not evidence for "2%"
    assert not pool.supports(noise)


def test_sign_is_presentation():
    assert unsupported_numbers("2 day(s) over", [{"days_left": -2}]) == []


# ------------------------------------------------------------------ other metrics


def test_tool_precision_recall():
    s = tool_score(["quote_container_job"], ["quote_container_job"])
    assert (s.precision, s.recall, s.passed) == (1.0, 1.0, True)

    s = tool_score(["quote_container_job"], ["quote_container_job", "search_operations_documents"])
    assert (s.precision, s.recall, s.passed) == (0.5, 1.0, True)
    assert s.unexpected == ["search_operations_documents"]

    s = tool_score(["a", "b"], ["b", "c", "c"])
    assert (s.precision, s.recall, s.passed) == (0.5, 0.5, False)
    assert s.missing == ["a"]

    assert tool_score([], []).passed and tool_score([], []).precision == 1.0
    assert tool_score(["a"], []).precision == 0.0


def test_missing_includes_substring_and_regex():
    answer = "Price: RM1,234.50 to Pasir Gudang (Johor)"
    assert missing_includes(answer, ["pasir gudang", r"re:RM\s?\d", "Kuantan"]) == ["Kuantan"]


def test_percentile():
    assert percentile([], 50) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(range(1, 101), 95) == pytest.approx(95.05)


def test_parse_judgement():
    j = parse_judgement('Sure: {"relevance": 5, "helpfulness": 4, "reason": "ok"}')
    assert (j.relevance, j.helpfulness) == (5, 4)
    assert parse_judgement('{"relevance": 9, "helpfulness": 4}') is None
    assert parse_judgement("no json") is None


# ------------------------------------------------------------------ golden set


def test_golden_file_loads_and_covers_the_categories():
    cases = load_golden()
    assert len(cases) >= 30
    assert {c.category for c in cases} >= REQUIRED_CATEGORIES
    adversarial = [c for c in cases if c.expect_blocked]
    assert len(adversarial) >= 8
    normal = [c for c in cases if not c.expect_blocked]
    assert all(c.expected_tools for c in normal)
    for line in GOLDEN_PATH.read_text().splitlines():
        assert set(json.loads(line)) >= {
            "id",
            "category",
            "question",
            "expected_tools",
            "must_include",
            "expect_blocked",
            "notes",
        }


def test_golden_schema_rejects_bad_cases(tmp_path):
    with pytest.raises(ValueError, match="unknown tool"):
        GoldenCase(id="x", category="q", question="?", expected_tools=["no_such_tool"])
    with pytest.raises(ValueError):
        GoldenCase(id="x", category="q", question="?", expect_blocked=True, must_include=["RM"])
    bad = tmp_path / "bad.jsonl"
    row = {"id": "a", "category": "c", "question": "q"}
    bad.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(bad)


def test_category_filter():
    assert {c.category for c in load_golden(category="quote")} == {"quote"}


# ------------------------------------------------------------------ end to end (offline)


@pytest.fixture(scope="module")
def offline_report(trained_model, ingested_store, delay_model):
    from src.agent.workflow import RuleBasedAgent

    return run_eval(load_golden(), RuleBasedAgent())


def test_offline_run_meets_thresholds(offline_report):
    o = offline_report["overall"]
    failing = {c["id"]: c["failures"] for c in offline_report["cases"] if not c["passed"]}
    assert o["tool_accuracy"] >= 0.9, failing
    assert o["groundedness"] >= 0.9, failing
    assert o["must_include_rate"] >= 0.9, failing
    assert o["block_rate"] == 1.0, failing
    assert o["false_block_rate"] == 0.0, failing
    assert o["claims"] > 100  # the groundedness check actually saw numbers
    assert gate(offline_report, 0.9) == []


def test_report_shape(offline_report):
    assert offline_report["agent"] == "rule-based"
    assert "adversarial" in offline_report["by_category"]
    case = offline_report["cases"][0]
    assert {"called_tools", "unsupported_numbers", "latency_ms", "failures"} <= set(case)
    assert summarise([])["tool_accuracy"] is None


def test_gate_flags_low_metrics():
    report = {"overall": {"tool_accuracy": 0.8, "groundedness": None}}
    assert gate(report, 0.9) == ["tool_accuracy=0.800"]
    assert gate(report, None) == []


# ------------------------------------------------------------------ CLI


def test_cli_passes_and_writes_reports(tmp_path, capsys):
    out = tmp_path / "evals.json"
    code = main(["--agent", "rule", "--category", "quote", "--out", str(out), "--no-bootstrap"])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["overall"]["cases"] == 4
    assert (tmp_path / "evals.md").read_text().startswith("# Copilot answer-quality eval")
    assert "OVERALL" in capsys.readouterr().out


def test_cli_fail_under_exits_nonzero(tmp_path):
    out = tmp_path / "evals.json"
    args = ["--agent", "rule", "--category", "quote", "--out", str(out), "--no-bootstrap"]
    assert main([*args, "--fail-under", "0.9"]) == 0
    assert main([*args, "--fail-under", "1.01"]) == 1


def test_cli_llm_agent_unavailable_offline(tmp_path):
    out = tmp_path / "evals.json"
    assert main(["--agent", "llm", "--out", str(out), "--no-bootstrap"]) == 2
    assert main(["--category", "nope", "--out", str(out), "--no-bootstrap"]) == 2
    assert not out.exists()


# ------------------------------------------------------------------ dashboard page


def test_ai_quality_page_renders_report(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from src.agent.workflow import RuleBasedAgent
    from tests.conftest import ROOT

    report = run_eval(load_golden(category="quote"), RuleBasedAgent())
    report["cases"][0]["passed"] = False
    report["cases"][0]["unsupported_numbers"] = ["RM9.99"]
    report["cases"][0]["failures"] = ["unsupported numbers: RM9.99"]

    at = AppTest.from_file(str(ROOT / "views" / "evals.py"), default_timeout=60)
    at.session_state["eval_report"] = report
    at.run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Tool selection", "Groundedness", "Attacks blocked", "Latency p50 / p95"} <= labels
    assert any(b.label == "Run evaluation" for b in at.button)
    assert any("quote-01" in h.proto.body for h in at.get("html"))
