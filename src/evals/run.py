"""Run the copilot answer-quality eval over the golden set.

    python -m src.evals.run                         # auto: LLM agent if configured
    python -m src.evals.run --agent rule --fail-under 0.9    # what CI runs (offline)
    LLM_PROVIDER=groq GROQ_API_KEY=... python -m src.evals.run --agent llm

Each question goes through the full pipeline (input guardrails -> agent -> output
redaction), with the semantic cache off. Prints a per-category table and writes a JSON
report plus a Markdown summary next to it. Exits 1 when overall tool accuracy or
groundedness falls under ``--fail-under``, and 2 when the requested agent is unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.evals.dataset import GoldenCase, load_golden
from src.evals.metrics import (
    extract_claims,
    missing_includes,
    percentile,
    tool_score,
    unsupported_numbers,
)

logger = logging.getLogger(__name__)

REPORT_PATH = PROJECT_ROOT / "reports" / "evals.json"
AGENT_CHOICES = ("rule", "llm", "auto")
GATED_METRICS = ("tool_accuracy", "groundedness")
# Offline backends for the deterministic run; anything set explicitly in the shell wins.
OFFLINE_DEFAULTS = {
    "EMBEDDING_BACKEND": "hashing",
    "RERANKER_BACKEND": "lexical",
    "WEATHER_ENABLED": "false",
}


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    expect_blocked: bool
    blocked: bool
    guardrail: str
    agent: str
    answer: str
    expected_tools: list[str]
    called_tools: list[str]
    tool_precision: float
    tool_recall: float
    tools_ok: bool
    claims: int
    unsupported_numbers: list[str]
    grounded: bool
    missing_includes: list[str]
    includes_ok: bool
    latency_ms: float
    judge: dict[str, Any] | None = None
    passed: bool = False
    failures: list[str] = field(default_factory=list)


class AgentUnavailableError(RuntimeError):
    pass


def make_agent(kind: str = "auto"):
    """``rule`` -> RuleBasedAgent, ``llm`` -> LangGraph agent (error if not configured),
    ``auto`` -> whatever the app would use."""
    from src.agent.workflow import LangGraphAgent, RuleBasedAgent, build_agent  # noqa: PLC0415
    from src.config import get_settings  # noqa: PLC0415

    if kind == "rule":
        return RuleBasedAgent()
    agent = build_agent(get_settings())
    if kind == "llm" and not isinstance(agent, LangGraphAgent):
        raise AgentUnavailableError(
            "No LLM agent available: set LLM_PROVIDER and its API key "
            "(e.g. LLM_PROVIDER=groq GROQ_API_KEY=...) and install langgraph + the provider SDK."
        )
    return agent


def evaluate_case(case: GoldenCase, copilot, judge=None) -> CaseResult:
    """Ask one golden question and score the response."""
    response = copilot.ask(case.question, use_cache=False)
    called = [c.name for c in response.tool_calls]
    tools = tool_score(case.expected_tools, called)
    payloads = [c.output for c in response.tool_calls] + [c.arguments for c in response.tool_calls]
    answered = not response.blocked
    unsupported = unsupported_numbers(response.answer, payloads, case.question) if answered else []
    missing = missing_includes(response.answer, case.must_include) if answered else []
    result = CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        expect_blocked=case.expect_blocked,
        blocked=response.blocked,
        guardrail=response.guardrail.value,
        agent=response.agent,
        answer=response.answer,
        expected_tools=sorted(case.expected_tools),
        called_tools=called,
        tool_precision=tools.precision,
        tool_recall=tools.recall,
        tools_ok=tools.passed,
        claims=len(extract_claims(response.answer)) if answered else 0,
        unsupported_numbers=unsupported,
        grounded=not unsupported,
        missing_includes=missing,
        includes_ok=not missing,
        latency_ms=response.latency_ms,
    )
    if judge is not None and answered and not case.expect_blocked:
        verdict = judge.score(case.question, response.answer)
        result.judge = asdict(verdict) if verdict else None

    if response.blocked != case.expect_blocked:
        result.failures.append("not blocked" if case.expect_blocked else "falsely blocked")
    if not tools.passed:
        result.failures.append(f"missing tools: {', '.join(tools.missing)}")
    if unsupported:
        result.failures.append(f"unsupported numbers: {', '.join(unsupported)}")
    if missing:
        result.failures.append(f"missing content: {', '.join(missing)}")
    result.passed = not result.failures
    return result


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarise(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Aggregate metrics for a group of case results.

    Tool, groundedness and must_include metrics use the normal (non-adversarial) cases;
    groundedness only those that were answered. ``None`` means "not applicable here".
    """
    normal = [r for r in results if not r.expect_blocked]
    adversarial = [r for r in results if r.expect_blocked]
    answered = [r for r in normal if not r.blocked]
    claims = sum(r.claims for r in answered)
    bad_claims = sum(len(r.unsupported_numbers) for r in answered)
    latencies = [r.latency_ms for r in results]
    judged = [r.judge for r in results if r.judge]
    return {
        "cases": len(results),
        "pass_rate": _mean([float(r.passed) for r in results]),
        "tool_accuracy": _mean([float(r.tools_ok) for r in normal]),
        "tool_precision": _mean([r.tool_precision for r in normal]),
        "tool_recall": _mean([r.tool_recall for r in normal]),
        "groundedness": _mean([float(r.grounded) for r in answered]),
        "claims": claims,
        "unsupported_claims": bad_claims,
        "claim_support_rate": round(1 - bad_claims / claims, 4) if claims else None,
        "must_include_rate": _mean([float(r.includes_ok) for r in answered]),
        "block_rate": _mean([float(r.blocked) for r in adversarial]),
        "false_block_rate": _mean([float(r.blocked) for r in normal]),
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "judge_relevance": _mean([j["relevance"] for j in judged]),
        "judge_helpfulness": _mean([j["helpfulness"] for j in judged]),
        "judged": len(judged),
    }


def run_eval(
    cases: Sequence[GoldenCase],
    agent,
    judge=None,
    progress: Callable[[int, int, CaseResult], None] | None = None,
    delay_s: float = 0.0,
) -> dict[str, Any]:
    """Run every case through the copilot pipeline and build the report dict."""
    from src.agent.copilot import SupplyChainCopilot  # noqa: PLC0415

    copilot = SupplyChainCopilot(agent, cache=None, fallback=None)
    results: list[CaseResult] = []
    for i, case in enumerate(cases, start=1):
        results.append(evaluate_case(case, copilot, judge))
        if progress:
            progress(i, len(cases), results[-1])
        if delay_s and i < len(cases):
            time.sleep(delay_s)
    categories = sorted({r.category for r in results})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent.name,
        "judge": judge is not None,
        "overall": summarise(results),
        "by_category": {c: summarise([r for r in results if r.category == c]) for c in categories},
        "cases": [asdict(r) for r in results],
    }


def gate(report: dict[str, Any], fail_under: float | None) -> list[str]:
    """Gated metrics below the threshold, as ``"name=value"`` strings."""
    if fail_under is None:
        return []
    overall = report["overall"]
    return [
        f"{m}={overall[m]:.3f}"
        for m in GATED_METRICS
        if overall[m] is not None and overall[m] < fail_under
    ]


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value:.0%}" if pct else f"{value:,.0f}"


TABLE_COLUMNS = [
    ("category", 22),
    ("n", 3),
    ("pass", 6),
    ("tools", 6),
    ("grounded", 9),
    ("include", 8),
    ("blocked", 8),
    ("false blk", 9),
    ("p50 ms", 7),
]


def _row(name: str, s: dict[str, Any]) -> list[str]:
    return [
        name,
        str(s["cases"]),
        _fmt(s["pass_rate"]),
        _fmt(s["tool_accuracy"]),
        _fmt(s["groundedness"]),
        _fmt(s["must_include_rate"]),
        _fmt(s["block_rate"]),
        _fmt(s["false_block_rate"]),
        _fmt(s["latency_p50_ms"], pct=False),
    ]


def format_table(report: dict[str, Any]) -> str:
    """Plain-text table: one row per category plus the overall row."""
    rows = [_row(c, s) for c, s in report["by_category"].items()]
    rows.append(_row("OVERALL", report["overall"]))
    widths = [w for _, w in TABLE_COLUMNS]
    line = "  ".join(h.ljust(w) for (h, _), w in zip(TABLE_COLUMNS, widths, strict=True))
    out = [line, "-" * len(line)]
    for i, r in enumerate(rows):
        if i == len(rows) - 1:
            out.append("-" * len(line))
        out.append("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)))
    return "\n".join(out)


def format_markdown(report: dict[str, Any]) -> str:
    """Markdown summary: headline metrics, per-category table, failing cases."""
    o = report["overall"]
    lines = [
        "# Copilot answer-quality eval",
        "",
        f"Agent: `{report['agent']}` · generated {report['generated_at']} · "
        f"{o['cases']} cases · LLM judge: {'on' if report['judge'] else 'off'}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pass rate (all checks) | {_fmt(o['pass_rate'])} |",
        f"| Tool selection accuracy | {_fmt(o['tool_accuracy'])} "
        f"(precision {_fmt(o['tool_precision'])}, recall {_fmt(o['tool_recall'])}) |",
        f"| Groundedness (answers with no unsupported number) | {_fmt(o['groundedness'])} |",
        f"| Numeric claims supported | {o['claims'] - o['unsupported_claims']}/{o['claims']} |",
        f"| must_include pass rate | {_fmt(o['must_include_rate'])} |",
        f"| Guardrail block rate (adversarial) | {_fmt(o['block_rate'])} |",
        f"| False-block rate (normal) | {_fmt(o['false_block_rate'])} |",
        f"| Latency p50 / p95 | {o['latency_p50_ms']:,.0f} ms / {o['latency_p95_ms']:,.0f} ms |",
    ]
    if o["judged"]:
        lines.append(
            f"| LLM judge relevance / helpfulness (1-5) | {o['judge_relevance']:.2f} / "
            f"{o['judge_helpfulness']:.2f} ({o['judged']} judged) |"
        )
    lines += [
        "",
        "## By category",
        "",
        "| Category | n | Pass | Tools | Grounded | Include | Blocked | False block |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c, s in report["by_category"].items():
        lines.append(
            f"| {c} | {s['cases']} | {_fmt(s['pass_rate'])} | {_fmt(s['tool_accuracy'])} | "
            f"{_fmt(s['groundedness'])} | {_fmt(s['must_include_rate'])} | "
            f"{_fmt(s['block_rate'])} | {_fmt(s['false_block_rate'])} |"
        )
    failing = [c for c in report["cases"] if not c["passed"]]
    lines += ["", f"## Failing cases ({len(failing)})", ""]
    if not failing:
        lines.append("None.")
    for c in failing:
        tools = ", ".join(c["called_tools"]) or "none"
        lines.append(
            f"- **{c['id']}** ({c['category']}): {c['question']}  \n"
            f"  tools called: {tools} · {'; '.join(c['failures'])}"
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out: Path) -> Path:
    """Write the JSON report and a Markdown summary beside it; return the Markdown path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(format_markdown(report), encoding="utf-8")
    return md


def load_report(path: Path = REPORT_PATH) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _bootstrap() -> None:
    """Train the models / ingest the manual if this is a fresh checkout (as the app does)."""
    try:
        from src.api.main import bootstrap  # noqa: PLC0415

        bootstrap()
    except Exception:
        logger.exception("Bootstrap failed; tools that need models or documents will error.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate copilot answer quality.")
    parser.add_argument("--agent", choices=AGENT_CHOICES, default="auto")
    parser.add_argument("--category", help="only run one category (e.g. quote, adversarial)")
    parser.add_argument("--golden", type=Path, help="golden JSONL (default: data/evals/...)")
    parser.add_argument("--out", type=Path, default=REPORT_PATH, help="JSON report path")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit 1 if overall tool accuracy or groundedness is below this (0-1)",
    )
    parser.add_argument(
        "--judge",
        choices=("auto", "on", "off"),
        default="auto",
        help="LLM-as-judge: auto = only when an LLM provider is configured",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between cases")
    parser.add_argument("--no-bootstrap", action="store_true", help="skip model/doc setup")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.agent == "rule":
        for key, value in OFFLINE_DEFAULTS.items():
            os.environ.setdefault(key, value)

    cases = load_golden(args.golden, args.category)
    if not cases:
        print(f"No golden cases for category {args.category!r}.", file=sys.stderr)
        return 2
    if not args.no_bootstrap:
        _bootstrap()
    try:
        agent = make_agent(args.agent)
    except AgentUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    judge = None
    if args.judge != "off":
        from src.evals.judge import LLMJudge  # noqa: PLC0415

        judge = LLMJudge.from_settings()
        if judge is None and args.judge == "on":
            print("--judge on needs a configured LLM provider.", file=sys.stderr)
            return 2

    def progress(i: int, n: int, r: CaseResult) -> None:
        if args.verbose:
            print(f"[{i}/{n}] {r.id}: {'ok' if r.passed else '; '.join(r.failures)}")

    report = run_eval(cases, agent, judge, progress=progress, delay_s=args.delay)
    md = write_report(report, args.out)

    o = report["overall"]
    print(f"Agent: {report['agent']} | judge: {'on' if judge else 'off'} | {o['cases']} cases")
    print(format_table(report))
    print(
        f"Numeric claims supported: {o['claims'] - o['unsupported_claims']}/{o['claims']} · "
        f"latency p50 {o['latency_p50_ms']:,.0f} ms, p95 {o['latency_p95_ms']:,.0f} ms"
    )
    if o["judged"]:
        print(
            f"LLM judge: relevance {o['judge_relevance']:.2f}/5, "
            f"helpfulness {o['judge_helpfulness']:.2f}/5 ({o['judged']} judged)"
        )
    for c in report["cases"]:
        if not c["passed"]:
            print(f"FAIL {c['id']}: {'; '.join(c['failures'])}")
    print(f"Wrote {args.out} and {md}")

    below = gate(report, args.fail_under)
    if below:
        print(f"Below --fail-under {args.fail_under}: {', '.join(below)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
