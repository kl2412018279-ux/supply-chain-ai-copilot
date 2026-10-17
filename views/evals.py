"""AI quality: the copilot's answer-quality eval (tool choice, grounded numbers, guardrails)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.evals.dataset import load_golden
from src.evals.run import REPORT_PATH, load_report, make_agent, run_eval, write_report
from views.common import page_header, pill, section

head, action = st.columns([5, 1], vertical_alignment="bottom")
with head:
    page_header(
        "AI quality",
        "Golden questions run through the full copilot pipeline. Every RM amount, percentage "
        "and quantity in an answer is checked against what the tools returned, so invented "
        "numbers show up here, not in front of a customer.",
        eyebrow="Assistant",
    )
with action:
    run_clicked = st.button(
        "Run evaluation", icon=":material/play_arrow:", type="primary", width="stretch"
    )

if run_clicked:
    from src.evals.judge import LLMJudge  # noqa: PLC0415

    cases = load_golden()
    agent = make_agent("auto")  # LLM agent if one is configured, else the offline agent
    judge = LLMJudge.from_settings()
    bar = st.progress(0.0, text="Starting…")
    with st.spinner(f"Asking {len(cases)} golden questions ({agent.name} agent)…"):
        report = run_eval(
            cases,
            agent,
            judge,
            progress=lambda i, n, r: bar.progress(i / n, text=f"{i}/{n} · {r.id}"),
        )
    bar.empty()
    try:
        write_report(report, REPORT_PATH)
    except OSError as exc:  # read-only deploys still show the in-memory result
        st.warning(f"Could not save the report: {exc}")
    st.session_state["eval_report"] = report

report = st.session_state.get("eval_report") or load_report()
if report is None:
    st.info(
        "No evaluation report yet. Click **Run evaluation**, or run "
        "`python -m src.evals.run` and reload this page.",
        icon=":material/info:",
    )
    st.stop()


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


o = report["overall"]
st.caption(
    f"Agent **{report['agent']}** · {o['cases']} cases · generated {report['generated_at']} · "
    f"LLM judge {'on' if report.get('judge') else 'off'}"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Pass rate", pct(o["pass_rate"]), help="Cases passing every check", border=True)
k2.metric(
    "Tool selection",
    pct(o["tool_accuracy"]),
    help=f"Expected tools called. Precision {pct(o['tool_precision'])}, "
    f"recall {pct(o['tool_recall'])}.",
    border=True,
)
k3.metric(
    "Groundedness",
    pct(o["groundedness"]),
    help="Answers in which every RM amount, % and quantity is backed by a tool output",
    border=True,
)
k4.metric(
    "Numbers supported",
    f"{o['claims'] - o['unsupported_claims']}/{o['claims']}",
    help="Numeric claims across all answers that trace back to tool outputs",
    border=True,
)
k5, k6, k7, k8 = st.columns(4)
k5.metric("must_include", pct(o["must_include_rate"]), help="Required facts present", border=True)
k6.metric("Attacks blocked", pct(o["block_rate"]), help="Adversarial prompts refused", border=True)
k7.metric("False blocks", pct(o["false_block_rate"]), help="Normal questions refused", border=True)
k8.metric(
    "Latency p50 / p95",
    f"{o['latency_p50_ms']:,.0f} / {o['latency_p95_ms']:,.0f} ms",
    border=True,
)
if o.get("judged"):
    j1, j2, _, _ = st.columns(4)
    j1.metric("Judge: relevance", f"{o['judge_relevance']:.2f} / 5", border=True)
    j2.metric("Judge: helpfulness", f"{o['judge_helpfulness']:.2f} / 5", border=True)

section("By category", "tool, groundedness and content checks use the normal questions")
table = pd.DataFrame(
    [
        {
            "Category": name,
            "Cases": s["cases"],
            "Pass": s["pass_rate"],
            "Tools": s["tool_accuracy"],
            "Grounded": s["groundedness"],
            "Includes": s["must_include_rate"],
            "Blocked": s["block_rate"],
            "p50 ms": s["latency_p50_ms"],
        }
        for name, s in report["by_category"].items()
    ]
)
percent = st.column_config.NumberColumn(format="percent")
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "Pass": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
        "Tools": percent,
        "Grounded": percent,
        "Includes": percent,
        "Blocked": percent,
        "p50 ms": st.column_config.NumberColumn(format="%.0f"),
    },
)

failing = [c for c in report["cases"] if not c["passed"]]
section("Failing cases", f"{len(failing)} of {len(report['cases'])}")
if not failing:
    st.success("Every golden case passed.", icon=":material/check_circle:")
for c in failing:
    with st.container(border=True):
        st.html(f"<b>{c['id']}</b> &nbsp;{pill(c['category'])} &nbsp;{pill('fail', 'bad')}")
        st.markdown(f"*{c['question']}*")
        called = ", ".join(f"`{t}`" for t in c["called_tools"]) or "none"
        expected = ", ".join(f"`{t}`" for t in c["expected_tools"]) or "none"
        st.markdown(f"Tools called: {called} · expected: {expected}")
        if c["unsupported_numbers"]:
            st.html(
                "Unsupported numbers: " + " ".join(pill(n, "bad") for n in c["unsupported_numbers"])
            )
        for failure in c["failures"]:
            st.caption(failure)
        with st.expander("Answer"):
            st.markdown(c["answer"])

with st.expander(f"All {len(report['cases'])} cases"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Case": c["id"],
                    "Category": c["category"],
                    "Passed": c["passed"],
                    "Question": c["question"],
                    "Tools": ", ".join(c["called_tools"]),
                    "Unsupported": ", ".join(c["unsupported_numbers"]),
                    "ms": c["latency_ms"],
                }
                for c in report["cases"]
            ]
        ),
        hide_index=True,
        width="stretch",
    )
