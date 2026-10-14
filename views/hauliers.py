"""Haulier scorecard: delivery performance + compliance evidence."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent.workflow import build_llm
from src.config import get_settings
from src.hauliers.scorecard import WEIGHTS, build_scorecard, llm_summary, monthly_on_time
from views.common import BLUE, page_header, rm, show, style

page_header(
    "Hauliers",
    "Subcontractor scorecard. Punctuality, damage and claims come from SQL over the delivery "
    "log; compliance evidence is quoted from each haulier's own policy document.",
)

scores = build_scorecard()
weights = " · ".join(f"{k.replace('_', '-')} {v:.0%}" for k, v in WEIGHTS.items())

table = pd.DataFrame(
    [
        {
            "Rank": s.rank,
            "Haulier": s.label,
            "Grade": s.grade,
            "Score": s.score,
            "On time": s.on_time_rate,
            "Damage": s.damage_rate,
            "Compliance": s.compliance_score,
            "Deliveries": s.deliveries,
            "Claims": s.claims_rm,
        }
        for s in scores
    ]
)
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        "On time": st.column_config.NumberColumn(format="percent"),
        "Damage": st.column_config.NumberColumn(format="percent"),
        "Compliance": st.column_config.ProgressColumn(
            "Compliance evidence", min_value=0, max_value=1, format="percent"
        ),
        "Claims": st.column_config.NumberColumn(format="RM %,.0f"),
    },
)
st.caption(f"Composite score weights: {weights}. Grades: A ≥ 0.80, B ≥ 0.65, C ≥ 0.50, D below.")

st.divider()
pick = st.selectbox(
    "Haulier",
    [s.haulier for s in scores],
    format_func=lambda k: next(s.label for s in scores if s.haulier == k),
)
s = next(x for x in scores if x.haulier == pick)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Grade", f"{s.grade} · #{s.rank}")
k2.metric(
    "On time", f"{s.on_time_rate:.1%}", help=f"Late jobs average {s.avg_late_hours:.0f} h late"
)
k3.metric("Damage rate", f"{s.damage_rate:.1%}")
k4.metric("Claims", rm(s.claims_rm, 0))
k5.metric("Compliance", f"{sum(e.met for e in s.compliance)}/{len(s.compliance)}")
st.caption(
    f"{s.fleet} · {', '.join(s.modes)} · {s.deliveries:,} deliveries, {s.containers:,} containers"
)

left, right = st.columns([1.3, 1], gap="large")
with left:
    st.markdown("**On-time rate by month**")
    trend = monthly_on_time(pick)
    fig = go.Figure(
        go.Scatter(
            x=trend["month"],
            y=trend["on_time_rate"],
            mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(size=6),
            customdata=trend["deliveries"],
            hovertemplate="%{x}: %{y:.0%} of %{customdata} deliveries<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.9,
        line=dict(color="rgba(128,128,128,0.6)", dash="dot", width=1),
        annotation_text="90% target",
        annotation_position="bottom right",
    )
    style(fig, height=300)
    fig.update_yaxes(tickformat=".0%", range=[max(0, trend["on_time_rate"].min() - 0.1), 1.0])
    show(fig)

    st.markdown("**Assessment**")
    st.write(s.summary)
    llm = build_llm(get_settings())
    if llm is not None and st.button("Write an AI assessment", icon=":material/edit_note:"):
        with st.spinner("Writing…"):
            try:
                st.info(llm_summary(s, llm))
            except Exception as exc:
                st.warning(
                    f"LLM unavailable ({type(exc).__name__}); showing the rule-based summary."
                )

with right:
    st.markdown("**Compliance evidence**")
    for e in s.compliance:
        with st.container(border=True):
            icon = ":material/check_circle:" if e.met else ":material/radio_button_unchecked:"
            st.markdown(f"{icon} **{e.label}**")
            if e.met:
                st.caption(f"“{e.evidence}” — {e.source}")
            else:
                st.caption("No evidence found in the policy document.")
