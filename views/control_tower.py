"""Control tower: what is costing money today, and where margin leaks."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.costing.rates import get_rate_card
from src.delays.model import ModelNotTrainedError, get_delay_model, upcoming_shipments
from src.hauliers.scorecard import DELIVERIES_CSV
from src.operations.freetime import Status, generate_board
from src.operations.margin import LEAKS, get_ledger, margin_summary, weekly_trend
from views.common import (
    BAD,
    BLUE,
    GOOD,
    MUTED,
    WARN,
    page_header,
    page_link,
    rm,
    rm_short,
    section,
    show,
    style,
)  # fmt: skip

today = dt.date.today()
rates = get_rate_card()
dests = rates.destinations
URGENCY_TONE = {"Overdue": BAD, "Today": BAD, "Tomorrow": WARN, "2-3 days": WARN, "OK": GOOD}


@st.cache_data(show_spinner=False)
def board(day: dt.date) -> pd.DataFrame:
    return generate_board(day)


@st.cache_data(show_spinner=False)
def arrivals(day: dt.date) -> pd.DataFrame | None:
    try:
        scored = get_delay_model().score_frame(upcoming_shipments(start=day.isoformat(), days=7))
    except ModelNotTrainedError:
        return None
    return scored


@st.cache_data(show_spinner=False)
def on_time_weekly() -> pd.DataFrame:
    d = pd.read_csv(DELIVERIES_CSV, parse_dates=["date"])
    end = d["date"].max()
    back = (end - d["date"]).dt.days // 7
    w = d[back < 13].groupby(back[back < 13])["on_time"].mean().sort_index(ascending=False)
    return w.reset_index(drop=True)


b = board(today)
overdue = b[b["days_left"] < 0]
due48 = b[b["days_left"].between(0, 1)]
m = margin_summary(90)
trend = weekly_trend()
ot = on_time_weekly()
scored = arrivals(today)

page_header(
    "Control tower",
    "What needs action today: containers running out of free time, arrivals likely to "
    "slip, and where job margin is leaking.",
    eyebrow=today.strftime("%A, %d %B %Y"),
)

# ----------------------------------------------------------------------------- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Past free time",
    f"{len(overdue)} boxes",
    delta=f"{rm(overdue['daily_rate_rm'].sum(), 0)}/day accruing" if len(overdue) else None,
    delta_color="inverse",
    delta_arrow="off",
    border=True,
    help="Containers past their last free day: demurrage + port storage at the port, "
    "detention while the empty is out.",
)
c2.metric(
    "Free time ends in 48 h",
    f"{len(due48)} boxes",
    border=True,
    help="Last free day is today or tomorrow. Move them now or start paying.",
)
c3.metric(
    "Charges at planned move",
    rm_short(b["projected_rm"].sum()),
    delta=f"{rm_short(b['accrued_rm'].sum())} already accrued",
    delta_color="off",
    delta_arrow="off",
    border=True,
    help="Demurrage, port storage and detention at the planned pickup / return dates.",
)
at_risk = scored[scored["risk"] != "Low"] if scored is not None else pd.DataFrame()
high = int((at_risk["risk"] == "High").sum()) if len(at_risk) else 0
c4.metric(
    "Arrivals likely to slip",
    f"{len(at_risk)} of {len(scored) if scored is not None else 0}",
    delta=f"{high} high risk · next 7 days",
    delta_color="off",
    delta_arrow="off",
    border=True,
    help="Medium or high delay risk from the delay-risk model (Operations → Delay risk).",
)

c5, c6, c7, c8 = st.columns(4)
c5.metric(
    "Gross margin (90 d)",
    f"{m.margin_pct:.1f}%",
    delta=f"{m.margin_pct - m.planned_margin_pct:+.1f} pts vs quoted",
    delta_arrow="down" if m.margin_pct < m.planned_margin_pct else "up",
    border=True,
    chart_data=trend["margin_pct"].tolist(),
    chart_type="line",
    help=f"Jobs {m.period}. Quoted margin {m.planned_margin_pct:.1f}%: the gap is leakage.",
)
c6.metric(
    "Margin leaked (90 d)",
    rm_short(m.leakage_rm),
    delta=f"{m.leakage_rm / m.revenue_rm:.1%} of revenue",
    delta_color="inverse",
    delta_arrow="off",
    border=True,
    chart_data=trend["leakage_rm"].round(0).tolist(),
    chart_type="bar",
)
c7.metric(
    "On-time delivery (30 d)",
    f"{ot.iloc[-4:].mean():.1%}",
    delta=f"{(ot.iloc[-4:].mean() - ot.iloc[:-4].mean()) * 100:+.1f} pts vs prior",
    border=True,
    chart_data=(ot * 100).round(1).tolist(),
    chart_type="area",
    help="All hauliers and carriers, from the delivery log.",
)
c8.metric(
    "Loss-making jobs (90 d)",
    f"{m.loss_making_jobs}",
    delta=f"of {m.jobs:,} jobs",
    delta_color="off",
    delta_arrow="off",
    border=True,
)

# ----------------------------------------------------------------------------- free time
section("Free-time clock", f"{len(b)} live import containers · Port Klang")

view = b.head(14).iloc[::-1]
start = pd.to_datetime(
    view.apply(lambda r: r["gate_out"] if r["clock"] == "Detention" else r["discharged"], axis=1)
)
lfd = pd.to_datetime(view["last_free_day"]) + pd.Timedelta(days=1)  # end of the day
move = pd.to_datetime(view["planned_move"])
labels = [f"{c} · {s}" for c, s in zip(view["container"], view["status"], strict=True)]
day_ms = 86_400_000
fig = go.Figure()
fig.add_bar(
    y=labels, x=(lfd - start).dt.days * day_ms, base=start, orientation="h",
    marker_color="rgba(42,120,214,0.35)", name="Free time",
    hovertemplate="%{y}<br>Free until %{customdata}<extra></extra>",
    customdata=view["last_free_day"].astype(str),
)  # fmt: skip
charged = (move - lfd).dt.days.clip(lower=0)
fig.add_bar(
    y=labels, x=charged * day_ms, base=lfd, orientation="h",
    marker_color=[URGENCY_TONE[u] for u in view["urgency"]], name="Charged days",
    customdata=view[["projected_rm", "action"]].values,
    text=[rm_short(v) if v else "" for v in view["projected_rm"]], textposition="outside",
    cliponaxis=False, textfont=dict(size=11),
    hovertemplate="%{y}<br>Charged until planned move: RM%{customdata[0]:,.0f}"
    "<br>Next: %{customdata[1]}<extra></extra>",
)  # fmt: skip
fig.add_vline(x=pd.Timestamp(today), line=dict(color=MUTED, width=1.5, dash="dot"))
fig.add_annotation(
    x=pd.Timestamp(today), y=1.0, yref="paper", text="today", showarrow=False,
    yanchor="bottom", font=dict(size=11),
)  # fmt: skip
style(fig, height=450)
fig.update_layout(
    margin=dict(t=24), barmode="overlay", bargap=0.35, legend=dict(x=0, y=1.07), showlegend=False
)
fig.update_xaxes(type="date", showgrid=True, gridcolor="rgba(128,128,128,0.12)",
                 tickformat="%a %d", dtick=day_ms, tickangle=0)  # fmt: skip
fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
show(fig)
st.caption(
    "Blue: free time from discharge (or gate-out for empties). Red / amber: days charged until "
    "the planned pickup or empty return, with the RM it will cost. Most urgent at the top."
)

status_filter = st.pills(
    "Filter by status",
    [s.value for s in Status],
    selection_mode="multi",
)
todo = b[b["status"].isin(status_filter)] if status_filter else b
st.dataframe(
    todo[["container", "size", "customer", "status", "last_free_day", "days_left",
          "accrued_rm", "projected_rm", "action"]],
    hide_index=True,
    width="stretch",
    column_config={
        "container": st.column_config.TextColumn("Container", pinned=True),
        "size": st.column_config.TextColumn("Size", width="small"),
        "customer": "Customer",
        "status": "Status",
        "last_free_day": st.column_config.DateColumn("Last free day", format="ddd D MMM"),
        "days_left": st.column_config.NumberColumn(
            "Days left", help="Negative = days past the last free day", width="small"
        ),
        "accrued_rm": st.column_config.NumberColumn("Accrued", format="RM %,.0f"),
        "projected_rm": st.column_config.NumberColumn(
            "At planned move", format="RM %,.0f", help="Total charges if moved on the planned date"
        ),
        "action": "Next action",
    },
)  # fmt: skip

# ----------------------------------------------------------------------------- margin
section("Where margin leaks", f"{m.jobs:,} completed jobs · {m.period}")

wf, cust = st.columns([1.15, 1], gap="large")
with wf:
    planned = m.revenue_rm * m.planned_margin_pct / 100
    leaks = [-m.leakage_by_reason[k] for k in LEAKS]
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["absolute", *["relative"] * len(LEAKS), "total"],
            y=["Quoted margin", *LEAKS, "Actual margin"],
            x=[planned, *leaks, 0],
            text=[rm_short(planned), *[f"−{rm_short(-v)}" for v in leaks], rm_short(m.margin_rm)],
            textposition="outside",
            cliponaxis=False,
            connector=dict(line=dict(color=MUTED, width=1)),
            increasing=dict(marker=dict(color=GOOD)),
            decreasing=dict(marker=dict(color=BAD)),
            totals=dict(marker=dict(color=BLUE)),
            hovertemplate="%{y}: RM%{x:,.0f}<extra></extra>",
        )
    )  # fmt: skip
    style(fig, height=330)
    fig.update_xaxes(showticklabels=False, range=[0, planned * 1.18])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    st.markdown("**Quoted → actual margin** · what each leak cost")
    show(fig)

with cust:
    by = pd.DataFrame(m.by_customer).sort_values("margin_pct")
    target = m.planned_margin_pct
    fig = go.Figure(
        go.Bar(
            x=by["margin_pct"], y=by["customer"], orientation="h", cliponaxis=False,
            marker_color=[BAD if v < target - 3 else WARN if v < target else GOOD
                          for v in by["margin_pct"]],
            text=[f"{v:.1f}%" for v in by["margin_pct"]], textposition="outside",
            customdata=by[["revenue_rm", "leakage_rm"]].values,
            hovertemplate="%{y}: %{x:.1f}% margin<br>Revenue RM%{customdata[0]:,.0f} · "
            "leaked RM%{customdata[1]:,.0f}<extra></extra>",
        )
    )  # fmt: skip
    fig.add_vline(x=target, line=dict(color=MUTED, dash="dot"))
    style(fig, height=330)
    fig.update_xaxes(showticklabels=False, range=[0, by["margin_pct"].max() * 1.25])
    fig.update_yaxes(showgrid=False)
    st.markdown(f"**Margin by customer** · dotted line = quoted {target:.1f}%")
    show(fig)

worst = pd.DataFrame(m.worst_lanes)
lane_txt = ", ".join(
    f"{r.lane.replace('-', ' ').title()} ({r.margin_pct:.1f}%)" for r in worst.head(3).itertuples()
)
biggest = max(m.leakage_by_reason, key=m.leakage_by_reason.get)
st.info(
    f"Biggest leak: **{biggest.lower()}** ({rm_short(m.leakage_by_reason[biggest])}). "
    f"Weakest lanes: {lane_txt}. Leaks are what separates the quoted "
    f"{m.planned_margin_pct:.1f}% from the {m.margin_pct:.1f}% actually earned.",
    icon=":material/troubleshoot:",
)

with st.expander("Loss-making jobs"):
    ledger = get_ledger()
    end = ledger["date"].max()
    losses = ledger[(ledger["date"] > end - pd.Timedelta(days=90)) & (ledger["margin_rm"] < 0)]
    losses = losses.assign(
        main_leak=losses[LEAKS].idxmax(axis=1),
        destination=losses["destination"].map(lambda d: dests[d]["label"]),
    ).sort_values("margin_rm")
    st.dataframe(
        losses[["date", "job", "customer", "destination", "haulier", "revenue_rm", "margin_rm",
                "main_leak"]],
        hide_index=True,
        width="stretch",
        column_config={
            "date": st.column_config.DateColumn("Date", format="D MMM"),
            "job": "Job",
            "customer": "Customer",
            "destination": "Destination",
            "haulier": "Haulier",
            "revenue_rm": st.column_config.NumberColumn("Revenue", format="RM %,.0f"),
            "margin_rm": st.column_config.NumberColumn("Margin", format="RM %,.0f"),
            "main_leak": "Main leak",
        },
    )  # fmt: skip

# ----------------------------------------------------------------------------- arrivals
if scored is not None:
    section("Arrivals at risk", "next 7 days · delay-risk model")
    risky = scored[scored["risk"] != "Low"].sort_values("p_high", ascending=False).head(6)
    carriers = rates.data["carriers"]
    cols = st.columns(3)
    for i, r in enumerate(risky.itertuples()):
        with cols[i % 3], st.container(border=True):
            tone = "bad" if r.risk == "High" else "warn"
            st.html(
                f'<span class="pill {tone}">{r.risk} risk</span> '
                f'<span style="opacity:.6;font-size:.85rem">ETA {r.eta}</span>'
            )
            st.markdown(
                f"**{dests[r.destination]['label']}** · {r.commodity}  \n"
                f"{carriers[r.carrier]['label']} by {r.mode} · P(High) {r.p_high:.0%}"
            )
    page_link("views/delays.py", label="Open delay risk", icon=":material/arrow_forward:")
