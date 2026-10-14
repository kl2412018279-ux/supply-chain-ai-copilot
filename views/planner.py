"""Route & carrier planner: weigh price, speed, CO2e and reliability."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.costing.quote import QuoteRequest
from src.costing.rates import get_rate_card
from src.costing.routing import Priorities, recommend
from views.common import BLUE, MUTED, page_header, show, style

rates = get_rate_card()
dests = rates.destinations

page_header(
    "Route planner",
    "Every mode and carrier that serves the destination, ranked by your priorities. Price comes "
    "from the rate card, reliability from each carrier's delivery history.",
)

form, result = st.columns([1, 1.9], gap="large")

with form, st.container(border=True):
    st.markdown("**Job**")
    dest = st.selectbox(
        "Deliver to", list(dests), index=list(dests).index("penang-prai"),
        format_func=lambda k: dests[k]["label"],
    )  # fmt: skip
    cargo = st.selectbox(
        "Cargo", list(rates.cargo_types), format_func=lambda k: rates.cargo_types[k]["label"]
    )
    c1, c2 = st.columns(2)
    size = c1.segmented_control(
        "Container", ["20", "40"], default="40", format_func=lambda s: f"{s}ft"
    ) or "40"  # fmt: skip
    qty = c2.number_input("Quantity", 1, 100, 1)
    st.markdown("**Priorities**")
    st.caption("Relative weights; they are normalised to 100%.")
    w_cost = st.slider("Low price", 0, 10, 4)
    w_speed = st.slider("Fast delivery", 0, 10, 2)
    w_co2 = st.slider("Low emissions", 0, 10, 2)
    w_rel = st.slider("Reliability", 0, 10, 2)

rec = recommend(
    QuoteRequest(destination=dest, cargo_type=cargo, container_size=size, containers=int(qty)),
    Priorities(cost=w_cost, speed=w_speed, co2=w_co2, reliability=w_rel),
)
best = rec.recommended

with result:
    st.success(rec.explanation, icon=":material/route:")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Price", f"RM{best.price_rm:,.0f}")
    k2.metric("Transit", f"{best.transit_days} day" + ("s" if best.transit_days > 1 else ""))
    k3.metric("CO2e", f"{best.co2e_kg:,.0f} kg")
    k4.metric("On time (history)", f"{best.on_time_rate:.0%}")

    names = [f"{o.carrier_label} · {o.mode.value}" for o in rec.options]
    fig = go.Figure(
        go.Scatter(
            x=[o.price_rm for o in rec.options],
            y=[o.co2e_kg for o in rec.options],
            mode="markers+text",
            text=names,
            textposition="top center",
            cliponaxis=False,
            marker=dict(
                size=[12 + 6 * o.transit_days for o in rec.options],
                color=[BLUE if o is best else MUTED for o in rec.options],
                line=dict(width=1.5, color="white"),
            ),
            customdata=[[o.transit_days, f"{o.on_time_rate:.0%}", o.fit] for o in rec.options],
            hovertemplate="%{text}<br>RM%{x:,.0f} · %{y:,.0f} kg CO2e<br>"
            "%{customdata[0]} day(s) · %{customdata[1]} on time · fit %{customdata[2]}"
            "<extra></extra>",
        )
    )
    style(fig, height=340, x_title="Price", y_title="CO2e (kg)")
    fig.update_xaxes(
        tickprefix="RM", tickformat=",.0f", showgrid=True, gridcolor="rgba(128,128,128,0.16)"
    )
    fig.update_layout(margin=dict(l=4, r=24, t=28, b=4))
    st.markdown("**Price vs emissions** · bubble size = transit days · blue = recommended")
    show(fig)

    table = pd.DataFrame(
        [
            {
                "Option": f"{o.carrier_label} ({o.mode.value})",
                "Price": o.price_rm,
                "Days": o.transit_days,
                "CO2e kg": o.co2e_kg,
                "On time": o.on_time_rate,
                "Fit": o.fit,
            }
            for o in rec.options
        ]
    )
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Price": st.column_config.NumberColumn(format="RM %,.0f"),
            "CO2e kg": st.column_config.NumberColumn(format="%,.0f"),
            "On time": st.column_config.NumberColumn(format="percent"),
            "Fit": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Days": st.column_config.NumberColumn(width="small"),
        },
    )  # fmt: skip
    dominated = [o.carrier_label for o in rec.options if not o.pareto_optimal]
    if dominated:
        st.caption(
            "Dominated (another option is at least as good on every criterion): "
            + ", ".join(dominated)
        )
    st.caption(
        "Fit = 100 minus the weighted distance from the best option on each criterion. "
        "Carriers and their "
        "factors are fictional examples in data/reference/rates.json."
    )
