"""Container-volume forecast with capacity and revenue implications."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.costing.quote import QuoteRequest, build_quote
from src.costing.rates import get_rate_card
from src.forecasting.predict import ModelNotTrainedError, get_forecaster
from views.common import BLUE, ORANGE, page_header, rm, show, style

page_header(
    "Forecast",
    "Daily container volume (TEU) per commodity from the LightGBM model, and what it means for "
    "trucks and revenue.",
)

try:
    forecaster = get_forecaster()
except ModelNotTrainedError as exc:
    st.error(str(exc))
    st.stop()

rates = get_rate_card()
road = {k: v for k, v in rates.destinations.items() if "truck" in v["modes"]}

commodity = st.segmented_control(
    "Commodity", ["all", *forecaster.commodities], default="all",
    format_func=lambda c: "All commodities" if c == "all" else c.replace("-", " ").capitalize(),
) or "all"  # fmt: skip
c1, c2, c3 = st.columns([1.4, 1, 1.2], vertical_alignment="bottom")
horizon = c1.segmented_control(
    "Horizon", [14, 30, 60, 90], default=30, format_func=lambda d: f"{d} days"
) or 30  # fmt: skip
grain = c2.segmented_control("View", ["Weekly", "Daily"], default="Weekly") or "Weekly"
basis = c3.selectbox("Typical delivery", list(road), format_func=lambda k: road[k]["label"])

selected = forecaster.commodities if commodity == "all" else [commodity]
lookback = max(horizon, 56)


@st.cache_data(show_spinner=False)
def series(commodities: tuple[str, ...], horizon: int, lookback: int):
    fc = pd.concat([forecaster.forecast(c, horizon).assign(commodity=c) for c in commodities])
    hist = pd.concat(
        [forecaster.recent_history(c, lookback).assign(commodity=c) for c in commodities]
    )
    return fc, hist


fc, hist = series(tuple(selected), horizon, lookback)
fc_total = fc.groupby("date")["forecast"].sum()
hist_total = hist.groupby("date")["teu"].sum()
prior = hist_total.iloc[-horizon:].sum()
total = fc_total.sum()
peak_day = fc_total.idxmax()

# Capacity: each 20ft-equivalent container is one truck job to the chosen delivery point.
trips_per_truck = max(1, math.floor(rates.value("truck", "shift_hours") / rates.trip_hours(basis)))
trucks_at_peak = math.ceil(fc_total.max() / trips_per_truck)
unit = build_quote(QuoteRequest(destination=basis, container_size="20"), rates)
revenue = sum(
    build_quote(QuoteRequest(destination=basis, container_size="20", cargo_type=c), rates).price_rm
    * fc[fc["commodity"] == c]["forecast"].sum()
    for c in selected
)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Forecast volume", f"{total:,.0f} TEU",
    delta=f"{(total / prior - 1) * 100:+.1f}% vs previous {horizon} days" if prior else None,
)  # fmt: skip
k2.metric(f"Busiest day · {pd.Timestamp(peak_day):%a %d %b}", f"{fc_total.max():,.0f} TEU")
k3.metric(
    "Prime movers at peak", f"{trucks_at_peak}",
    help=f"{trips_per_truck} trip(s) per truck per shift to {road[basis]['label']} "
    f"({rates.trip_hours(basis):.1f} h round trip incl. port turnaround).",
)  # fmt: skip
k4.metric(
    "Projected haulage revenue",
    rm(revenue, 0),
    help="Forecast TEU × current 20ft quote per commodity.",
)

h = hist_total.copy()
f = fc_total.copy()
h.index, f.index = pd.to_datetime(h.index), pd.to_datetime(f.index)
if grain == "Weekly":
    # Aggregate actual + forecast together so the week spanning "today" is not split.
    last_actual = h.index[-1]
    combined = pd.concat([h, f])
    weekly = combined.resample("W-SUN").sum()
    days = combined.resample("W-SUN").count()
    weekly = weekly[days == 7]  # full weeks only
    h = weekly[weekly.index <= last_actual]
    f = weekly[weekly.index > last_actual]
y_title = "TEU per week" if grain == "Weekly" else "TEU per day"

fig = go.Figure()
fig.add_scatter(
    x=h.index, y=h.values, name="Actual", mode="lines", line=dict(color=BLUE, width=2),
    hovertemplate="%{x|%d %b %Y}: %{y:,.0f} TEU<extra>Actual</extra>",
)  # fmt: skip
fig.add_scatter(
    x=[h.index[-1], *f.index], y=[h.values[-1], *f.values], name="Forecast", mode="lines",
    line=dict(color=ORANGE, width=2, dash="dash"),
    hovertemplate="%{x|%d %b %Y}: %{y:,.0f} TEU<extra>Forecast</extra>",
)  # fmt: skip
fig.add_vline(x=h.index[-1], line=dict(color="rgba(128,128,128,0.4)", width=1, dash="dot"))
style(fig, height=380, y_title=y_title)
fig.update_layout(hovermode="x unified")
show(fig)
st.caption(
    f"Model validation MAE {forecaster.metrics.get('mae', float('nan')):.1f} TEU/day. "
    "Chinese New Year and Hari Raya are model features, so holiday dips are forecast."
)

if commodity == "all":
    rows = []
    for c in selected:
        nxt = fc[fc["commodity"] == c]["forecast"].sum()
        before = hist[hist["commodity"] == c]["teu"].iloc[-horizon:].sum()
        rows.append(
            {
                "Commodity": c.replace("-", " ").capitalize(),
                "Forecast TEU": round(nxt),
                "Previous period": round(before),
                "Change": (nxt / before - 1) if before else None,
                "Share": nxt / total if total else 0,
            }
        )
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={
            "Change": st.column_config.NumberColumn(format="percent"),
            "Share": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )  # fmt: skip

with st.expander("Daily forecast table"):
    table = fc.pivot_table(index="date", columns="commodity", values="forecast").round(1)
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Download (CSV)", table.to_csv().encode(), file_name=f"teu_forecast_{horizon}d.csv",
        mime="text/csv", icon=":material/download:",
    )  # fmt: skip

st.caption(
    f"One 20ft general-cargo job to {road[basis]['label']} is quoted at {rm(unit.price_rm)}."
)
