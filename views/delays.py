"""Delay-risk monitor: upcoming arrivals at risk and a single-shipment check."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.costing.rates import get_rate_card
from src.delays.model import DelayInput, ModelNotTrainedError, get_delay_model, upcoming_shipments
from views.common import BLUE, page_header, rm, show, style

page_header(
    "Delay risk",
    "Random-forest classifier trained on shipment history: congestion, monsoon, holiday weeks, "
    "customs inspection, carrier and more. Explanations show what drives each prediction.",
)

try:
    model = get_delay_model()
except ModelNotTrainedError as exc:
    st.error(str(exc))
    st.stop()

rates = get_rate_card()
carriers = {k: v for k, v in rates.data["carriers"].items() if not k.startswith("_")}
dests = rates.destinations
COLUMNS = [
    "eta", "risk", "p_high", "carrier", "mode", "destination", "commodity", "congestion",
    "customs_inspection",
]  # fmt: skip
RISK_COLORS = {"Low": "#1baf7a", "Medium": "#eda100", "High": "#e34948"}


@st.cache_data(show_spinner=False)
def manifest(days: int) -> pd.DataFrame:
    scored = model.score_frame(upcoming_shipments(days=days))
    scored["_order"] = scored["risk"].map({"High": 0, "Medium": 1, "Low": 2})
    return scored.sort_values(["_order", "p_high"], ascending=[True, False], ignore_index=True)


monitor, check = st.tabs(["Upcoming arrivals", "Check a shipment"])

with monitor:
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    days = (
        c1.segmented_control("Window", [7, 14, 30], default=14, format_func=lambda d: f"{d} days")
        or 14
    )
    risk_filter = c2.pills(
        "Show", ["High", "Medium", "Low"], selection_mode="multi", default=["High", "Medium"]
    )
    m = manifest(days)
    counts = m["risk"].value_counts()
    k1, k2, k3 = st.columns(3)
    k1.metric("Arrivals", f"{len(m)}", border=True)
    k2.metric("High risk", f"{counts.get('High', 0)}", border=True)
    k3.metric("Medium risk", f"{counts.get('Medium', 0)}", border=True)
    view = m[m["risk"].isin(risk_filter or ["High", "Medium", "Low"])].copy()
    view["carrier"] = view["carrier"].map(lambda c: carriers[c]["label"])
    view["destination"] = view["destination"].map(lambda d: dests[d]["label"])
    st.dataframe(
        view[COLUMNS],
        hide_index=True,
        width="stretch",
        column_config={
            "eta": st.column_config.TextColumn("ETA"),
            "risk": st.column_config.TextColumn("Risk"),
            "p_high": st.column_config.ProgressColumn(
                "P(High)", min_value=0, max_value=1, format="percent"
            ),
            "carrier": "Carrier",
            "mode": "Mode",
            "destination": "Destination",
            "commodity": "Cargo",
            "congestion": st.column_config.NumberColumn("Berth occupancy", format="percent"),
            "customs_inspection": st.column_config.CheckboxColumn("Inspection"),
        },
    )
    st.caption("Synthetic manifest of upcoming arrivals, scored by the model.")

with check:
    form, out = st.columns([1, 1.6], gap="large")
    with form, st.container(border=True):
        dest = st.selectbox(
            "Destination",
            list(dests),
            index=list(dests).index("kuching"),
            format_func=lambda k: dests[k]["label"],
        )
        options = [k for k, c in carriers.items() if c["modes"][0] in dests[dest]["modes"]]
        carrier = st.selectbox("Carrier", options, format_func=lambda k: carriers[k]["label"])
        commodity = st.selectbox(
            "Cargo", list(rates.cargo_types), format_func=lambda k: rates.cargo_types[k]["label"]
        )
        eta = st.date_input("ETA", value=dt.date(2026, 12, 10))
        port = st.segmented_control(
            "Port", ["westport", "northport"], default="westport", format_func=str.capitalize
        ) or "westport"  # fmt: skip
        congestion = st.slider("Berth occupancy", 0.40, 0.99, 0.78, 0.01, format="%.2f")
        inspection = st.toggle("Selected for customs inspection")
        containers = st.number_input("Containers", 1, 50, 1)
        rain_override = st.toggle(
            "Set rainfall manually",
            help="By default the model looks up rain at the destination on the ETA "
            "(Open-Meteo forecast, or the archive for past and far-off dates).",
        )
        rain = st.slider("Rain on ETA day (mm)", 0, 150, 20) if rain_override else None

    pred = model.predict(
        DelayInput(
            destination=dest,
            carrier=carrier,
            commodity=commodity,
            eta=eta.isoformat(),
            port=port,
            congestion=congestion,
            customs_inspection=inspection,
            containers=int(containers),
            rain_mm=rain,
        )
    )

    with out:
        k1, k2, k3 = st.columns(3)
        k1.metric("Delay risk", pred.risk, border=True)
        k2.metric("Expected delay", f"{pred.expected_delay_hours:,.0f} h", border=True)
        k3.metric(
            "Demurrage exposure",
            rm(pred.demurrage_exposure_rm, 0),
            help="Expected port days beyond the free time × demurrage rate (rate card).",
            border=True,
        )

        st.caption(
            f":material/rainy: Rain at {dests[dest]['label']} on {pred.weather.date}: "
            f"**{pred.weather.rain_mm:.0f} mm** ({pred.weather.source})"
        )

        fig = go.Figure()
        for label in ["Low", "Medium", "High"]:
            p = pred.probabilities[label]
            fig.add_bar(
                y=["Probability"],
                x=[p],
                orientation="h",
                name=label,
                marker_color=RISK_COLORS[label],
                text=f"{label} {p:.0%}" if p >= 0.08 else "",
                textposition="inside",
                hovertemplate=f"{label}: {p:.1%}<extra></extra>",
            )
        style(fig, height=64)
        fig.update_layout(barmode="stack", showlegend=False, bargap=0.2)
        fig.update_xaxes(range=[0, 1], showticklabels=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        show(fig)

        st.markdown("**What drives this prediction**")
        if pred.drivers:
            fig = go.Figure(
                go.Bar(
                    x=[d.contribution for d in pred.drivers][::-1],
                    y=[d.factor for d in pred.drivers][::-1],
                    orientation="h",
                    marker_color=BLUE,
                    cliponaxis=False,
                    text=[d.detail for d in pred.drivers][::-1],
                    textposition="outside",
                    hovertemplate="%{y}: +%{x:.2f} risk<extra></extra>",
                )
            )
            style(fig, height=60 + 44 * len(pred.drivers))
            fig.update_xaxes(
                showticklabels=False, range=[0, max(d.contribution for d in pred.drivers) * 2.4]
            )
            fig.update_yaxes(showgrid=False)
            show(fig)
            st.caption(
                "Each bar is how much the risk score (P(Medium)/2 + P(High)) falls if that "
                "factor were favourable — averaged over removing it from this shipment and "
                "adding it to a best case."
            )
        else:
            st.caption("No single factor raises the risk noticeably.")

        with st.expander("Model performance"):
            b = model.bundle
            rows = [{"Model": "Random forest", **b["metrics"]}] + [
                {"Model": name.replace("_", " ").capitalize(), **v}
                for name, v in b["baselines"].items()
            ]
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                    "macro_f1": st.column_config.NumberColumn("Macro F1", format="%.3f"),
                    "high_recall": st.column_config.NumberColumn("High-risk recall", format="%.3f"),
                },
            )
            st.caption(
                f"Time-based split: trained on {b['n_train']:,} past shipments, "
                f"tested on the latest {b['n_test']:,}."
            )
