"""Build an itemised RM quote, see the route, compare modes and test what-ifs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.costing.quote import Mode, QuoteRequest, build_quote, compare_modes
from src.costing.rates import get_rate_card
from views.common import BLUE, MUTED, ORANGE, page_header, rm, show, style

base_rates = get_rate_card()
base_diesel = base_rates.value("fuel", "diesel_rm_per_litre_peninsular")
base_margin = float(base_rates.data["margin_pct"])
dests = base_rates.destinations

page_header(
    "Quote",
    "Itemised job price in ringgit from the rate card, with CO2e from the same diesel litres. "
    "Change any input and everything updates.",
)

form, result = st.columns([1, 1.7], gap="large")

# ------------------------------------------------------------------------------ inputs
with form, st.container(border=True):
    st.markdown("**Job**")
    dest = st.selectbox("Deliver to", list(dests), format_func=lambda k: dests[k]["label"])
    origin = st.segmented_control(
        "Collect from", list(base_rates.origins), default="westport",
        format_func=lambda k: base_rates.origins[k]["label"].split(" (")[0],
    ) or "westport"  # fmt: skip
    cargo = st.selectbox(
        "Cargo", list(base_rates.cargo_types),
        format_func=lambda k: base_rates.cargo_types[k]["label"],
    )  # fmt: skip
    c1, c2 = st.columns(2)
    size = c1.segmented_control(
        "Container", ["20", "40"], default="40", format_func=lambda s: f"{s}ft"
    ) or "40"  # fmt: skip
    containers = c2.number_input("Quantity", min_value=1, max_value=100, value=1)
    weight = st.number_input(
        "Cargo weight per container (kg)", min_value=500, max_value=30000, value=18000, step=500
    )
    modes = dests[dest]["modes"]
    mode = (
        st.segmented_control("Mode", modes, default=modes[0], format_func=str.capitalize)
        or modes[0]
    )
    customs = st.toggle("Customs clearance (K1/K2)", value=True)

    with st.expander("Warehousing"):
        w1, w2 = st.columns(2)
        pallets = w1.number_input("Pallets", min_value=0, max_value=5000, value=0, step=10)
        days = w2.number_input("Days", min_value=0, max_value=365, value=0)

    with st.expander("What-if"):
        diesel = st.slider(
            "Diesel (RM/L)", 2.00, 4.00, value=base_diesel, step=0.05,
            help=f"Rate card: RM{base_diesel:.2f}/L (MOF weekly price)",
        )  # fmt: skip
        margin = st.slider("Margin (%)", 0, 40, value=int(base_margin), step=1)

rates = base_rates.with_overrides(diesel_rm_per_litre=diesel, margin_pct=margin)
req = QuoteRequest(
    destination=dest, origin=origin, cargo_type=cargo, container_size=size,
    containers=int(containers), cargo_weight_kg=float(weight), mode=Mode(mode),
    customs_clearance=customs, warehouse_pallets=int(pallets), warehouse_days=int(days),
)  # fmt: skip
q = build_quote(req, rates)
baseline = build_quote(req, base_rates)
scenario = diesel != base_diesel or margin != base_margin

# ----------------------------------------------------------------------------- results
with result:
    for w in q.warnings:
        st.warning(w, icon=":material/warning:")

    k1, k2, k3, k4 = st.columns(4)
    delta = q.price_rm - baseline.price_rm
    k1.metric(
        "Quoted price", rm(q.price_rm, 0),
        delta=f"{rm(delta, 0)} vs rate card" if scenario and abs(delta) >= 0.5 else None,
        delta_color="inverse",
    )  # fmt: skip
    k2.metric("Per container", rm(q.price_per_container_rm, 0))
    k3.metric("Cost", rm(q.cost_rm, 0), help=f"{q.margin_pct:.0f}% margin = {rm(q.margin_rm)}")
    k4.metric("CO2e", f"{q.co2e_kg:,.0f} kg", help=f"Diesel ≈ {q.diesel_litres:,.0f} L")
    st.caption(
        f"{q.containers} × {q.container_size}ft {base_rates.cargo_types[cargo]['label'].lower()} · "
        f"{q.origin_label} → {q.destination_label} · {q.distance_km:,.0f} km by {q.mode.value}"
    )

    view = st.segmented_control(
        "View", ["Breakdown", "Route", "Modes", "Diesel sensitivity"], default="Breakdown",
        label_visibility="collapsed",
    ) or "Breakdown"  # fmt: skip

    items = pd.DataFrame([i.model_dump() for i in q.line_items])

    if view == "Breakdown":
        by_cat = items.groupby("category", sort=False)["amount_rm"].sum().sort_values()
        share = by_cat / by_cat.sum() * 100
        fig = go.Figure(
            go.Bar(
                x=by_cat.values, y=by_cat.index.str.capitalize(), orientation="h",
                marker_color=BLUE, cliponaxis=False,
                text=[f"{rm(v, 0)} · {s:.0f}%" for v, s in zip(by_cat, share, strict=True)],
                textposition="outside",
                hovertemplate="%{y}: RM%{x:,.2f}<extra></extra>",
            )
        )  # fmt: skip
        style(fig, height=max(220, 44 * len(by_cat)))
        fig.update_xaxes(showticklabels=False, range=[0, by_cat.max() * 1.35])
        fig.update_yaxes(showgrid=False)
        show(fig)
        st.dataframe(
            items[["category", "description", "amount_rm"]].assign(
                category=items["category"].str.capitalize()
            ),
            hide_index=True, width="stretch",
            column_config={
                "category": st.column_config.TextColumn("Category", width="small"),
                "description": st.column_config.TextColumn("Line item"),
                "amount_rm": st.column_config.NumberColumn("Amount", format="RM %.2f"),
            },
        )  # fmt: skip

    elif view == "Route":
        o = base_rates.origins[origin]
        d = dests[dest]
        others = [k for k in dests if k != dest]
        fig = go.Figure()
        fig.add_trace(
            go.Scattermap(
                lat=[dests[k]["lat"] for k in others], lon=[dests[k]["lon"] for k in others],
                mode="markers", marker=dict(size=8, color=MUTED), name="Other destinations",
                text=[dests[k]["label"] for k in others], hovertemplate="%{text}<extra></extra>",
            )
        )  # fmt: skip
        fig.add_trace(
            go.Scattermap(
                lat=[o["lat"], d["lat"]], lon=[o["lon"], d["lon"]], mode="lines",
                line=dict(width=3, color=BLUE), hoverinfo="skip", name="Route",
            )
        )  # fmt: skip
        fig.add_trace(
            go.Scattermap(
                lat=[o["lat"], d["lat"]], lon=[o["lon"], d["lon"]], mode="markers+text",
                marker=dict(size=14, color=[ORANGE, BLUE]),
                text=[o["label"].split(" (")[0], d["label"]], textposition="top right",
                textfont=dict(size=13), hovertemplate="%{text}<extra></extra>", name="Job",
            )
        )  # fmt: skip
        lats, lons = [o["lat"], d["lat"]], [o["lon"], d["lon"]]
        span = max(max(lats) - min(lats), max(lons) - min(lons))
        zoom = 8.6 if span < 0.4 else 7.4 if span < 1.2 else 6.2 if span < 3.5 else 4.6
        fig.update_layout(
            map=dict(
                style="carto-positron",
                center=dict(lat=sum(lats) / 2, lon=sum(lons) / 2),
                zoom=zoom,
            ),
            height=440, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        )  # fmt: skip
        show(fig)
        st.caption(
            f"Straight-line view; quoted on {q.distance_km:,.0f} km by {q.mode.value}"
            + (" (truck round trip, empty return)." if q.mode is Mode.truck else ".")
            + " Scroll to zoom, drag to pan."
        )

    elif view == "Modes":
        if len(modes) < 2:
            st.info(
                f"Only {modes[0]} is offered to {dests[dest]['label']}.", icon=":material/info:"
            )
        else:
            cmp = compare_modes(req, rates)
            names = [x.mode.value.capitalize() for x in cmp.quotes]
            colors = [BLUE if x.mode.value == mode else MUTED for x in cmp.quotes]
            m1, m2 = st.columns(2)
            for col, title, values, fmt in (
                (m1, "Price", [x.price_rm for x in cmp.quotes], "RM{:,.0f}"),
                (m2, "CO2e (kg)", [x.co2e_kg for x in cmp.quotes], "{:,.0f} kg"),
            ):
                fig = go.Figure(
                    go.Bar(
                        x=names, y=values, marker_color=colors, cliponaxis=False,
                        text=[fmt.format(v) for v in values], textposition="outside",
                        hovertemplate="%{x}: %{y:,.2f}<extra></extra>",
                    )
                )  # fmt: skip
                style(fig, height=260)
                fig.update_yaxes(
                    showticklabels=False, showgrid=False, range=[0, max(values) * 1.25]
                )
                with col:
                    st.markdown(f"**{title}**")
                    show(fig)
            cheap, green = cmp.cheapest_mode.value, cmp.lowest_co2_mode.value
            if cmp.carbon_abatement_rm_per_tonne is None:
                st.success(
                    f"{green.capitalize()} is both the cheapest and the lowest-emission option."
                )
            else:
                st.info(
                    f"{green.capitalize()} cuts CO2e but costs "
                    f"{rm(cmp.carbon_abatement_rm_per_tonne, 0)} per tonne avoided vs {cheap}."
                )

    else:  # diesel sensitivity
        prices = np.round(np.arange(2.0, 4.01, 0.1), 2)
        totals = [
            build_quote(
                req, base_rates.with_overrides(diesel_rm_per_litre=p, margin_pct=margin)
            ).price_rm
            for p in prices
        ]
        fig = go.Figure()
        fig.add_scatter(
            x=prices, y=totals, mode="lines", line=dict(color=BLUE, width=2), name="Quote",
            hovertemplate="RM%{x:.2f}/L → RM%{y:,.0f}<extra></extra>",
        )  # fmt: skip
        fig.add_scatter(
            x=[diesel], y=[q.price_rm], mode="markers", name="Selected",
            marker=dict(size=11, color=ORANGE, line=dict(width=2, color="white")),
            hovertemplate="RM%{x:.2f}/L → RM%{y:,.0f}<extra></extra>",
        )  # fmt: skip
        style(fig, height=300, x_title="Diesel price per litre", y_title="Quoted price")
        fig.update_xaxes(tickprefix="RM", tickformat=".2f")
        fig.update_yaxes(tickprefix="RM", tickformat=",.0f")
        fig.update_layout(showlegend=False, hovermode="x")
        show(fig)
        step = (totals[-1] - totals[0]) / (len(prices) - 1)
        st.caption(
            f"Every RM0.10/L change in diesel moves this quote by about {rm(step, 0)}. "
            "Quotes are valid for 7 days because diesel is repriced weekly."
        )

    csv = items.assign(price_total_rm=q.price_rm).to_csv(index=False).encode()
    st.download_button(
        "Download quote (CSV)", csv, file_name=f"quote_{dest}_{size}ft.csv",
        mime="text/csv", icon=":material/download:",
    )  # fmt: skip
