"""Streamlit dashboard - Freight Ops Copilot for a Port Klang 3PL.

Runs fully in-process (no API server needed), so it deploys as-is to Streamlit
Community Cloud or Hugging Face Spaces.

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent.copilot import get_copilot
from src.agent.tools import estimate_haulage_spend
from src.api.main import bootstrap
from src.costing.quote import Mode, QuoteRequest, build_quote, compare_modes
from src.costing.rates import get_rate_card
from src.forecasting.predict import ModelNotTrainedError, get_forecaster
from src.rag.ingest_pdf import ingest_bytes
from src.rag.retriever import get_retriever
from src.rag.vector_store import get_vector_store

# Reference categorical palette, slots 1-2 (validated light/dark, CVD-safe).
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
GRID = "rgba(128,128,128,0.18)"
COMPANY = "Muara Logistics Sdn Bhd"

st.set_page_config(page_title="Freight Ops Copilot", page_icon="🚛", layout="wide")


@st.cache_resource(show_spinner="Preparing model and documents…")
def _startup() -> bool:
    bootstrap()
    return True


def _style(fig: go.Figure, y_title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(title=y_title, gridcolor=GRID, zeroline=False),
        xaxis=dict(showgrid=False),
    )
    return fig


def _bar(labels: list[str], values: list[float], y_title: str, fmt: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=labels, y=values, marker_color=SERIES_1, marker_line=dict(width=0),
            text=[fmt.format(v) for v in values], textposition="outside",
            hovertemplate="%{x}: %{y:,.2f}<extra></extra>",
        )
    )  # fmt: skip
    fig.update_traces(cliponaxis=False)
    fig.update_layout(bargap=0.45)
    return _style(fig, y_title, height=300)


_startup()
copilot = get_copilot()
rates = get_rate_card()

# ---------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🚛 Freight Ops Copilot")
    st.caption(
        f"For **{COMPANY}**, a *fictional* Port Klang 3PL: customs clearance, haulage, "
        "warehousing and delivery of chemicals, palm oil and plastics. All rates and "
        "volumes are example data."
    )
    st.divider()
    st.markdown(f"**Agent:** `{copilot.agent.name}`")
    if copilot.agent.name == "rule-based":
        st.caption("Offline mode. Set `GROQ_API_KEY` (free) to enable the LangGraph LLM agent.")
    st.markdown(f"**Vector store:** `{type(get_vector_store()).__name__}`")
    if copilot.cache is not None:
        stats = copilot.cache.stats()
        st.markdown(f"**Semantic cache:** {stats['hits']} hits / {stats['misses']} misses")
    try:
        m = get_forecaster().metrics
        st.markdown(f"**Volume model:** MAE {m['mae']:.1f} TEU/day")
    except ModelNotTrainedError:
        st.warning("Volume model not trained.")
    diesel = rates.value("fuel", "diesel_rm_per_litre_peninsular")
    st.markdown(f"**Diesel (rate card):** RM{diesel:.2f}/L · as of {rates.data['as_of']}")

st.title("🚛 Freight Ops Copilot")
tab_chat, tab_quote, tab_forecast, tab_docs = st.tabs(
    ["💬 Copilot", "💰 Quote & Cost", "📈 Volume Forecast", "📄 Documents"]
)

# ------------------------------------------------------------------------------- chat
with tab_chat:
    st.caption(
        "Try: *Quote 2 x 40ft palm oil containers from Westport to Pasir Gudang* · "
        "*Compare truck vs rail for a 40ft chemicals container to Penang* · "
        "*Forecast chemicals volume for the next 2 weeks* · "
        "*What documents do I need to clear a chemical import?*"
    )
    history = st.session_state.setdefault("messages", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    if prompt := st.chat_input("Ask about quotes, haulage, volumes, customs or ESG…"):
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            res = copilot.ask(prompt)
            st.markdown(res.answer)
            badges = [f"agent: {res.agent}", f"{res.latency_ms:.0f} ms"]
            if res.cached:
                badges.append("⚡ semantic cache hit")
            if res.blocked:
                badges.append(f"🛡️ blocked: {res.guardrail.value}")
            meta = " · ".join(badges)
            st.caption(meta)
            if res.tool_calls:
                with st.expander(f"🔧 {len(res.tool_calls)} tool call(s)"):
                    for call in res.tool_calls:
                        st.markdown(f"**{call.name}** `{call.arguments}`")
            if res.sources:
                with st.expander(f"📚 {len(res.sources)} source(s)"):
                    for s in res.sources:
                        st.markdown(f"**{s.source}** p.{s.page} — score {s.score:.3f}")
                        st.caption(s.snippet[:500])
        history.append({"role": "assistant", "content": res.answer, "meta": meta})

# ------------------------------------------------------------------------------ quote
with tab_quote:
    dest_keys = list(rates.destinations)
    c1, c2, c3 = st.columns(3)
    dest = c1.selectbox(
        "Destination", dest_keys, format_func=lambda k: rates.destinations[k]["label"]
    )
    origin = c2.selectbox(
        "From", list(rates.origins), format_func=lambda k: rates.origins[k]["label"]
    )
    cargo = c3.selectbox(
        "Cargo", list(rates.cargo_types), format_func=lambda k: rates.cargo_types[k]["label"]
    )
    c4, c5, c6 = st.columns(3)
    size = c4.radio(
        "Container", ["20", "40"], index=1, horizontal=True, format_func=lambda s: f"{s}ft"
    )
    containers = c5.number_input("Containers", min_value=1, max_value=100, value=1)
    weight = c6.number_input(
        "Cargo weight per container (kg)", min_value=100, max_value=30000, value=18000, step=500
    )
    c7, c8, c9 = st.columns(3)
    modes = rates.destinations[dest]["modes"]
    mode = c7.selectbox("Mode", modes)
    pallets = c8.number_input("Warehouse pallets", min_value=0, max_value=5000, value=0)
    days = c9.number_input("Storage days", min_value=0, max_value=365, value=0)
    customs = st.checkbox("Include customs clearance (K1/K2)", value=True)

    req = QuoteRequest(
        destination=dest, origin=origin, cargo_type=cargo, container_size=size,
        containers=int(containers), cargo_weight_kg=float(weight), mode=Mode(mode),
        customs_clearance=customs, warehouse_pallets=int(pallets), warehouse_days=int(days),
    )  # fmt: skip
    q = build_quote(req)
    for w in q.warnings:
        st.warning(w)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Quoted price", f"RM{q.price_rm:,.2f}")
    k2.metric("Cost", f"RM{q.cost_rm:,.2f}", help=f"Margin {q.margin_pct:.0f}% on cost")
    k3.metric("Per container", f"RM{q.price_per_container_rm:,.2f}")
    k4.metric("CO2e", f"{q.co2e_kg:,.0f} kg", help=f"Diesel ≈ {q.diesel_litres:,.0f} L")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Cost breakdown")
        items = pd.DataFrame([i.model_dump() for i in q.line_items])
        by_cat = items.groupby("category", sort=False)["amount_rm"].sum().sort_values()
        fig = go.Figure(
            go.Bar(
                x=by_cat.values, y=by_cat.index, orientation="h", marker_color=SERIES_1,
                text=[f"RM{v:,.0f}" for v in by_cat.values], textposition="outside",
                hovertemplate="%{y}: RM%{x:,.2f}<extra></extra>",
            )
        )  # fmt: skip
        fig.update_traces(cliponaxis=False)
        fig.update_layout(bargap=0.35, hovermode="closest")
        _style(fig, "", height=280).update_xaxes(title="RM", showgrid=True, gridcolor=GRID)
        fig.update_layout(margin=dict(l=8, r=64, t=8, b=8))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Line items"):
            st.dataframe(
                items.rename(columns={"amount_rm": "amount (RM)"}),
                hide_index=True,
                use_container_width=True,
            )
    with right:
        st.subheader("Mode comparison")
        if len(modes) > 1:
            cmp = compare_modes(req)
            labels = [x.mode.value for x in cmp.quotes]
            st.plotly_chart(
                _bar(labels, [x.price_rm for x in cmp.quotes], "Price (RM)", "RM{:,.0f}"),
                use_container_width=True,
            )
            st.plotly_chart(
                _bar(labels, [x.co2e_kg for x in cmp.quotes], "CO2e (kg)", "{:,.0f} kg"),
                use_container_width=True,
            )
            if cmp.carbon_abatement_rm_per_tonne is not None:
                st.caption(
                    f"Choosing {cmp.lowest_co2_mode.value} costs "
                    f"RM{cmp.carbon_abatement_rm_per_tonne:,.0f} per tonne of CO2e avoided."
                )
            else:
                st.caption(f"**{cmp.cheapest_mode.value}** is both the cheapest and greenest.")
        else:
            st.info(f"Only {modes[0]} is offered to {rates.destinations[dest]['label']}.")
    st.caption(
        "Example rate card (data/reference/rates.json): diesel from the MOF weekly price, port "
        "handling from the Westports 2026 tariff, other rates are assumptions. Edit the file "
        "with real contract rates."
    )

# --------------------------------------------------------------------------- forecast
with tab_forecast:
    try:
        forecaster = get_forecaster()
    except ModelNotTrainedError as exc:
        st.error(str(exc))
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        commodity = c1.selectbox("Commodity", forecaster.commodities)
        road_dests = [k for k in dest_keys if "truck" in rates.destinations[k]["modes"]]
        spend_dest = c2.selectbox(
            "Typical delivery to", road_dests, format_func=lambda k: rates.destinations[k]["label"]
        )
        horizon = c3.slider("Forecast horizon (days)", 7, 90, 30)
        fc = forecaster.forecast(commodity, horizon)
        hist = forecaster.recent_history(commodity, 56)
        spend = estimate_haulage_spend(commodity, horizon, spend_dest, "20")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Forecast volume", f"{fc['forecast'].sum():,.0f} TEU")
        k2.metric("Avg / day", f"{fc['forecast'].mean():,.1f} TEU")
        k3.metric("Projected haulage revenue", f"RM{spend['projected_revenue_rm']:,.0f}")
        k4.metric("Projected CO2e", f"{spend['projected_co2e_kg'] / 1000:,.1f} t")

        fig = go.Figure()
        fig.add_scatter(
            x=hist["date"], y=hist["teu"], name="Actual TEU", mode="lines",
            line=dict(color=SERIES_1, width=2),
        )  # fmt: skip
        fc_x = [hist["date"].iloc[-1], *fc["date"]]
        fc_y = [hist["teu"].iloc[-1], *fc["forecast"]]
        fig.add_scatter(
            x=fc_x, y=fc_y, name="Forecast", mode="lines",
            line=dict(color=SERIES_2, width=2, dash="dash"),
        )  # fmt: skip
        st.plotly_chart(_style(fig, "TEU / day"), use_container_width=True)
        st.caption(
            f"Revenue projection: {spend['containers']} x 20ft to {spend['destination']} at "
            f"RM{spend['price_per_container_rm']:,.2f} each (current rate card). "
            f"Validation MAE {forecaster.metrics.get('mae', float('nan')):.1f} TEU/day."
        )
        with st.expander("Forecast table"):
            st.dataframe(fc, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------- documents
with tab_docs:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Upload")
        upload = st.file_uploader(
            "SOPs, manuals, policies (PDF, TXT, MD)", type=["pdf", "txt", "md"]
        )
        if upload is not None and st.button("Ingest document"):
            try:
                n = ingest_bytes(upload.getvalue(), upload.name)
                if copilot.cache is not None:
                    copilot.cache.clear()
                st.success(f"Ingested {upload.name}: {n} chunks")
            except Exception as exc:
                st.error(f"Could not ingest {upload.name}: {exc}")
        st.subheader("Indexed documents")
        sources = get_vector_store().sources()
        if sources:
            st.dataframe(
                pd.DataFrame(sources.items(), columns=["document", "chunks"]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No documents yet.")
    with right:
        st.subheader("Hybrid search (vector + keyword → rerank)")
        query = st.text_input("Search query", "How hot should CPO be before discharge?")
        if query:
            for r in get_retriever().search(query, k=4):
                ranks = f"vector #{r.vector_rank or '-'} · keyword #{r.keyword_rank or '-'}"
                st.markdown(
                    f"**{r.chunk.source}** p.{r.chunk.page} — score {r.score:.3f} ({ranks})"
                )
                st.caption(r.chunk.text)
