"""Streamlit dashboard - Supply Chain & ESG AI Copilot.

Runs fully in-process (no API server needed), so it deploys as-is to Streamlit
Community Cloud or Hugging Face Spaces.

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent.copilot import get_copilot
from src.api.main import bootstrap
from src.config import get_settings
from src.forecasting.esg_calculator import EMISSION_FACTORS_KG_PER_TKM, calculate_emissions
from src.forecasting.predict import ModelNotTrainedError, get_forecaster
from src.rag.ingest_pdf import ingest_bytes
from src.rag.retriever import get_retriever
from src.rag.vector_store import get_vector_store

# Reference categorical palette, slots 1-2 (validated light/dark, CVD-safe).
SERIES_1 = "#2a78d6"  # history
SERIES_2 = "#eb6834"  # forecast
GRID = "rgba(128,128,128,0.18)"

st.set_page_config(page_title="Supply Chain AI Copilot", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Preparing model and documents…")
def _startup() -> bool:
    bootstrap()
    return True


def _style(fig: go.Figure, y_title: str) -> go.Figure:
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(title=y_title, gridcolor=GRID, zeroline=False),
        xaxis=dict(showgrid=False),
    )
    return fig


_startup()
settings = get_settings()
copilot = get_copilot()

# ---------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📦 Copilot status")
    st.markdown(f"**Agent:** `{copilot.agent.name}`")
    if copilot.agent.name == "rule-based":
        st.caption("Offline mode. Set `GROQ_API_KEY` (free) to enable the LangGraph LLM agent.")
    st.markdown(f"**Vector store:** `{type(get_vector_store()).__name__}`")
    if copilot.cache is not None:
        stats = copilot.cache.stats()
        st.markdown(f"**Semantic cache:** {stats['hits']} hits / {stats['misses']} misses")
    try:
        m = get_forecaster().metrics
        st.markdown(f"**Forecast model:** MAE {m['mae']:.1f} · R² {m['r2']:.2f}")
    except ModelNotTrainedError:
        st.warning("Forecast model not trained.")

st.title("📦 Enterprise Supply Chain & ESG AI Copilot")
tab_chat, tab_forecast, tab_esg, tab_docs = st.tabs(
    ["💬 AI Assistant", "📈 Demand Forecast", "🌱 Shipping Emissions", "📄 ESG Documents"]
)

# ------------------------------------------------------------------------------- chat
with tab_chat:
    st.caption(
        "Try: *Forecast demand for SKU-101 for the next 2 weeks* · "
        "*Emissions for 500 kg shipped 300 km by truck* · "
        "*Can SKU-104 battery packs be shipped by air?* · *Show sales history for SKU-103*"
    )
    history = st.session_state.setdefault("messages", [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    if prompt := st.chat_input("Ask about demand, emissions, sales or ESG policy…"):
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

# --------------------------------------------------------------------------- forecast
with tab_forecast:
    try:
        forecaster = get_forecaster()
    except ModelNotTrainedError as exc:
        st.error(str(exc))
    else:
        c1, c2 = st.columns([1, 2])
        sku = c1.selectbox("SKU", forecaster.skus)
        horizon = c2.slider("Forecast horizon (days)", 7, 90, 28)
        fc = forecaster.forecast(sku, horizon)
        hist = forecaster.recent_history(sku, 56)

        k1, k2, k3 = st.columns(3)
        k1.metric("Forecast total", f"{fc['forecast'].sum():,.0f} units")
        k2.metric("Avg / day", f"{fc['forecast'].mean():,.1f}")
        k3.metric("Validation MAE", f"{forecaster.metrics.get('mae', float('nan')):.1f} units/day")

        fig = go.Figure()
        fig.add_scatter(
            x=hist["date"], y=hist["sales"], name="Actual sales", mode="lines",
            line=dict(color=SERIES_1, width=2),
        )  # fmt: skip
        # Connect the forecast to the last actual point so the line is continuous.
        fc_x = [hist["date"].iloc[-1], *fc["date"]]
        fc_y = [hist["sales"].iloc[-1], *fc["forecast"]]
        fig.add_scatter(
            x=fc_x, y=fc_y, name="Forecast", mode="lines",
            line=dict(color=SERIES_2, width=2, dash="dash"),
        )  # fmt: skip
        st.plotly_chart(_style(fig, "Units / day"), use_container_width=True)
        with st.expander("Forecast table"):
            st.dataframe(fc, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------- ESG
with tab_esg:
    c1, c2, c3 = st.columns(3)
    dist = c1.number_input("Distance (km)", min_value=1.0, value=250.0, step=10.0)
    weight = c2.number_input("Weight (kg)", min_value=1.0, value=500.0, step=50.0)
    mode = c3.selectbox("Transport mode", list(EMISSION_FACTORS_KG_PER_TKM))
    result = calculate_emissions(dist, weight, mode)

    k1, k2 = st.columns(2)
    k1.metric("Estimated footprint", f"{result.co2e_kg:,.2f} kg CO2e")
    best = result.lowest_emission_mode.value
    saving = result.co2e_kg - result.alternatives_co2e_kg[best]
    k2.metric(f"Saving by switching to {best}", f"{saving:,.2f} kg CO2e")

    modes = list(result.alternatives_co2e_kg)
    values = list(result.alternatives_co2e_kg.values())
    fig = go.Figure(
        go.Bar(
            x=modes, y=values, marker_color=SERIES_1,
            marker_line=dict(width=0),
            text=[f"{v:,.1f}" for v in values], textposition="outside",
            hovertemplate="%{x}: %{y:,.2f} kg CO2e<extra></extra>",
        )
    )  # fmt: skip
    fig.update_layout(bargap=0.45)
    st.plotly_chart(_style(fig, "kg CO2e"), use_container_width=True)
    st.caption(
        "Factors (kg CO2e per tonne-km): "
        + ", ".join(f"{m} {f}" for m, f in EMISSION_FACTORS_KG_PER_TKM.items())
        + ". Typical GLEC / DEFRA averages - override for audited reporting."
    )

# ------------------------------------------------------------------------- documents
with tab_docs:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Upload")
        upload = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"])
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
        query = st.text_input("Search query", "Can lithium battery packs be shipped by air?")
        if query:
            for r in get_retriever().search(query, k=4):
                ranks = f"vector #{r.vector_rank or '-'} · keyword #{r.keyword_rank or '-'}"
                st.markdown(
                    f"**{r.chunk.source}** p.{r.chunk.page} — score {r.score:.3f} ({ranks})"
                )
                st.caption(r.chunk.text)
