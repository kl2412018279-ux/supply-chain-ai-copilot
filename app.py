"""Streamlit dashboard - Freight Ops Copilot for a Port Klang 3PL.

Runs fully in-process (no API server needed), so it deploys as-is to Streamlit
Community Cloud or Hugging Face Spaces.

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.agent.copilot import get_copilot
from src.api.main import bootstrap
from src.costing.rates import get_rate_card
from src.forecasting.predict import ModelNotTrainedError, get_forecaster
from src.rag.vector_store import get_vector_store
from views.common import COMPANY

st.set_page_config(
    page_title="Freight Ops Copilot", page_icon=":material/local_shipping:", layout="wide"
)


@st.cache_resource(show_spinner="Preparing model and documents…")
def _startup() -> bool:
    bootstrap()
    return True


_startup()

with st.sidebar:
    st.markdown("**Freight Ops Copilot**")
    st.caption(
        f"Built for {COMPANY}, a fictional Port Klang 3PL handling customs clearance, haulage, "
        "warehousing and delivery of chemicals, palm oil and plastics. Rates and volumes are "
        "example data."
    )
    st.divider()
    copilot = get_copilot()
    rates = get_rate_card()
    rows = [("Agent", copilot.agent.name), ("Vector store", type(get_vector_store()).__name__)]
    try:
        rows.append(("Forecast MAE", f"{get_forecaster().metrics['mae']:.1f} TEU/day"))
    except ModelNotTrainedError:
        rows.append(("Forecast model", "not trained"))
    diesel = rates.value("fuel", "diesel_rm_per_litre_peninsular")
    rows.append(("Diesel", f"RM{diesel:.2f}/L"))
    if copilot.cache is not None:
        stats = copilot.cache.stats()
        rows.append(("Cache hits", f"{stats['hits']} of {stats['hits'] + stats['misses']} answers"))
    st.markdown(
        "\n".join(f"<small>{k}</small><br>**{v}**<br>" for k, v in rows), unsafe_allow_html=True
    )
    if copilot.agent.name == "rule-based":
        st.caption("Offline mode: set `GROQ_API_KEY` to enable the LLM agent.")
    st.divider()
    st.caption("[Source on GitHub](https://github.com/kl2412018279-ux/supply-chain-ai-copilot)")

page = st.navigation(
    [
        st.Page("views/copilot.py", title="Copilot", icon=":material/forum:", default=True),
        st.Page("views/quote.py", title="Quote", icon=":material/request_quote:"),
        st.Page("views/forecast.py", title="Forecast", icon=":material/insights:"),
        st.Page("views/documents.py", title="Documents", icon=":material/description:"),
    ],
    position="top",
)
page.run()
