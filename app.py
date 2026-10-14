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
from views.common import COMPANY, inject_css

st.set_page_config(
    page_title="Freight Ops Copilot", page_icon=":material/local_shipping:", layout="wide"
)


@st.cache_resource(show_spinner="Preparing model and documents…")
def _startup() -> bool:
    bootstrap()
    return True


_startup()

inject_css()

with st.sidebar:
    st.html(
        '<div class="brand"><div class="mark">M</div><div><div class="name">Muara Logistics</div>'
        '<div class="tag">Port Klang · Freight Ops Copilot</div></div></div>'
    )
    rates = get_rate_card()
    diesel = rates.value("fuel", "diesel_rm_per_litre_peninsular")
    dem = rates.data["demurrage"]["rm_per_day"]
    rows = [
        ("Diesel (APM)", f"RM{diesel:.2f}/L"),
        ("Margin target", f"{rates.data['margin_pct']:.0f}%"),
        ("Demurrage 40ft", f"RM{dem['40']:,.0f}/day"),
    ]
    st.html("".join(f'<div class="kv"><span>{k}</span><span>{v}</span></div>' for k, v in rows))
    st.caption(
        f"{COMPANY} is a fictional 3PL handling customs clearance, haulage, warehousing and "
        "delivery of chemicals, palm oil and plastics. Rates, jobs and containers are example data."
    )
    with st.expander("System", icon=":material/memory:"):
        copilot = get_copilot()
        info = [("Agent", copilot.agent.name), ("Vector store", type(get_vector_store()).__name__)]
        try:
            info.append(("Forecast MAE", f"{get_forecaster().metrics['mae']:.1f} TEU/day"))
        except ModelNotTrainedError:
            info.append(("Forecast model", "not trained"))
        if copilot.cache is not None:
            stats = copilot.cache.stats()
            info.append(("Cache hits", f"{stats['hits']} / {stats['hits'] + stats['misses']}"))
        st.html("".join(f'<div class="kv"><span>{k}</span><span>{v}</span></div>' for k, v in info))
        if copilot.agent.name == "rule-based":
            st.caption("Offline mode: set `GROQ_API_KEY` to enable the LLM agent.")
    st.caption("[Source on GitHub](https://github.com/kl2412018279-ux/supply-chain-ai-copilot)")

page = st.navigation(
    {
        "": [
            st.Page(
                "views/control_tower.py",
                title="Control tower",
                icon=":material/radar:",
                default=True,
            ),
            st.Page("views/copilot.py", title="Copilot", icon=":material/forum:"),
        ],
        "Commercial": [
            st.Page("views/quote.py", title="Quote", icon=":material/request_quote:"),
            st.Page("views/planner.py", title="Route planner", icon=":material/alt_route:"),
        ],
        "Operations": [
            st.Page("views/delays.py", title="Delay risk", icon=":material/schedule:"),
            st.Page("views/forecast.py", title="Forecast", icon=":material/insights:"),
            st.Page("views/intake.py", title="Document intake", icon=":material/document_scanner:"),
        ],
        "Partners": [
            st.Page("views/hauliers.py", title="Hauliers", icon=":material/local_shipping:")
        ],
        "Knowledge": [
            st.Page("views/documents.py", title="Documents", icon=":material/description:")
        ],
    },
    position="top",
)
page.run()
