"""Streamlit dashboard - Freight Ops Copilot for a Port Klang 3PL.

Runs fully in-process (no API server needed), so it deploys as-is to Streamlit
Community Cloud or Hugging Face Spaces.

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.agent.copilot import get_copilot
from src.api.main import bootstrap
from src.auth.users import ROLE_LABELS, Role, User
from src.config import PROJECT_ROOT, get_settings
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

# ----------------------------------------------------------------------------- login
if get_settings().auth_required and "user" not in st.session_state:
    st.navigation(
        [st.Page("views/login.py", title="Sign in", icon=":material/login:")], position="hidden"
    ).run()
    st.stop()

_u = st.session_state.get("user")
user = User(_u["username"], Role(_u["role"]), _u["display_name"], _u["demo"]) if _u else None

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
    if user is not None:
        st.divider()
        who, out = st.columns([3, 1.3], vertical_alignment="center")
        who.markdown(f"**{user.display_name}**  \n{ROLE_LABELS[user.role]}")
        if out.button("Sign out", key="sign_out"):
            st.session_state.pop("user", None)
            st.rerun()
        if user.demo:
            st.caption("Demo account: your uploads stay in this session only.")

# (section, file stem, title, icon). Pages whose file is missing are skipped.
PAGES = [
    ("", "control_tower", "Control tower", ":material/radar:"),
    ("", "copilot", "Copilot", ":material/forum:"),
    ("Commercial", "quote", "Quote", ":material/request_quote:"),
    ("Commercial", "planner", "Route planner", ":material/alt_route:"),
    ("Operations", "delays", "Delay risk", ":material/schedule:"),
    ("Operations", "forecast", "Forecast", ":material/insights:"),
    ("Operations", "intake", "Document intake", ":material/document_scanner:"),
    ("Operations", "data_import", "Data import", ":material/upload_file:"),
    ("Operations", "alerts", "Alerts", ":material/notifications:"),
    ("Partners", "hauliers", "Hauliers", ":material/local_shipping:"),
    ("Knowledge", "documents", "Documents", ":material/description:"),
    ("Knowledge", "evals", "AI quality", ":material/fact_check:"),
]

sections: dict[str, list] = {}
for section, stem, title, icon in PAGES:
    if not (PROJECT_ROOT / "views" / f"{stem}.py").exists():
        continue
    if user is not None and not user.can_open(stem):
        continue
    sections.setdefault(section, []).append(
        st.Page(f"views/{stem}.py", title=title, icon=icon, default=stem == "control_tower")
    )
st.navigation(sections, position="top").run()
