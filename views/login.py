"""Sign-in page, with the public demo accounts listed for visitors."""

from __future__ import annotations

import streamlit as st

from src.auth.users import DEMO_ACCOUNTS, DEMO_PASSWORD, ROLE_LABELS, LoginError, authenticate
from src.config import get_settings
from views.common import COMPANY

ROLE_NOTES = {
    "manager": "Everything: control tower, commercial, operations, partners, AI quality",
    "ops": "Control tower, delays, forecast, document intake, data import, alerts",
    "sales": "Control tower, quotes, route planner, forecast, hauliers",
}


def _fill(username: str) -> None:
    st.session_state["login_username"] = username
    st.session_state["login_password"] = DEMO_PASSWORD


_, mid, _ = st.columns([1, 1.25, 1])
with mid:
    st.html(
        '<div class="brand" style="margin-top:1.5rem"><div class="mark">M</div><div>'
        '<div class="name">Muara Logistics</div>'
        '<div class="tag">Port Klang · Freight Ops Copilot</div></div></div>'
    )
    st.markdown("## Sign in")
    st.caption(
        f"AI logistics operations platform for {COMPANY}, a fictional Port Klang 3PL "
        "(portfolio case study)."
    )

    with st.form("login", border=True):
        username = st.text_input("Username", key="login_username", autocomplete="username")
        password = st.text_input(
            "Password", type="password", key="login_password", autocomplete="current-password"
        )
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
    if submitted:
        try:
            user = authenticate(username, password)
        except LoginError as exc:
            st.error(str(exc), icon=":material/lock:")
        else:
            st.session_state["user"] = {
                "username": user.username,
                "role": user.role.value,
                "display_name": user.display_name,
                "demo": user.demo,
            }
            for key in ("login_username", "login_password"):
                st.session_state.pop(key, None)
            st.rerun()

    if get_settings().demo_accounts:
        st.markdown("**Demo accounts** · open to anyone, pick a role to look around")
        for name, role, person in DEMO_ACCOUNTS:
            with st.container(border=True):
                info, action = st.columns([3, 1], vertical_alignment="center")
                info.markdown(
                    f"**{ROLE_LABELS[role]}** · {person}  \n"
                    f"Username `{name}` · Password `{DEMO_PASSWORD}`"
                )
                info.caption(ROLE_NOTES[name])
                action.button(
                    "Use", key=f"use_{name}", on_click=_fill, args=(name,), width="stretch"
                )
        st.caption(
            "Demo accounts can try every feature, but uploads stay in your own session and "
            "Telegram alerts are disabled, so the shared data stays clean."
        )
