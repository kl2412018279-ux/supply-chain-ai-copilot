"""Chat with the copilot."""

from __future__ import annotations

import streamlit as st

from src.agent.copilot import get_copilot
from views.common import page_header

SUGGESTIONS = [
    "Quote 2 x 40ft palm oil containers from Westport to Pasir Gudang",
    "Compare truck vs rail for a 40ft chemicals container to Penang",
    "Forecast chemicals volume for the next 2 weeks",
    "What documents do I need to clear a chemical import?",
]

copilot = get_copilot()
history: list[dict] = st.session_state.setdefault("messages", [])

head, action = st.columns([5, 1], vertical_alignment="bottom")
with head:
    page_header(
        "Copilot",
        "Ask about quotes, haulage, container volumes, customs or ESG. Figures come from tools "
        "(rate card, forecast model, operations manual), never from the model's memory.",
    )
with action:
    if history and st.button("Clear chat", icon=":material/delete:", use_container_width=True):
        history.clear()
        st.rerun()


def render_details(msg: dict) -> None:
    calls, sources = msg.get("tool_calls", []), msg.get("sources", [])
    if not calls and not sources:
        return
    with st.expander("How this was answered"):
        for call in calls:
            st.markdown(f"**Tool** `{call['name']}`")
            st.json(call["arguments"], expanded=False)
        for s in sources:
            page = f" · p.{s['page']}" if s.get("page") else ""
            st.markdown(f"**Source** {s['source']}{page}")
            st.caption(s["snippet"][:400])


def render(msg: dict) -> None:
    with st.chat_message(
        msg["role"],
        avatar=":material/person:" if msg["role"] == "user" else ":material/local_shipping:",
    ):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])
        render_details(msg)


prompt = st.session_state.pop("pending_prompt", None)

if not history and prompt is None:
    st.markdown("##### Try one of these")
    cols = st.columns(2)
    for i, text in enumerate(SUGGESTIONS):
        if cols[i % 2].button(text, key=f"s{i}", use_container_width=True):
            st.session_state["pending_prompt"] = text
            st.rerun()

for msg in history:
    render(msg)

prompt = st.chat_input("Ask the copilot…") or prompt
if prompt:
    user_msg = {"role": "user", "content": prompt}
    history.append(user_msg)
    render(user_msg)
    with st.spinner("Working…"):
        res = copilot.ask(prompt)
    parts = [res.agent, f"{res.latency_ms:,.0f} ms"]
    if res.cached:
        parts.append("served from cache")
    if res.blocked:
        parts.append(f"blocked by guardrail ({res.guardrail.value})")
    reply = {
        "role": "assistant",
        "content": res.answer,
        "meta": " · ".join(parts),
        "tool_calls": [c.model_dump(mode="json") for c in res.tool_calls],
        "sources": [s.model_dump() for s in res.sources],
    }
    history.append(reply)
    render(reply)
