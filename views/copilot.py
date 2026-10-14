"""Chat with the copilot."""

from __future__ import annotations

import streamlit as st

from src.agent.copilot import get_copilot
from views.common import page_header, section

SUGGESTIONS = {
    ("Money at risk", ":material/payments:"): [
        "Which containers are about to incur demurrage?",
        "Where are we losing margin?",
    ],
    ("Commercial", ":material/request_quote:"): [
        "Quote 2 x 40ft palm oil containers from Westport to Pasir Gudang",
        "Which carrier is best for chemicals to Penang? Cheapest please",
    ],
    ("Operations", ":material/schedule:"): [
        "Will a chemicals shipment to Kuching on 2026-12-10 be delayed?",
        "Forecast chemicals volume for the next 2 weeks",
    ],
    ("Partners & compliance", ":material/verified_user:"): [
        "Show the haulier scorecard",
        "What documents do I need to clear a chemical import?",
    ],
}

copilot = get_copilot()
history: list[dict] = st.session_state.setdefault("messages", [])

head, action = st.columns([5, 1], vertical_alignment="bottom")
with head:
    page_header(
        "Copilot",
        "Ask about demurrage, margins, quotes, carriers, delays, customs or compliance. Figures "
        "come from tools (rate card, job log, models, operations manual), never from memory.",
    )
with action:
    if history and st.button("Clear chat", icon=":material/delete:", width="stretch"):
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
    section("Start with a question", "answers come from the tools, with the working shown")
    cols = st.columns(2, gap="medium")
    for i, ((title, icon), prompts) in enumerate(SUGGESTIONS.items()):
        with cols[i % 2], st.container(border=True):
            st.markdown(f"{icon} **{title}**")
            for j, text in enumerate(prompts):
                if st.button(
                    text,
                    key=f"s{i}{j}",
                    width="stretch",
                    type="tertiary",
                    icon=":material/north_east:",
                ):
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
