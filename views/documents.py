"""Search and manage the documents the copilot answers from."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from src.agent.copilot import get_copilot
from src.rag.ingest_pdf import ingest_bytes
from src.rag.retriever import get_retriever
from src.rag.vector_store import get_vector_store
from views.common import page_header

EXAMPLES = ["CPO heating temperature", "K8 transshipment", "demurrage free time", "DG segregation"]

page_header(
    "Documents",
    "Hybrid search over SOPs and manuals: vector + keyword retrieval, fused and reranked. "
    "The copilot cites these passages in its answers.",
)

store = get_vector_store()
search, library = st.columns([2.2, 1], gap="large")


def highlight(text: str, query: str) -> str:
    terms = sorted(
        {t for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 2}, key=len, reverse=True
    )
    text = " ".join(text.split())
    if not terms:
        return text
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)


with search:
    picked = st.pills("Examples", EXAMPLES, label_visibility="collapsed")
    query = st.text_input(
        "Search",
        value=picked or "",
        placeholder="e.g. what documents are needed for a K1 declaration?",
    )
    o1, o2 = st.columns([1, 1])
    k = o1.slider("Results", 1, 8, 4)
    rerank = o2.toggle("Rerank", value=True, help="Second-stage reranking of the fused candidates")
    if query:
        results = get_retriever().search(query, k=k, rerank=rerank)
        if not results:
            st.info("No matching passages.", icon=":material/search_off:")
        for r in results:
            with st.container(border=True):
                page = f" · page {r.chunk.page}" if r.chunk.page else ""
                ranks = f"vector #{r.vector_rank or '–'} · keyword #{r.keyword_rank or '–'}"
                st.markdown(f"**{r.chunk.source}**{page}")
                st.caption(f"score {r.score:.3f} · {ranks}")
                st.markdown(highlight(r.chunk.text, query))

with library:
    st.markdown("**Library**")
    sources = store.sources()
    if sources:
        st.dataframe(
            pd.DataFrame(sources.items(), columns=["Document", "Chunks"]),
            hide_index=True, width="stretch",
        )  # fmt: skip
    else:
        st.info("No documents indexed yet.")
    with st.popover("Add document", icon=":material/upload_file:", width="stretch"):
        upload = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"])
        if upload is not None and st.button("Index document", type="primary"):
            try:
                n = ingest_bytes(upload.getvalue(), upload.name)
                cache = get_copilot().cache
                if cache is not None:
                    cache.clear()
                st.success(f"Indexed {upload.name} ({n} chunks).")
            except Exception as exc:
                st.error(f"Could not index {upload.name}: {exc}")
    st.caption("Re-uploading a file with the same name replaces it.")
