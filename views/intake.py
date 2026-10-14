"""Document intake: extract, review and save Bill of Lading / invoice fields."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agent.workflow import build_llm
from src.config import PROJECT_ROOT, get_settings
from src.costing.quote import QuoteRequest, build_quote
from src.extraction.extract import ShippingDocument, extract, validate
from src.extraction.iso6346 import is_valid
from src.extraction.store import get_document_store
from views.common import page_header, rm

SAMPLES = PROJECT_ROOT / "data" / "samples"
EDITABLE = [
    "document_number", "issue_date", "shipper", "consignee", "vessel_voyage", "port_of_loading",
    "port_of_discharge", "container_size", "container_count", "packages", "commodity_description",
    "hs_code", "gross_weight_kg", "net_weight_kg", "incoterm", "currency", "total_amount",
    "country_of_origin",
]  # fmt: skip

page_header(
    "Document intake",
    "Upload a Bill of Lading or commercial invoice. Fields are extracted (LLM when configured, "
    "rule-based offline), validated, and saved to the database after review.",
)

src_col, file_col = st.columns([1, 2], vertical_alignment="bottom")
source = src_col.segmented_control("Source", ["Sample", "Upload"], default="Sample") or "Sample"
if source == "Sample":
    samples = sorted(p.name for p in SAMPLES.glob("*.pdf"))
    name = file_col.selectbox("Sample document (fictional)", samples)
    data = (SAMPLES / name).read_bytes() if name else None
else:
    upload = file_col.file_uploader("PDF", type=["pdf"])
    name, data = (upload.name, upload.getvalue()) if upload else (None, None)

if not data:
    st.info("Choose a sample or upload a PDF.", icon=":material/upload_file:")
    st.stop()


@st.cache_data(show_spinner="Extracting…")
def run_extraction(payload: bytes, file_name: str):
    return extract(payload, file_name, build_llm(get_settings()))


try:
    result = run_extraction(data, name)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

doc = result.document
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown(
        f"**{doc.doc_type.value.replace('_', ' ').title()}** · extracted by "
        f"{'LLM' if result.method == 'llm' else 'rule-based parser'} · "
        f"{result.text_chars:,} characters"
    )
    labels = {
        "hs_code": "HS code",
        "gross_weight_kg": "Gross weight (kg)",
        "net_weight_kg": "Net weight (kg)",
        "vessel_voyage": "Vessel / voyage",
    }
    rows = pd.DataFrame(
        [
            {
                "Field": labels.get(f, f.replace("_", " ").capitalize()),
                "key": f,
                "Value": getattr(doc, f),
            }
            for f in EDITABLE
        ]
    )
    rows["Value"] = rows["Value"].map(lambda v: "" if v is None else str(v))
    edited = st.data_editor(
        rows[["Field", "Value"]], hide_index=True, width="stretch", disabled=["Field"],
        key=f"edit-{name}", height=36 * (len(rows) + 1) + 4,
    )  # fmt: skip

    values = {}
    for key, raw in zip(rows["key"], edited["Value"], strict=True):
        raw = (raw or "").strip()
        values[key] = raw or None
    for numeric in ("gross_weight_kg", "net_weight_kg", "total_amount"):
        try:
            values[numeric] = (
                float(str(values[numeric]).replace(",", "")) if values[numeric] else None
            )
        except ValueError:
            st.error(f"{numeric.replace('_', ' ')} must be a number.")
            st.stop()
    values["container_count"] = (
        int(float(values["container_count"])) if values["container_count"] else None
    )
    reviewed = ShippingDocument(
        **values, doc_type=doc.doc_type, container_numbers=doc.container_numbers,
        dangerous_goods=doc.dangerous_goods,
    )  # fmt: skip
    issues, missing = validate(reviewed)

with right:
    st.markdown("**Validation**")
    if not issues and not missing:
        st.success("All checks passed.", icon=":material/check_circle:")
    for i in issues:
        (st.error if i.severity == "error" else st.warning)(i.message)
    if missing:
        st.warning("Missing: " + ", ".join(m.replace("_", " ") for m in missing))

    st.markdown("**Containers**")
    for c in reviewed.container_numbers:
        ok = is_valid(c)
        st.markdown(
            f"{':material/check_circle:' if ok else ':material/cancel:'} `{c}` "
            + ("" if ok else "· check digit fails")
        )
    if reviewed.dangerous_goods:
        st.caption("Dangerous goods detected: DG surcharge and documentation apply.")

    suggestion = result.suggested_quote
    if suggestion:
        st.markdown("**Haulage quote for this shipment**")
        q = build_quote(QuoteRequest(**suggestion))
        st.metric(
            f"{q.containers} × {q.container_size}ft {q.cargo_type} → {q.destination_label}",
            rm(q.price_rm, 0),
            help=f"From {q.origin_label}; cost {rm(q.cost_rm)}, CO2e {q.co2e_kg:,.0f} kg.",
        )
        for w in q.warnings:
            st.caption(w)

    blocked = any(i.severity == "error" for i in issues) or not reviewed.document_number
    if st.button(
        "Save to database",
        type="primary",
        icon=":material/save:",
        disabled=blocked,
        width="stretch",
    ):
        get_document_store().save(reviewed, name, [i.model_dump() for i in issues])
        st.toast(f"Saved {reviewed.document_number}")
    if blocked:
        st.caption("Fix errors (e.g. container numbers) before saving.")

st.divider()
st.markdown("**Saved documents**")
records = get_document_store().list()
if records:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Saved": r["saved_at"], "Type": r["doc_type"].replace("_", " "),
                    "Number": r["document_number"], "Shipper": r["data"].get("shipper"),
                    "Consignee": r["data"].get("consignee"),
                    "Containers": len(r["data"].get("container_numbers") or []),
                    "Issues": len(r["issues"]), "File": r["file_name"],
                }
                for r in records
            ]
        ),
        hide_index=True, width="stretch",
    )  # fmt: skip
else:
    st.caption("Nothing saved yet.")
