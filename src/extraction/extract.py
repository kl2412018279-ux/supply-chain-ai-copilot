"""Extract structured fields from Bills of Lading and commercial invoices.

Pipeline::

    PDF -> text (pypdf) -> fields (LLM structured output, or rule-based parser offline)
        -> ShippingDocument (Pydantic) -> business validation -> database

Validation goes beyond types: ISO 6346 container check digits, HS-code format,
gross weight per container against the 25 t Port Klang limit, missing mandatory
fields per document type. The result also maps the document onto the quote engine
(cargo type from the HS chapter, destination from the consignee address).

Scanned images have no text layer; those need OCR or a vision model first.
"""

from __future__ import annotations

import io
import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.extraction.iso6346 import find_containers, is_valid

logger = logging.getLogger(__name__)


class DocType(str, Enum):
    bill_of_lading = "bill_of_lading"
    commercial_invoice = "commercial_invoice"
    unknown = "unknown"


class ShippingDocument(BaseModel):
    """Fields a 3PL needs to clear and move a shipment."""

    doc_type: DocType = DocType.unknown
    document_number: str | None = Field(default=None, description="B/L number or invoice number")
    issue_date: str | None = Field(default=None, description="YYYY-MM-DD")
    shipper: str | None = Field(default=None, description="Shipper / seller / exporter")
    consignee: str | None = Field(default=None, description="Consignee / buyer / importer")
    vessel_voyage: str | None = None
    port_of_loading: str | None = None
    port_of_discharge: str | None = None
    container_numbers: list[str] = Field(default_factory=list)
    container_count: int | None = None
    container_size: str | None = Field(default=None, description="'20' or '40'")
    packages: str | None = None
    commodity_description: str | None = None
    hs_code: str | None = None
    gross_weight_kg: float | None = None
    net_weight_kg: float | None = None
    incoterm: str | None = None
    currency: str | None = None
    total_amount: float | None = None
    country_of_origin: str | None = None
    dangerous_goods: bool = False

    @field_validator("container_numbers")
    @classmethod
    def _normalise_containers(cls, v: list[str]) -> list[str]:
        return [re.sub(r"[\s-]", "", c.upper()) for c in v]


class ValidationIssue(BaseModel):
    severity: str = Field(description="error or warning")
    field: str
    message: str


class ExtractionResult(BaseModel):
    file_name: str
    method: str = Field(description="llm or rules")
    document: ShippingDocument
    issues: list[ValidationIssue]
    missing_fields: list[str]
    suggested_quote: dict[str, Any] | None = None
    text_chars: int


# ------------------------------------------------------------------------------ text


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader  # noqa: PLC0415

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# --------------------------------------------------------------------- rule parser

LABELS: dict[str, list[str]] = {
    "bl_number": [r"B/?L\s*(?:No|Number)", r"Bill of Lading\s*(?:No|Number)"],
    "invoice_number": [r"Invoice\s*(?:No|Number)"],
    "issue_date": [r"Date of Issue", r"Invoice Date", r"Issue Date", r"Date"],
    "shipper": [r"Shipper", r"Seller", r"Exporter"],
    "consignee": [r"Consignee", r"Buyer", r"Importer"],
    "vessel_voyage": [r"Vessel\s*/\s*Voyage", r"Vessel"],
    "port_of_loading": [r"Port of Loading", r"POL"],
    "port_of_discharge": [r"Port of Discharge", r"POD"],
    "container_numbers": [r"Container\s*(?:No|Number)s?\.?", r"Containers?"],
    "container_size": [r"Container Size", r"Equipment"],
    "packages": [r"Packages", r"No\. of Packages"],
    "commodity_description": [r"Description of Goods", r"Description"],
    "hs_code": [r"HS\s*Code", r"H\.S\. Code", r"Tariff Code"],
    "gross_weight": [r"Gross Weight", r"G\.W\."],
    "net_weight": [r"Net Weight", r"N\.W\."],
    "incoterm": [r"Incoterms?", r"Terms of Delivery"],
    "total_amount": [r"Total Amount", r"Invoice Total", r"Total Value"],
    "country_of_origin": [r"Country of Origin"],
}
INCOTERMS = r"\b(EXW|FCA|FAS|FOB|CFR|CIF|CPT|CIP|DAP|DPU|DDP)\b"


def _field(text: str, key: str) -> str | None:
    for label in LABELS[key]:
        m = re.search(rf"^\s*{label}\s*[:.]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def _weight_kg(raw: str | None) -> float | None:
    if not raw:
        return None
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*(KGS?|KILOGRAMS?|MT|TONNES?|TONS?)", raw, re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return value * 1000 if m.group(2).upper().startswith(("MT", "TON")) else value


def parse_rules(text: str) -> ShippingDocument:
    upper = text.upper()
    doc_type = (
        DocType.bill_of_lading
        if "BILL OF LADING" in upper
        else DocType.commercial_invoice
        if "INVOICE" in upper
        else DocType.unknown
    )
    size_raw = _field(text, "container_size") or ""
    size = re.search(r"\b(20|40|45)\s*['’]|\b(20|40|45)\s*(?:FT|FOOT|GP|HC|DV)", size_raw.upper())
    count = re.search(r"\b(\d+)\s*[xX×]\s*(?:20|40|45)", size_raw)
    # Read containers from the container line; B/L numbers often share the same shape.
    containers = find_containers(_field(text, "container_numbers") or "")
    doc_number = _field(text, "bl_number") or _field(text, "invoice_number")
    if not containers:
        containers = [c for c in find_containers(text) if c != re.sub(r"\W", "", doc_number or "")]
    amount_raw = _field(text, "total_amount") or ""
    amount = re.search(r"([A-Z]{3})\s*([\d,]+(?:\.\d+)?)", amount_raw)
    incoterm = re.search(INCOTERMS, _field(text, "incoterm") or "", re.IGNORECASE)
    hs = re.search(r"\b\d{4}(?:\.?\d{2}){0,3}\b", _field(text, "hs_code") or "")
    description = _field(text, "commodity_description")
    return ShippingDocument(
        doc_type=doc_type,
        document_number=_field(
            text, "bl_number" if doc_type is DocType.bill_of_lading else "invoice_number"
        )
        or _field(text, "bl_number")
        or _field(text, "invoice_number"),
        issue_date=_field(text, "issue_date"),
        shipper=_field(text, "shipper"),
        consignee=_field(text, "consignee"),
        vessel_voyage=_field(text, "vessel_voyage"),
        port_of_loading=_field(text, "port_of_loading"),
        port_of_discharge=_field(text, "port_of_discharge"),
        container_numbers=containers,
        container_count=int(count.group(1)) if count else (len(containers) or None),
        container_size=(size.group(1) or size.group(2)) if size else None,
        packages=_field(text, "packages"),
        commodity_description=description,
        hs_code=hs.group(0) if hs else None,
        gross_weight_kg=_weight_kg(_field(text, "gross_weight")),
        net_weight_kg=_weight_kg(_field(text, "net_weight")),
        incoterm=incoterm.group(1).upper() if incoterm else None,
        currency=amount.group(1) if amount else None,
        total_amount=float(amount.group(2).replace(",", "")) if amount else None,
        country_of_origin=_field(text, "country_of_origin"),
        dangerous_goods=bool(re.search(r"\bUN\s?\d{4}\b|DANGEROUS GOODS|\bIMDG\b", upper)),
    )


# ---------------------------------------------------------------------- LLM parser

LLM_INSTRUCTIONS = """Extract the shipping document fields from the text below.
Rules: copy values exactly as written; use null when a field is absent (never guess);
weights in kilograms; container numbers as 11 characters without spaces; dates as
YYYY-MM-DD; container_size is "20" or "40".

{text}"""


def parse_llm(text: str, llm) -> ShippingDocument:
    structured = llm.with_structured_output(ShippingDocument)
    doc = structured.invoke(LLM_INSTRUCTIONS.format(text=text[:12000]))
    if isinstance(doc, dict):
        doc = ShippingDocument(**doc)
    # Never trust the model alone for identifiers: keep only container numbers that
    # really appear in the source text.
    in_text = set(find_containers(text))
    doc.container_numbers = [c for c in doc.container_numbers if c in in_text] or sorted(in_text)
    return doc


# ---------------------------------------------------------------------- validation

REQUIRED = {
    DocType.bill_of_lading: [
        "document_number", "shipper", "consignee", "port_of_loading", "port_of_discharge",
        "container_numbers", "commodity_description", "gross_weight_kg",
    ],
    DocType.commercial_invoice: [
        "document_number", "shipper", "consignee", "commodity_description", "hs_code",
        "currency", "total_amount", "incoterm",
    ],
}  # fmt: skip


def hs_cargo_type(hs_code: str | None, description: str | None = "") -> str:
    """Map an HS code (chapter) to the quote engine's cargo types."""
    chapter = int(re.sub(r"\D", "", hs_code or "")[:2] or 0)
    if chapter == 15:  # animal/vegetable fats & oils incl. palm oil (1511)
        return "palm-oil"
    if chapter == 39:  # plastics and articles thereof
        return "plastics"
    if 28 <= chapter <= 38:  # chemical products
        return "chemicals"
    text = (description or "").lower()
    if "palm" in text:
        return "palm-oil"
    return "general"


def validate(doc: ShippingDocument) -> tuple[list[ValidationIssue], list[str]]:
    from src.costing.rates import get_rate_card  # noqa: PLC0415

    issues: list[ValidationIssue] = []
    for c in doc.container_numbers:
        if not is_valid(c):
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="container_numbers",
                    message=f"{c} fails the ISO 6346 check digit (likely a typo).",
                )
            )
    if (
        doc.container_count
        and doc.container_numbers
        and doc.container_count != len(doc.container_numbers)
    ):
        issues.append(
            ValidationIssue(
                severity="warning",
                field="container_count",
                message=f"{doc.container_count} containers declared but "
                f"{len(doc.container_numbers)} numbers listed.",
            )
        )
    if doc.hs_code and not re.fullmatch(r"\d{4}(\.?\d{2}){0,3}", doc.hs_code):
        issues.append(
            ValidationIssue(severity="error", field="hs_code", message="HS code format is invalid.")
        )
    rates = get_rate_card()
    n = doc.container_count or len(doc.container_numbers) or 1
    if doc.gross_weight_kg and doc.container_size in {"20", "40"}:
        tare = rates.container(doc.container_size)["tare_kg"]
        per_box = doc.gross_weight_kg / n + tare
        limit = rates.value("truck", "max_gross_weight_kg")
        if per_box > limit:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="gross_weight_kg",
                    message=f"{per_box:,.0f} kg per container incl. tare exceeds the "
                    f"{limit:,.0f} kg haulage limit at Port Klang.",
                )
            )
    if doc.net_weight_kg and doc.gross_weight_kg and doc.net_weight_kg > doc.gross_weight_kg:
        issues.append(
            ValidationIssue(
                severity="error", field="net_weight_kg", message="Net weight exceeds gross weight."
            )
        )
    if doc.dangerous_goods and not re.search(r"\bUN\s?\d{4}\b", doc.commodity_description or ""):
        issues.append(
            ValidationIssue(
                severity="warning",
                field="commodity_description",
                message="Dangerous goods without a UN number in the description.",
            )
        )
    missing = [f for f in REQUIRED.get(doc.doc_type, []) if getattr(doc, f) in (None, "", [])]
    return issues, missing


def suggested_quote(doc: ShippingDocument) -> dict[str, Any] | None:
    """Turn an inbound document into quote-engine inputs (destination from consignee)."""
    from src.costing.rates import get_rate_card  # noqa: PLC0415

    dest = get_rate_card().find_destination(doc.consignee or "")
    if not dest or "port klang" not in (doc.port_of_discharge or "").lower():
        return None
    n = doc.container_count or len(doc.container_numbers) or 1
    size = doc.container_size if doc.container_size in {"20", "40"} else "40"
    weight = (doc.gross_weight_kg or 18000 * n) / n
    return {
        "destination": dest,
        "container_size": size,
        "containers": n,
        "cargo_type": hs_cargo_type(doc.hs_code, doc.commodity_description),
        "cargo_weight_kg": round(min(max(weight, 100.0), 30000.0), 1),
        "origin": "northport"
        if "northport" in (doc.port_of_discharge or "").lower()
        else "westport",
    }


def extract(data: bytes, file_name: str, llm=None) -> ExtractionResult:
    """Extract, validate and map one PDF. Uses ``llm`` when given, rules otherwise."""
    text = pdf_text(data)
    if len(text.strip()) < 20:
        raise ValueError(
            f"{file_name} has no text layer (scanned image?). Run OCR or a vision model first."
        )
    method = "rules"
    doc = None
    if llm is not None:
        try:
            doc, method = parse_llm(text, llm), "llm"
        except Exception as exc:  # provider error or schema failure -> deterministic parser
            logger.warning("LLM extraction failed (%s); using rule-based parser.", exc)
    if doc is None:
        doc = parse_rules(text)
    issues, missing = validate(doc)
    return ExtractionResult(
        file_name=file_name,
        method=method,
        document=doc,
        issues=issues,
        missing_fields=missing,
        suggested_quote=suggested_quote(doc),
        text_chars=len(text),
    )
