import os

import pytest

from src.extraction.extract import (
    DocType,
    ShippingDocument,
    extract,
    hs_cargo_type,
    parse_llm,
    parse_rules,
    validate,
)
from src.extraction.iso6346 import check_digit, find_containers, is_valid
from src.extraction.store import InMemoryDocumentStore, PostgresDocumentStore
from tests.conftest import ROOT

SAMPLES = ROOT / "data" / "samples"


def _extract(name, **kw):
    path = SAMPLES / name
    return extract(path.read_bytes(), path.name, **kw)


def test_iso6346_check_digit_matches_published_example():
    assert check_digit("CSQU305438") == 3
    assert is_valid("CSQU3054383") and is_valid("csqu 305438-3")
    assert not is_valid("CSQU3054384") and not is_valid("CSQX3054383")


def test_find_containers_dedupes_and_normalises():
    assert find_containers("qzxu 261004 2, QZXU2610042 and QZXU2610058") == [
        "QZXU2610042",
        "QZXU2610058",
    ]


def test_bill_of_lading_fields():
    r = _extract("bl_resin_import.pdf")
    d = r.document
    assert r.method == "rules" and d.doc_type is DocType.bill_of_lading
    assert d.document_number == "SPLU2610045"  # not mistaken for a container number
    assert d.container_numbers == ["QZXU2610042", "QZXU2610058"]
    assert d.container_count == 2 and d.container_size == "40"
    assert d.gross_weight_kg == 41800 and d.hs_code == "3902.10"
    assert d.shipper.startswith("Lianhe Polymer") and "Shah Alam" in d.consignee
    assert r.issues == [] and r.missing_fields == []


def test_invoice_fields():
    d = _extract("invoice_cpo_export.pdf").document
    assert d.doc_type is DocType.commercial_invoice
    assert (d.currency, d.total_amount, d.incoterm) == ("USD", 68355.0, "FOB")
    assert d.net_weight_kg == 63000 and len(d.container_numbers) == 3


def test_errors_are_caught():
    r = _extract("bl_chemicals_errors.pdf")
    messages = " ".join(i.message for i in r.issues)
    assert "QZXU7001125 fails the ISO 6346 check digit" in messages
    assert "exceeds the 25,000 kg" in messages
    assert r.document.dangerous_goods


def test_suggested_quote_links_to_quote_engine():
    q = _extract("bl_resin_import.pdf").suggested_quote
    assert q == {
        "destination": "shah-alam",
        "container_size": "40",
        "containers": 2,
        "cargo_type": "plastics",
        "cargo_weight_kg": 20900.0,
        "origin": "westport",
    }
    assert _extract("invoice_cpo_export.pdf").suggested_quote is None  # export to Rotterdam


@pytest.mark.parametrize(
    ("hs", "expected"),
    [
        ("1511.90", "palm-oil"),
        ("3902.10", "plastics"),
        ("2915.21", "chemicals"),
        ("8471", "general"),
    ],
)
def test_hs_mapping(hs, expected):
    assert hs_cargo_type(hs) == expected


def test_missing_fields_and_bad_values_reported():
    doc = ShippingDocument(
        doc_type=DocType.commercial_invoice, document_number="X1", hs_code="12AB",
        gross_weight_kg=100, net_weight_kg=200,
    )  # fmt: skip
    issues, missing = validate(doc)
    assert {"shipper", "consignee", "currency", "total_amount"} <= set(missing)
    assert {i.field for i in issues} >= {"hs_code", "net_weight_kg"}


def test_scanned_pdf_without_text_is_rejected():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "make_sample_pdf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(ValueError, match="no text layer"):
        extract(mod.text_to_pdf(""), "scan.pdf")


def test_llm_path_keeps_only_container_numbers_present_in_text():
    text = (SAMPLES / "bl_resin_import.pdf").read_bytes()
    from src.extraction.extract import pdf_text

    raw = pdf_text(text)

    class FakeStructured:
        def invoke(self, prompt):
            assert "never guess" in prompt
            doc = parse_rules(raw)
            doc.container_numbers = ["QZXU2610042", "ABCU1234560"]  # second one is hallucinated
            return doc

    class FakeLLM:
        def with_structured_output(self, schema):
            assert schema is ShippingDocument
            return FakeStructured()

    doc = parse_llm(raw, FakeLLM())
    assert doc.container_numbers == ["QZXU2610042"]
    assert _extract("bl_resin_import.pdf", llm=FakeLLM()).method == "llm"


def test_llm_failure_falls_back_to_rules():
    class BrokenLLM:
        def with_structured_output(self, schema):
            raise RuntimeError("provider down")

    r = _extract("bl_resin_import.pdf", llm=BrokenLLM())
    assert r.method == "rules" and r.document.document_number == "SPLU2610045"


def test_store_upserts_by_document_number():
    store = InMemoryDocumentStore()
    d = _extract("bl_resin_import.pdf").document
    store.save(d, "a.pdf", [])
    store.save(d, "b.pdf", [])
    rows = store.list()
    assert len(rows) == 1 and rows[0]["file_name"] == "b.pdf"
    with pytest.raises(ValueError):
        store.save(ShippingDocument(), "c.pdf", [])


@pytest.mark.integration
def test_postgres_store():
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("TEST_DATABASE_URL not set")
    store = PostgresDocumentStore(dsn)
    r = _extract("bl_chemicals_errors.pdf")
    store.save(r.document, r.file_name, [i.model_dump() for i in r.issues])
    row = next(x for x in store.list() if x["document_number"] == "SPLU2610311")
    assert row["data"]["hs_code"] == "2915.21" and len(row["issues"]) == 2
