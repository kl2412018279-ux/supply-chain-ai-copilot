"""OCR path for scanned / photographed B/Ls: engines, selection, wiring and parsing."""

import base64
import importlib.util
import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image, ImageDraw

from src.config import Settings
from src.extraction import extract as ex
from src.extraction import ocr
from src.extraction.extract import extract, parse_rules, pdf_text
from src.extraction.iso6346 import find_containers
from tests.conftest import ROOT

SAMPLES = ROOT / "data" / "samples"
CLEAN_TEXT = pdf_text((SAMPLES / "bl_resin_import.pdf").read_bytes())
needs_pdfium = pytest.mark.skipif(
    importlib.util.find_spec("pypdfium2") is None, reason="pypdfium2 not installed"
)
needs_tesseract = pytest.mark.skipif(
    not ocr.tesseract_available(), reason="tesseract binary or pytesseract not installed"
)


def _settings(**kw) -> Settings:
    base = {"llm_provider": "groq", "groq_api_key": "", "vision_model": "", "ocr_engine": "auto"}
    return Settings(**{**base, **kw})


def _fake_ocr(calls: list):
    def fake(images, settings=None, engine=None, client=None):
        calls.append(images)
        return ocr.OcrResult(text=CLEAN_TEXT, engine="tesseract", confidence=0.9)

    return fake


# ------------------------------------------------------------------- real Tesseract


@needs_tesseract
@pytest.mark.parametrize(
    "name",
    ["bl_resin_import_scan.png", pytest.param("bl_resin_import_scan.pdf", marks=needs_pdfium)],
)
def test_tesseract_reads_the_sample_scan(name):
    r = extract((SAMPLES / name).read_bytes(), name)
    d = r.document
    assert r.text_source == "ocr: tesseract" and 0 < r.ocr_confidence <= 1
    assert d.document_number == "SPLU2610045"
    assert d.container_numbers == ["QZXU2610042", "QZXU2610058"]
    assert d.gross_weight_kg == 41800
    assert d.hs_code == "3902.10" and d.container_size == "40" and d.container_count == 2
    assert r.issues == [] and r.missing_fields == []
    assert r.suggested_quote["destination"] == "shah-alam"
    assert len(r.warnings) == 1 and "container numbers and weights" in r.warnings[0]
    assert "SPLU2610045" in r.raw_text


# --------------------------------------------------------------------- wiring


@needs_pdfium
def test_image_only_pdf_goes_through_ocr(monkeypatch):
    calls: list = []
    monkeypatch.setattr(ex, "ocr_text", _fake_ocr(calls))
    r = extract((SAMPLES / "bl_resin_import_scan.pdf").read_bytes(), "scan.pdf")
    assert r.text_source == "ocr: tesseract" and r.ocr_confidence == 0.9
    assert len(calls) == 1 and isinstance(calls[0][0], Image.Image)  # page rendered by pdfium
    assert r.document.document_number == "SPLU2610045" and r.warnings


def test_image_upload_goes_through_ocr(monkeypatch):
    calls: list = []
    monkeypatch.setattr(ex, "ocr_text", _fake_ocr(calls))
    r = extract((SAMPLES / "bl_resin_import_scan.png").read_bytes(), "photo.PNG")
    assert r.text_source == "ocr: tesseract" and calls[0][0].size == (1375, 842)


def test_digital_pdf_keeps_its_text_layer(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("OCR must not run on a digital PDF")

    monkeypatch.setattr(ex, "ocr_text", boom)
    r = extract((SAMPLES / "bl_resin_import.pdf").read_bytes(), "bl.pdf")
    assert r.text_source == "pdf text" and r.ocr_confidence is None and r.warnings == []


def test_needs_ocr_threshold():
    assert ex.needs_ocr([""]) and ex.needs_ocr(["Page 1", "Page 2"])  # page numbers only
    assert not ex.needs_ocr(["x" * 40]) and not ex.needs_ocr([CLEAN_TEXT])


def test_scan_without_ocr_engine_explains_how_to_fix(monkeypatch):
    monkeypatch.setattr(ocr, "tesseract_available", lambda: False)
    data = (SAMPLES / "bl_resin_import_scan.png").read_bytes()
    with pytest.raises(ocr.OcrUnavailable, match="no text layer") as info:
        extract(data, "photo.png")
    assert isinstance(info.value, ValueError)  # UI/API treat it like other bad input
    assert "apt-get install tesseract-ocr" in str(info.value)
    assert "VISION_MODEL" in str(info.value)


def test_api_accepts_images(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)  # no lifespan: no model training needed for this endpoint
    png = (SAMPLES / "bl_resin_import_scan.png").read_bytes()
    monkeypatch.setattr(ocr, "tesseract_available", lambda: False)
    r = client.post("/documents/extract", files={"file": ("scan.png", png, "image/png")})
    assert r.status_code == 503 and "tesseract" in r.json()["detail"].lower()

    monkeypatch.setattr(ex, "ocr_text", _fake_ocr([]))
    r = client.post("/documents/extract", files={"file": ("scan.png", png, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["text_source"] == "ocr: tesseract"
    assert body["document"]["container_numbers"] == ["QZXU2610042", "QZXU2610058"]
    assert client.post("/documents/extract", files={"file": ("x.gif", b"GIF8")}).status_code == 415


# -------------------------------------------------------------- engine selection


@pytest.mark.parametrize(
    ("engine", "vision", "tesseract", "expected"),
    [
        ("auto", True, True, "vision"),
        ("auto", False, True, "tesseract"),
        ("auto", True, False, "vision"),
        ("tesseract", True, True, "tesseract"),
        ("vision", True, False, "vision"),
    ],
)
def test_engine_selection(monkeypatch, engine, vision, tesseract, expected):
    monkeypatch.setattr(ocr, "tesseract_available", lambda: tesseract)
    key = {"vision_model": "m", "groq_api_key": "k"} if vision else {}
    s = _settings(ocr_engine=engine, **key)
    assert ocr.resolve_engine(s) == expected


@pytest.mark.parametrize(
    ("engine", "message"),
    [
        ("auto", "No OCR engine is available"),
        ("tesseract", "Tesseract is not installed"),
        ("vision", "needs VISION_MODEL"),
        ("none", "switched off"),
        ("magic", "Unknown OCR_ENGINE"),
    ],
)
def test_ocr_unavailable_messages(monkeypatch, engine, message):
    monkeypatch.setattr(ocr, "tesseract_available", lambda: False)
    with pytest.raises(ocr.OcrUnavailable, match=message):
        ocr.ocr_text([Image.new("L", (10, 10))], _settings(ocr_engine=engine))


def test_vision_model_without_key_is_not_configured():
    assert not ocr.vision_configured(_settings(vision_model="m"))
    s = _settings(vision_model="m", llm_provider="openai", openai_api_key="sk")
    assert ocr.vision_endpoint(s) == (ocr.VISION_ENDPOINTS["openai"], "sk")


# ------------------------------------------------------------------ vision backend


def test_vision_backend_request_and_response():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        assert body["model"] == "test-vision-model" and body["temperature"] == 0
        assert "verbatim" in content[0]["text"] and "summarise" in content[0]["text"]
        url = content[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert max(img.size) <= ocr.MAX_VISION_SIDE_PX
        return httpx.Response(200, json={"choices": [{"message": {"content": CLEAN_TEXT}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    s = _settings(vision_model="test-vision-model", groq_api_key="gsk-test")
    pages = [Image.new("RGB", (3000, 2000), "white"), Image.new("L", (800, 600), 255)]
    result = ocr.ocr_text(pages, s, client=client)
    assert result.engine == "vision" and result.confidence is None
    assert result.text.count("SPLU2610045") == 2  # one call per page, joined
    assert len(seen) == 2 and str(seen[0].url) == ocr.VISION_ENDPOINTS["groq"]
    assert seen[0].headers["authorization"] == "Bearer gsk-test"


def test_vision_failure_falls_back_to_tesseract_in_auto_mode(monkeypatch):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    s = _settings(vision_model="m", groq_api_key="k")
    monkeypatch.setattr(ocr, "tesseract_available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_tesseract", lambda imgs: ocr.OcrResult("ok", "tesseract"))
    assert ocr.ocr_text([Image.new("L", (10, 10))], s, client=client).engine == "tesseract"
    with pytest.raises(httpx.HTTPStatusError):  # explicit vision: no silent fallback
        ocr.ocr_text([Image.new("L", (10, 10))], replace(s, ocr_engine="vision"), client=client)


# ------------------------------------------------------------- images & parsing


@needs_pdfium
def test_page_images_renders_pdf_pages():
    pages = ocr.page_images((SAMPLES / "bl_resin_import.pdf").read_bytes(), dpi=100)
    assert len(pages) == 1 and pages[0].size == (850, 1100)  # US letter at 100 dpi


def test_preprocess_upscales_and_straightens():
    img = Image.new("L", (1200, 900), 255)
    draw = ImageDraw.Draw(img)
    for y in range(100, 800, 40):
        draw.rectangle((100, y, 1100, y + 12), fill=0)  # text-like lines
    tilted = img.rotate(1.5, resample=Image.BICUBIC, fillcolor=255)
    assert abs(ocr._skew_angle(tilted) + 1.5) <= 0.25
    out = ocr.preprocess(tilted.convert("RGB"))
    assert out.mode == "L" and out.width >= ocr.MIN_WIDTH_PX


def test_container_repair_only_when_check_digit_confirms():
    assert find_containers("QZXU26lOO42", repair=True) == ["QZXU2610042"]
    assert find_containers("QZXU26lOO42") == []  # strict mode unchanged
    assert find_containers("QZXU7001125", repair=True) == ["QZXU7001125"]  # kept, flagged later
    assert find_containers("QZXU26lOO43", repair=True) == []  # repair would not validate


def test_rule_parser_tolerates_ocr_noise():
    noisy = CLEAN_TEXT.replace("B/L No:", "| B / L No.: ").replace(
        "QZXU2610042, QZXU2610058", "QZXU261OO42 | QZXU2610O58"
    )
    noisy = noisy.replace("41,800.000 KGS", "41, 800.000  KGS |")
    d = parse_rules(noisy)
    assert d.document_number == "SPLU2610045"
    assert d.container_numbers == ["QZXU2610042", "QZXU2610058"]
    assert d.gross_weight_kg == 41800
