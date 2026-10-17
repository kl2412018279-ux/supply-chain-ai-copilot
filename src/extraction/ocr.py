"""OCR for scanned or photographed shipping documents (images and image-only PDFs).

Pipeline::

    PDF -> page images (pypdfium2, no poppler) | image upload (Pillow)
        -> preprocessing (grayscale, auto-contrast, upscale, deskew)
        -> text (vision LLM when VISION_MODEL is set, else Tesseract when installed)

Both engines return plain text, so the same parsers and ISO 6346 / weight validation
run on OCR output as on digital PDFs. OCR misreads are expected: the caller flags OCR
text for review, and container check digits catch most misread container numbers.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image, ImageOps, ImageSequence

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
ENGINES = ("auto", "tesseract", "vision", "none")
VISION_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}
INSTALL_HINT = (
    "Install Tesseract (apt-get install tesseract-ocr, or brew install tesseract) plus "
    "`pip install pytesseract`, or set VISION_MODEL and GROQ_API_KEY to use a vision model."
)
VISION_PROMPT = (
    "Transcribe all text in this scanned shipping document verbatim, line by line, in "
    "reading order. Keep each label and its value on one line as printed (e.g. "
    "'B/L No: ...'). Copy container numbers, weights, dates and codes character for "
    "character. Do not summarise, translate, correct or add anything; output only the "
    "transcribed text."
)
MIN_WIDTH_PX = 1600  # upscale small photos so ~10 pt text is >= 20 px tall
MAX_VISION_SIDE_PX = 2000  # keep base64 payloads well under provider limits
TESSERACT_TIMEOUT_S = 120  # per page


class OcrUnavailable(ValueError):
    """No OCR engine is installed or configured, or OCR is switched off."""


@dataclass(frozen=True)
class OcrResult:
    text: str
    engine: str  # "tesseract" or "vision"
    confidence: float | None = None  # mean word confidence 0-1 (Tesseract only)


# -------------------------------------------------------------------------- images


def page_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    """Render every PDF page to a PIL image (pypdfium2: pure pip, no poppler)."""
    try:
        import pypdfium2 as pdfium  # noqa: PLC0415
    except ImportError as exc:
        raise OcrUnavailable("Rendering scanned PDF pages needs `pip install pypdfium2`.") from exc

    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        return [page.render(scale=dpi / 72).to_pil() for page in pdf]
    finally:
        pdf.close()


def load_images(data: bytes) -> list[Image.Image]:
    """All frames of an uploaded image (multi-page TIFFs have several)."""
    try:
        img = Image.open(io.BytesIO(data))
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
    except Exception as exc:  # Pillow raises several error types for bad files
        raise ValueError(f"Could not read the image ({exc}).") from exc
    return [ImageOps.exif_transpose(f) or f for f in frames]  # phone photos carry rotation


def _skew_angle(gray: Image.Image, max_deg: float = 3.0, step: float = 0.25) -> float:
    """Angle that makes text lines horizontal: maximise the row-projection variance."""
    small = gray.copy()
    small.thumbnail((900, 900))
    ink = small.point(lambda p: 255 if p < 128 else 0)  # text white on black
    best, best_score = 0.0, -1.0
    for i in range(int(-max_deg / step), int(max_deg / step) + 1):
        angle = i * step
        rows = np.asarray(ink.rotate(angle, resample=Image.BILINEAR), dtype=np.float32).sum(1)
        score = float(np.var(np.diff(rows)))
        if score > best_score:
            best, best_score = angle, score
    return best


def preprocess(img: Image.Image, deskew: bool = True) -> Image.Image:
    """Grayscale, auto-contrast, upscale small images and (optionally) straighten."""
    gray = ImageOps.grayscale(img.convert("RGB") if img.mode in {"P", "RGBA", "LA"} else img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    if gray.width < MIN_WIDTH_PX:
        factor = MIN_WIDTH_PX / gray.width
        gray = gray.resize((MIN_WIDTH_PX, round(gray.height * factor)), Image.LANCZOS)
    if deskew:
        angle = _skew_angle(gray)
        if angle:
            gray = gray.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    return gray


# ------------------------------------------------------------------------- engines


def tesseract_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def vision_endpoint(settings: Settings) -> tuple[str, str]:
    """(URL, API key) for the OpenAI-compatible vision endpoint; Groq unless on OpenAI."""
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return VISION_ENDPOINTS["openai"], settings.openai_api_key
    return VISION_ENDPOINTS["groq"], settings.groq_api_key


def vision_configured(settings: Settings) -> bool:
    return bool(settings.vision_model and vision_endpoint(settings)[1])


def ocr_tesseract(images: Sequence[Image.Image]) -> OcrResult:
    """Tesseract (LSTM engine, automatic page segmentation) with word confidences."""
    import pytesseract  # noqa: PLC0415

    # Tesseract's OpenMP threads crawl on busy/shared CPUs (e.g. Streamlit Cloud); one thread
    # per call is faster there and barely slower on an idle machine.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    pages, confs = [], []
    for img in images:
        data = pytesseract.image_to_data(
            img,
            config="--oem 1 --psm 3",
            output_type=pytesseract.Output.DICT,
            timeout=TESSERACT_TIMEOUT_S,
        )
        lines: dict[tuple[int, int, int], list[str]] = {}
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            if float(data["conf"][i]) >= 0:
                confs.append(float(data["conf"][i]))
        pages.append("\n".join(" ".join(words) for words in lines.values()))
    confidence = round(sum(confs) / len(confs) / 100, 3) if confs else None
    return OcrResult(text="\n\n".join(pages), engine="tesseract", confidence=confidence)


def _data_url(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail((MAX_VISION_SIDE_PX, MAX_VISION_SIDE_PX))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ocr_vision(
    images: Sequence[Image.Image],
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> OcrResult:
    """Transcribe each page with a vision chat model (OpenAI-compatible API, one call/page)."""
    settings = settings or get_settings()
    url, key = vision_endpoint(settings)
    own_client = client is None
    client = client or httpx.Client(timeout=60)
    pages = []
    try:
        for img in images:
            resp = client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": settings.vision_model,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": VISION_PROMPT},
                                {"type": "image_url", "image_url": {"url": _data_url(img)}},
                            ],
                        }
                    ],
                },
            )
            resp.raise_for_status()
            pages.append((resp.json()["choices"][0]["message"]["content"] or "").strip())
    finally:
        if own_client:
            client.close()
    return OcrResult(text="\n\n".join(pages), engine="vision")


def resolve_engine(settings: Settings | None = None, engine: str | None = None) -> str:
    """Pick "vision" or "tesseract" for ``engine`` (default OCR_ENGINE) or raise OcrUnavailable."""
    settings = settings or get_settings()
    engine = (engine or settings.ocr_engine or "auto").lower()
    if engine not in ENGINES:
        raise OcrUnavailable(f"Unknown OCR_ENGINE {engine!r}; use one of {', '.join(ENGINES)}.")
    if engine == "none":
        raise OcrUnavailable("OCR is switched off (OCR_ENGINE=none).")
    if engine == "vision":
        if not vision_configured(settings):
            raise OcrUnavailable("OCR_ENGINE=vision needs VISION_MODEL and GROQ_API_KEY.")
        return "vision"
    if engine == "tesseract":
        if not tesseract_available():
            raise OcrUnavailable(f"Tesseract is not installed. {INSTALL_HINT}")
        return "tesseract"
    if vision_configured(settings):
        return "vision"
    if tesseract_available():
        return "tesseract"
    raise OcrUnavailable(f"No OCR engine is available. {INSTALL_HINT}")


def ocr_text(
    images: Sequence[Image.Image],
    settings: Settings | None = None,
    engine: str | None = None,
    client: httpx.Client | None = None,
) -> OcrResult:
    """OCR page images with the configured engine.

    In "auto" mode a failing vision call falls back to Tesseract when it is installed.
    """
    settings = settings or get_settings()
    chosen = resolve_engine(settings, engine)
    if chosen == "vision":
        try:
            return ocr_vision(images, settings, client)
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            auto = (engine or settings.ocr_engine or "auto").lower() == "auto"
            if not (auto and tesseract_available()):
                raise
            logger.warning("Vision OCR failed (%s); falling back to Tesseract.", exc)
    return ocr_tesseract([preprocess(img) for img in images])
