"""Generate "phone photo" scans of the sample Bill of Lading for the OCR demo.

    python scripts/make_sample_shipping_docs.py   # first, if data/samples is empty
    python scripts/make_scanned_samples.py

Writes ``bl_resin_import_scan.png`` (the clean B/L page rendered, rotated ~1 degree,
softened, unevenly lit on a grey desk with sensor noise) and ``bl_resin_import_scan.pdf``
(the same picture as an image-only PDF, i.e. no text layer). Output is deterministic.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extraction.ocr import page_images  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
SOURCE = "bl_resin_import.pdf"
DPI = 150
ROTATION_DEG = 0.8
MAX_BYTES = 500_000


def phone_photo(page: Image.Image, seed: int = 7) -> Image.Image:
    """Make a clean page look like a quick phone photo of paper on a desk."""
    rng = np.random.default_rng(seed)
    paper = page.convert("L").crop((0, 0, page.width, int(page.height * 0.45)))  # text part
    paper = paper.point(lambda p: 70 + p * (236 - 70) // 255)  # ink not black, paper off-white
    margin = 50
    desk = Image.new("L", (paper.width + 2 * margin, paper.height + 2 * margin), 128)
    desk.paste(paper, (margin, margin))
    photo = desk.rotate(ROTATION_DEG, resample=Image.BICUBIC, fillcolor=128)
    photo = photo.filter(ImageFilter.GaussianBlur(0.7))
    # Uneven lighting: darker towards the bottom-right corner, plus mild sensor noise.
    h, w = photo.height, photo.width
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 1.0 - 0.18 * ((xx / w) * 0.6 + (yy / h) * 0.4)
    arr = np.asarray(photo, dtype=np.float32) * shade + rng.normal(0, 5, (h, w))
    arr = np.clip(arr, 0, 255)
    # Quantise to 32 levels: invisible on screen, keeps the PNG well under 500 KB.
    arr = (arr // 8 * 8 + 4).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main() -> None:
    page = page_images((SAMPLES / SOURCE).read_bytes(), dpi=DPI)[0]
    photo = phone_photo(page)
    png = SAMPLES / "bl_resin_import_scan.png"
    photo.save(png, optimize=True)
    buf = io.BytesIO()
    photo.save(buf, format="PDF", resolution=DPI, quality=70)  # JPEG inside, no text layer
    pdf = SAMPLES / "bl_resin_import_scan.pdf"
    pdf.write_bytes(buf.getvalue())
    for path in (png, pdf):
        size = path.stat().st_size
        assert size < MAX_BYTES, f"{path.name} is {size:,} bytes"
        print(f"Wrote {path} ({size / 1000:.0f} KB)")


if __name__ == "__main__":
    main()
