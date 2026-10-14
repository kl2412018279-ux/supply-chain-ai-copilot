"""Render a plain-text/markdown file to a minimal multi-page PDF (no dependencies).

Usage: python scripts/make_sample_pdf.py scripts/sample_docs/esg_policy.md data/raw/esg_policy.pdf
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

LINES_PER_PAGE = 50
WRAP = 95


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_to_pdf(text: str) -> bytes:
    lines: list[str] = []
    for para in text.splitlines():
        lines.extend(textwrap.wrap(para, WRAP) or [""])
    pages = [lines[i : i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)] or [[]]

    objects: list[bytes] = []
    n_pages = len(pages)
    page_ids = [3 + 2 * i for i in range(n_pages)]
    font_id = 3 + 2 * n_pages
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, page_lines in enumerate(pages):
        content_id = page_ids[i] + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        body = "BT /F1 10 Tf 13 TL 50 750 Td\n"
        body += "".join(f"({_escape(line)}) Tj T*\n" for line in page_lines) + "ET"
        stream = body.encode("latin-1", errors="replace")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_bytes(text_to_pdf(src.read_text(encoding="utf-8")))
    print(f"Wrote {dst}")
