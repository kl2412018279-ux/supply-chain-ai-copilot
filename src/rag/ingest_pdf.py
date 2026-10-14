"""Read PDFs (or .txt/.md), split into overlapping chunks, embed and store them.

Usage::

    python -m src.rag.ingest_pdf data/raw/*.pdf data/raw/*.md
"""

from __future__ import annotations

import argparse
import io
import logging
import re
from pathlib import Path

from src.config import get_settings
from src.rag.embeddings import Embedder, get_embedder
from src.rag.schema import Chunk
from src.rag.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def extract_pages(data: bytes, filename: str) -> list[tuple[int, str]]:
    """Return ``(page_number, text)`` pairs. Text files count as a single page."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )
    if suffix == ".pdf":
        from pypdf import PdfReader  # noqa: PLC0415

        reader = PdfReader(io.BytesIO(data))
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    return [(1, data.decode("utf-8", errors="replace"))]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Paragraph-aware chunking: pack paragraphs up to ``chunk_size`` characters,
    splitting oversized paragraphs on sentence boundaries, with character overlap."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for s in sentences:
            while len(s) > chunk_size:
                pieces.append(s[:chunk_size])
                s = s[chunk_size - overlap :]
            pieces.append(s)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
        current = f"{tail} {piece}".strip() if len(tail) + len(piece) < chunk_size else piece
    if current:
        chunks.append(current)
    return chunks


def ingest_bytes(
    data: bytes,
    filename: str,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> int:
    """Ingest one document. Re-ingesting the same filename replaces its chunks."""
    settings = get_settings()
    store = store or get_vector_store()
    embedder = embedder or get_embedder()
    source = Path(filename).name

    chunks: list[Chunk] = []
    for page, text in extract_pages(data, filename):
        for piece in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(Chunk(source=source, chunk_index=len(chunks), page=page, text=piece))
    if not chunks:
        raise ValueError(f"No extractable text in {filename} (scanned PDF? run OCR first).")

    store.delete_source(source)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    added = store.add(chunks, embeddings)
    logger.info("Ingested %s: %d chunks", source, added)
    return added


def ingest_file(path: str | Path, store: VectorStore | None = None, embedder=None) -> int:
    path = Path(path)
    return ingest_bytes(path.read_bytes(), path.name, store, embedder)


def ingest_directory(directory: str | Path, store: VectorStore | None = None, embedder=None) -> int:
    total = 0
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES and not path.name.lower().startswith("readme"):
            total += ingest_file(path, store, embedder)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("paths", nargs="*", help="files to ingest (default: DOCUMENTS_DIR)")
    args = parser.parse_args()
    if args.paths:
        total = sum(ingest_file(p) for p in args.paths)
    else:
        total = ingest_directory(get_settings().documents_dir)
    print(f"Ingested {total} chunks. Store now holds {get_vector_store().count()} chunks.")


if __name__ == "__main__":
    main()
