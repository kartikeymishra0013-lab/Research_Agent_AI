"""
PDF document loader using PyMuPDF (fitz) with fallback to pdfminer.
Extracts text, metadata, and page-level information.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.ingestion.base_loader import BaseLoader, Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """
    Loads PDF files and extracts full text with metadata.

    Uses PyMuPDF (fitz) as primary engine for speed and accuracy.
    Falls back to pdfminer.six for complex/scanned-like PDFs.
    """

    def __init__(self, extract_images: bool = False, ocr_enabled: bool = False):
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled

    def load(self, source: str) -> Document:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {source}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {source}")

        logger.info(f"Loading PDF: {path.name}")
        try:
            return self._load_with_fitz(path)
        except ImportError:
            logger.warning("PyMuPDF not available, falling back to pdfminer")
            return self._load_with_pdfminer(path)

    def _load_with_fitz(self, path: Path) -> Document:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pages_text.append(page.get_text("text"))

        full_text = "\n\n".join(pages_text)

        # Extract metadata
        meta = doc.metadata or {}
        metadata = {
            "title": meta.get("title") or path.stem,
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "keywords": meta.get("keywords", ""),
            "page_count": len(doc),
            "file_size_bytes": path.stat().st_size,
            "pages_text": pages_text,
        }
        doc.close()

        logger.info(f"Extracted {len(full_text)} chars from {len(pages_text)} pages")
        return Document(
            content=full_text,
            source=str(path),
            doc_type="pdf",
            metadata=metadata,
        )

    def _load_with_pdfminer(self, path: Path) -> Document:
        from pdfminer.high_level import extract_text
        from pdfminer.pdfpage import PDFPage

        full_text = extract_text(str(path))

        # Count pages
        with open(path, "rb") as f:
            page_count = sum(1 for _ in PDFPage.get_pages(f))

        metadata = {
            "title": path.stem,
            "page_count": page_count,
            "file_size_bytes": path.stat().st_size,
        }

        logger.info(f"Extracted {len(full_text)} chars via pdfminer ({page_count} pages)")
        return Document(
            content=full_text,
            source=str(path),
            doc_type="pdf",
            metadata=metadata,
        )
