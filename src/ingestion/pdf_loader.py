"""
PDF document loader using PyMuPDF (fitz) with pdfminer fallback and OCR support.

Extraction priority:
  1. PyMuPDF  — fast, accurate for digital PDFs
  2. pdfminer — secondary, handles some edge cases
  3. OCR      — pytesseract via pdf2image, used when text yield is low (scanned PDFs)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.ingestion.base_loader import BaseLoader, Document
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Pages with fewer words than this trigger OCR fallback
OCR_WORDS_PER_PAGE_THRESHOLD = 10


class PDFLoader(BaseLoader):
    """
    Loads PDF files and extracts full text with metadata.

    Strategies (in order of preference):
        1. PyMuPDF  — fast, accurate for digital PDFs
        2. pdfminer — handles complex layouts / ligatures
        3. OCR      — pytesseract + pdf2image for scanned / image-only PDFs

    Args:
        extract_images      : Unused flag kept for API compatibility.
        ocr_enabled         : If True, use OCR when digital extraction yields
                              fewer than `ocr_words_per_page_threshold` words/page.
        ocr_dpi             : DPI for pdf2image rasterization (default 300).
        ocr_language        : Tesseract language code(s) (default "eng").
        ocr_words_threshold : Words-per-page below which OCR is triggered.
    """

    def __init__(
        self,
        extract_images: bool = False,
        ocr_enabled: bool = True,
        ocr_dpi: int = 300,
        ocr_language: str = "eng",
        ocr_words_threshold: int = OCR_WORDS_PER_PAGE_THRESHOLD,
    ):
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled
        self.ocr_dpi = ocr_dpi
        self.ocr_language = ocr_language
        self.ocr_words_threshold = ocr_words_threshold

    def load(self, source: str) -> Document:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {source}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {source}")

        logger.info(f"Loading PDF: {path.name}")

        # Try PyMuPDF first
        try:
            doc = self._load_with_fitz(path)
        except ImportError:
            logger.warning("PyMuPDF not available, falling back to pdfminer")
            doc = self._load_with_pdfminer(path)
        except Exception as e:
            logger.warning(f"PyMuPDF failed ({e}), falling back to pdfminer")
            doc = self._load_with_pdfminer(path)

        # OCR fallback: if text yield is suspiciously low, the PDF is likely scanned
        if self.ocr_enabled and self._needs_ocr(doc):
            logger.info(
                f"Low text yield detected ({doc.word_count} words, "
                f"{doc.metadata.get('page_count', 1)} pages) — attempting OCR"
            )
            try:
                ocr_doc = self._load_with_ocr(path, doc.metadata)
                if ocr_doc.word_count > doc.word_count:
                    logger.info(
                        f"OCR improved extraction: {doc.word_count} → {ocr_doc.word_count} words"
                    )
                    return ocr_doc
                else:
                    logger.info("OCR did not improve yield; keeping original extraction")
            except Exception as e:
                logger.warning(f"OCR failed: {e} — using original extraction")

        return doc

    # ─── Extraction backends ───────────────────────────────────────────────────

    def _load_with_fitz(self, path: Path) -> Document:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pages_text.append(page.get_text("text"))

        full_text = "\n\n".join(pages_text)
        meta = doc.metadata or {}
        metadata = {
            "title": meta.get("title") or path.stem,
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "keywords": meta.get("keywords", ""),
            "page_count": len(doc),
            "file_size_bytes": path.stat().st_size,
            "pages_text": pages_text,
            "extraction_method": "pymupdf",
        }
        doc.close()

        logger.info(f"Extracted {len(full_text)} chars from {len(pages_text)} pages (PyMuPDF)")
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

        with open(path, "rb") as f:
            page_count = sum(1 for _ in PDFPage.get_pages(f))

        metadata = {
            "title": path.stem,
            "page_count": page_count,
            "file_size_bytes": path.stat().st_size,
            "extraction_method": "pdfminer",
        }

        logger.info(f"Extracted {len(full_text)} chars via pdfminer ({page_count} pages)")
        return Document(
            content=full_text,
            source=str(path),
            doc_type="pdf",
            metadata=metadata,
        )

    def _load_with_ocr(self, path: Path, existing_metadata: dict) -> Document:
        """
        Rasterize each PDF page with pdf2image and run Tesseract OCR on the images.

        Returns a new Document whose content is the concatenated OCR text.
        """
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ImportError(
                "pdf2image is required for OCR. Install it with: pip install pdf2image"
            ) from e

        try:
            import pytesseract
        except ImportError as e:
            raise ImportError(
                "pytesseract is required for OCR. Install it with: pip install pytesseract"
            ) from e

        logger.info(f"Running OCR on {path.name} at {self.ocr_dpi} DPI (lang={self.ocr_language!r})")

        images = convert_from_path(str(path), dpi=self.ocr_dpi)
        pages_text = []
        for i, image in enumerate(images):
            try:
                page_text = pytesseract.image_to_string(
                    image,
                    lang=self.ocr_language,
                    config="--psm 3",  # fully automatic page segmentation
                )
                pages_text.append(page_text)
            except Exception as e:
                logger.warning(f"OCR failed on page {i + 1}: {e}")
                pages_text.append("")

        full_text = "\n\n".join(pages_text)
        metadata = {
            **existing_metadata,
            "page_count": len(images),
            "extraction_method": "ocr_tesseract",
            "ocr_dpi": self.ocr_dpi,
            "ocr_language": self.ocr_language,
        }

        logger.info(f"OCR extracted {len(full_text)} chars from {len(images)} pages")
        return Document(
            content=full_text,
            source=str(path),
            doc_type="pdf",
            metadata=metadata,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _needs_ocr(self, doc: Document) -> bool:
        """Return True if the document's word yield is below the OCR trigger threshold."""
        page_count = doc.metadata.get("page_count", 1) or 1
        words_per_page = doc.word_count / page_count
        return words_per_page < self.ocr_words_threshold
