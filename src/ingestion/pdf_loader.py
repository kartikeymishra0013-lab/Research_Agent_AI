"""
PDF loader: PyMuPDF primary → pdfminer fallback → OCR (pytesseract) for scanned PDFs.
OCR triggers when words/page < ocr_words_threshold (default 10).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.ingestion.base_loader import BaseLoader, Document
from src.utils.logger import get_logger

logger = get_logger(__name__)
OCR_WORDS_PER_PAGE_THRESHOLD = 10


class PDFLoader(BaseLoader):
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
        try:
            doc = self._load_with_fitz(path)
        except ImportError:
            logger.warning("PyMuPDF not available, falling back to pdfminer")
            doc = self._load_with_pdfminer(path)
        except Exception as e:
            logger.warning(f"PyMuPDF failed ({e}), falling back to pdfminer")
            doc = self._load_with_pdfminer(path)

        if self.ocr_enabled and self._needs_ocr(doc):
            logger.info(f"Low text yield ({doc.word_count} words) — attempting OCR")
            try:
                ocr_doc = self._load_with_ocr(path, doc.metadata)
                if ocr_doc.word_count > doc.word_count:
                    logger.info(f"OCR improved: {doc.word_count} -> {ocr_doc.word_count} words")
                    return ocr_doc
            except Exception as e:
                logger.warning(f"OCR failed: {e} — using original extraction")
        return doc

    def _load_with_fitz(self, path: Path) -> Document:
        import fitz
        doc = fitz.open(str(path))
        pages_text = [doc[i].get_text("text") for i in range(len(doc))]
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
        return Document(content=full_text, source=str(path), doc_type="pdf", metadata=metadata)

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
        return Document(content=full_text, source=str(path), doc_type="pdf", metadata=metadata)

    def _load_with_ocr(self, path: Path, existing_metadata: dict) -> Document:
        from pdf2image import convert_from_path
        import pytesseract
        logger.info(f"Running OCR on {path.name} at {self.ocr_dpi} DPI")
        images = convert_from_path(str(path), dpi=self.ocr_dpi)
        pages_text = []
        for i, image in enumerate(images):
            try:
                pages_text.append(pytesseract.image_to_string(image, lang=self.ocr_language, config="--psm 3"))
            except Exception as e:
                logger.warning(f"OCR failed on page {i+1}: {e}")
                pages_text.append("")
        full_text = "\n\n".join(pages_text)
        metadata = {**existing_metadata, "page_count": len(images),
                    "extraction_method": "ocr_tesseract",
                    "ocr_dpi": self.ocr_dpi, "ocr_language": self.ocr_language}
        logger.info(f"OCR extracted {len(full_text)} chars from {len(images)} pages")
        return Document(content=full_text, source=str(path), doc_type="pdf", metadata=metadata)

    def _needs_ocr(self, doc: Document) -> bool:
        page_count = doc.metadata.get("page_count", 1) or 1
        return (doc.word_count / page_count) < self.ocr_words_threshold
