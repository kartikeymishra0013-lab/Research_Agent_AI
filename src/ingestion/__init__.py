"""
Document ingestion layer — loaders for PDF, DOCX, text, and web URLs.
"""
from src.ingestion.base_loader import BaseLoader, Document
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.docx_loader import DocxLoader
from src.ingestion.text_loader import TextLoader
from src.ingestion.web_loader import WebLoader

__all__ = ["BaseLoader", "Document", "PDFLoader", "DocxLoader", "TextLoader", "WebLoader", "get_loader"]


def get_loader(source: str, config: dict = None) -> BaseLoader:
    """
    Factory: returns the appropriate loader for a given source.

    Args:
        source: File path or URL string.
        config: Optional loader-specific config overrides.

    Returns:
        A configured loader instance.
    """
    cfg = config or {}
    source_lower = source.lower()

    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        return WebLoader(**cfg.get("web", {}))

    from pathlib import Path
    ext = Path(source).suffix.lower()

    if ext == ".pdf":
        return PDFLoader(**cfg.get("pdf", {}))
    elif ext in {".docx", ".doc"}:
        return DocxLoader(**cfg.get("docx", {}))
    elif ext in {".txt", ".md", ".rst", ".markdown", ".text"}:
        return TextLoader(**cfg.get("text", {}))
    else:
        raise ValueError(f"No loader available for source: {source!r} (extension: {ext!r})")
