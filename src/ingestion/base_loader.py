"""
Abstract base class for all document loaders.
Every loader must return a standardized Document object.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Document:
    """
    Standardized document representation returned by all loaders.

    Attributes:
        content   : Full extracted text content
        source    : Original source path or URL
        doc_type  : Detected document type (pdf, docx, txt, url)
        metadata  : Additional metadata (title, author, pages, etc.)
    """
    content: str
    source: str
    doc_type: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Document(type={self.doc_type!r}, source={self.source!r}, preview={preview!r}...)"

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        return len(self.content)


class BaseLoader(ABC):
    """Abstract base loader — all document loaders must implement `load`."""

    @abstractmethod
    def load(self, source: str) -> Document:
        """
        Load and extract text from the given source.

        Args:
            source: File path or URL string.

        Returns:
            Document with extracted text and metadata.
        """
        ...

    @staticmethod
    def _infer_doc_type(source: str) -> str:
        source_lower = source.lower()
        if source_lower.startswith("http://") or source_lower.startswith("https://"):
            return "url"
        ext = Path(source).suffix.lower()
        mapping = {".pdf": "pdf", ".docx": "docx", ".doc": "docx",
                   ".txt": "txt", ".md": "markdown", ".rst": "rst"}
        return mapping.get(ext, "unknown")
