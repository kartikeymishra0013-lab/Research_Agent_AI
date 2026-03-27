"""
Plain text and Markdown file loader.
"""
from __future__ import annotations

from pathlib import Path

from src.ingestion.base_loader import BaseLoader, Document
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".text", ".markdown"}


class TextLoader(BaseLoader):
    """Loads plain text and Markdown files."""

    def __init__(self, encoding: str = "utf-8", strip_markdown: bool = False):
        self.encoding = encoding
        self.strip_markdown = strip_markdown

    def load(self, source: str) -> Document:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {source}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported text format: {path.suffix}")

        logger.info(f"Loading text file: {path.name}")
        content = path.read_text(encoding=self.encoding, errors="replace")

        doc_type = "markdown" if path.suffix.lower() in {".md", ".markdown"} else "txt"

        if self.strip_markdown and doc_type == "markdown":
            content = self._strip_markdown(content)

        metadata = {
            "title": path.stem,
            "file_size_bytes": path.stat().st_size,
            "line_count": content.count("\n"),
            "encoding": self.encoding,
        }

        logger.info(f"Loaded {len(content)} chars from {path.name}")
        return Document(
            content=content,
            source=str(path),
            doc_type=doc_type,
            metadata=metadata,
        )

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove basic Markdown syntax to get plain text."""
        import re
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)
        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
        # Remove links
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", text)
        # Remove horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        return text.strip()
