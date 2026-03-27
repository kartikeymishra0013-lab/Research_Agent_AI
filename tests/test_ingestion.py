"""
Tests for the document ingestion layer.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.ingestion import get_loader
from src.ingestion.base_loader import Document
from src.ingestion.text_loader import TextLoader


class TestTextLoader:
    def test_load_markdown_file(self, sample_text_path):
        loader = TextLoader()
        doc = loader.load(str(sample_text_path))

        assert isinstance(doc, Document)
        assert doc.doc_type == "markdown"
        assert "Test Document" in doc.content
        assert doc.metadata["title"] == "test_doc"
        assert doc.word_count > 0

    def test_load_strips_markdown(self, temp_dir):
        content = "# Heading\n\n**Bold text** and _italic_.\n\n[Link](https://example.com)"
        path = temp_dir / "test.md"
        path.write_text(content)

        loader = TextLoader(strip_markdown=True)
        doc = loader.load(str(path))

        assert "Heading" in doc.content
        assert "**" not in doc.content
        assert "[Link]" not in doc.content

    def test_load_nonexistent_file_raises(self):
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")

    def test_load_unsupported_extension_raises(self, temp_dir):
        path = temp_dir / "test.xyz"
        path.write_text("content")
        loader = TextLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(str(path))


class TestLoaderFactory:
    def test_get_pdf_loader(self):
        from src.ingestion.pdf_loader import PDFLoader
        loader = get_loader("test.pdf")
        assert isinstance(loader, PDFLoader)

    def test_get_docx_loader(self):
        from src.ingestion.docx_loader import DocxLoader
        loader = get_loader("test.docx")
        assert isinstance(loader, DocxLoader)

    def test_get_text_loader(self):
        loader = get_loader("test.txt")
        from src.ingestion.text_loader import TextLoader
        assert isinstance(loader, TextLoader)

    def test_get_markdown_loader(self):
        loader = get_loader("test.md")
        from src.ingestion.text_loader import TextLoader
        assert isinstance(loader, TextLoader)

    def test_get_web_loader(self):
        from src.ingestion.web_loader import WebLoader
        loader = get_loader("https://example.com/article")
        assert isinstance(loader, WebLoader)

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError):
            get_loader("test.xyz")


class TestDocument:
    def test_document_word_count(self, sample_document):
        assert sample_document.word_count > 50

    def test_document_char_count(self, sample_document):
        assert sample_document.char_count > 200

    def test_document_repr(self, sample_document):
        repr_str = repr(sample_document)
        assert "pdf" in repr_str
        assert "attention_paper" in repr_str
