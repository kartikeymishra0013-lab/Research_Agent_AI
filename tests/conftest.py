"""
Shared pytest fixtures for the Scientific Document Intelligence Pipeline tests.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.base_loader import Document
from src.utils.chunker import Chunk


# ─── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_document():
    """A minimal Document object for testing."""
    return Document(
        content=(
            "Attention mechanisms have become a key component of sequence modeling. "
            "In this paper, we propose the Transformer, a model architecture based "
            "solely on attention mechanisms, dispensing with recurrence and convolutions. "
            "Experiments on two machine translation tasks show these models to be superior "
            "in quality while being more parallelizable and requiring significantly less "
            "training time. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German "
            "translation task and 41.8 BLEU on the WMT 2014 English-to-French translation task."
        ),
        source="/data/input/attention_paper.pdf",
        doc_type="pdf",
        metadata={
            "title": "Attention Is All You Need",
            "author": "Vaswani et al.",
            "page_count": 15,
        },
    )


@pytest.fixture
def sample_chunks():
    """A list of Chunk objects for testing."""
    texts = [
        "Attention mechanisms have become a key component of sequence modeling.",
        "In this paper, we propose the Transformer, a model architecture based solely on attention.",
        "Experiments on two machine translation tasks show these models to be superior in quality.",
    ]
    return [
        Chunk(text=t, index=i, char_start=i * 100, char_end=(i + 1) * 100)
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client that returns predictable responses."""
    client = MagicMock()

    # Mock chat completion
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "title": "Attention Is All You Need",
        "authors": ["Vaswani", "Shazeer", "Parmar"],
        "abstract": "We propose the Transformer architecture.",
        "keywords": ["attention", "transformer", "NLP"],
        "methodology": "Multi-head self-attention mechanism",
        "key_results": "28.4 BLEU on WMT 2014 English-German",
        "contributions": ["Self-attention only architecture", "Multi-head attention"],
    })
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_completion

    # Mock embedding
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 1536
    mock_embed_response = MagicMock()
    mock_embed_response.data = [mock_embedding]
    client.embeddings.create.return_value = mock_embed_response

    return client


@pytest.fixture
def temp_dir():
    """Temporary directory for test output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Create a minimal PDF file for ingestion testing."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test PDF Document\nThis is a test document for the pipeline.")
        pdf_path = temp_dir / "test_doc.pdf"
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path
    except ImportError:
        pytest.skip("PyMuPDF not installed")


@pytest.fixture
def sample_text_path(temp_dir):
    """Create a temporary .txt file for text loader testing."""
    content = "# Test Document\n\nThis is a test markdown document.\n\nIt has multiple paragraphs."
    path = temp_dir / "test_doc.md"
    path.write_text(content)
    return path
