"""
Tests for the storage layer — JSON store and chunker.
"""
import json
import pytest
from pathlib import Path

from src.storage.json_store import JsonStore
from src.utils.chunker import TextChunker, Chunk


class TestJsonStore:
    def test_save_extraction(self, temp_dir):
        store = JsonStore(output_dir=str(temp_dir))
        extracted = {"title": "Test Paper", "authors": ["Alice", "Bob"], "keywords": ["NLP"]}
        path = store.save_extraction(extracted, source="/data/test.pdf", schema_type="research_paper")

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["title"] == "Test Paper"
        assert "_meta" in data
        assert data["_meta"]["schema_type"] == "research_paper"

    def test_append_to_dataset(self, temp_dir):
        store = JsonStore(output_dir=str(temp_dir))
        records = [
            {"doc_id": "doc1", "title": "Paper 1"},
            {"doc_id": "doc2", "title": "Paper 2"},
        ]
        for r in records:
            store.append_to_dataset(r, dataset_name="test_dataset")

        loaded = store.load_dataset("test_dataset")
        assert len(loaded) == 2
        assert loaded[0]["doc_id"] == "doc1"
        assert loaded[1]["title"] == "Paper 2"

    def test_save_summary_report(self, temp_dir):
        store = JsonStore(output_dir=str(temp_dir))
        path = store.save_summary_report(
            summary="This paper proposes the Transformer architecture.",
            source="/data/test.pdf",
            metadata={"title": "Attention Is All You Need"},
        )
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "generated_at" in data

    def test_save_graph_data(self, temp_dir):
        store = JsonStore(output_dir=str(temp_dir))
        graph_data = {
            "entities": [{"id": "bert", "label": "BERT", "type": "Technology"}],
            "relationships": [],
        }
        path = store.save_graph_data(graph_data, source="/data/test.pdf")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data["entities"]) == 1

    def test_list_outputs(self, temp_dir):
        store = JsonStore(output_dir=str(temp_dir))
        store.save_extraction({"title": "Paper"}, source="test.pdf")
        store.save_summary_report("Summary text", source="test.pdf")
        outputs = store.list_outputs()
        assert len(outputs) == 2

    def test_load_empty_dataset(self, temp_dir):
        store = JsonStore(output_dir=str(temp_dir))
        result = store.load_dataset("nonexistent")
        assert result == []


class TestTextChunker:
    def test_fixed_chunking(self):
        text = "A" * 5000
        chunker = TextChunker(strategy="fixed", chunk_size=1000, chunk_overlap=100)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 1000

    def test_sentence_chunking(self):
        text = (
            "The Transformer model uses self-attention. "
            "It was proposed by Vaswani et al. in 2017. "
            "The architecture has an encoder and decoder. "
            "Multi-head attention allows parallel computation. "
            "The model achieved state-of-the-art results on translation tasks."
        )
        chunker = TextChunker(strategy="sentence", chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_paragraph_chunking(self):
        text = "First paragraph.\n\nSecond paragraph with more content.\n\nThird paragraph here."
        chunker = TextChunker(strategy="paragraph", chunk_size=200, chunk_overlap=0, min_chunk_size=0)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_empty_text_returns_empty(self):
        chunker = TextChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_metadata_preserved(self):
        text = "Test sentence for chunking with metadata."
        chunker = TextChunker(strategy="fixed", chunk_size=200, chunk_overlap=0, min_chunk_size=0)
        meta = {"source": "test.pdf", "doc_id": "doc_001"}
        chunks = chunker.chunk(text, metadata=meta)
        assert chunks[0].metadata["source"] == "test.pdf"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            TextChunker(strategy="invalid_strategy")

    def test_min_chunk_size_filters(self):
        text = "Short. " + "A" * 300 + ". End."
        chunker = TextChunker(strategy="fixed", chunk_size=400, chunk_overlap=0, min_chunk_size=200)
        chunks = chunker.chunk(text)
        for c in chunks:
            assert len(c.text) >= 200

    def test_chunk_token_estimate(self):
        text = "A" * 400  # ~100 tokens
        chunker = TextChunker(strategy="fixed", chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(text)
        assert chunks[0].token_estimate == 100
