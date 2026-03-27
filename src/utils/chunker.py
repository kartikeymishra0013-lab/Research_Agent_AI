"""
Text chunking utilities — fixed, sentence-aware, paragraph, and semantic strategies.

TextChunker    : Fast, no-API chunking (fixed / sentence / paragraph)
SemanticChunker: Embedding-based chunking that splits on topic shifts
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A single text chunk with associated metadata."""
    text: str
    index: int
    char_start: int
    char_end: int
    token_estimate: int = field(init=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.token_estimate = len(self.text) // 4  # ~4 chars per token


# ═══════════════════════════════════════════════════════════════════════
# TEXT CHUNKER — no API calls needed
# ═══════════════════════════════════════════════════════════════════════

class TextChunker:
    """
    Flexible text chunker with three strategies.

    Strategies:
        sentence  : Accumulates sentences until size limit (default)
        paragraph : Splits on blank-line paragraph boundaries
        fixed     : Fixed character size with configurable overlap
    """

    def __init__(
        self,
        strategy: str = "sentence",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        _dispatch = {
            "fixed":     self._fixed_chunks,
            "sentence":  self._sentence_chunks,
            "paragraph": self._paragraph_chunks,
        }
        if strategy not in _dispatch:
            raise ValueError(
                f"Unknown chunking strategy: {strategy!r}. "
                f"Choose from {list(_dispatch)} or use SemanticChunker for 'semantic'."
            )
        self._chunker = _dispatch[strategy]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        text = text.strip()
        if not text:
            return []
        raw = self._chunker(text)
        result = []
        for i, (chunk_text, start, end) in enumerate(raw):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            result.append(Chunk(
                text=chunk_text.strip(),
                index=i,
                char_start=start,
                char_end=end,
                metadata=metadata or {},
            ))
        logger.debug(f"Chunked into {len(result)} chunks (strategy={self.strategy!r})")
        return result

    # ─── Strategies ───────────────────────────────────────────────

    def _fixed_chunks(self, text: str) -> list:
        chunks, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append((text[start:end], start, end))
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sentence_chunks(self, text: str) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current, current_start = [], "", 0
        char_pos = 0
        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) > self.chunk_size and current:
                end = current_start + len(current)
                chunks.append((current, current_start, end))
                overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current_start = end - len(overlap)
                current = (overlap + " " + sentence).strip()
            else:
                current = candidate
            char_pos += len(sentence) + 1
        if current:
            chunks.append((current, current_start, current_start + len(current)))
        return chunks

    def _paragraph_chunks(self, text: str) -> list:
        paragraphs = re.split(r'\n\s*\n', text)
        chunks, current, current_start, char_pos = [], "", 0, 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            candidate = (current + "\n\n" + para).strip() if current else para
            if len(candidate) > self.chunk_size and current:
                end = current_start + len(current)
                chunks.append((current, current_start, end))
                current_start = char_pos
                current = para
            else:
                current = candidate
            char_pos += len(para) + 2
        if current:
            chunks.append((current, current_start, current_start + len(current)))
        return chunks


# ═══════════════════════════════════════════════════════════════════════
# SEMANTIC CHUNKER — uses OpenAI embeddings to detect topic shifts
# ═══════════════════════════════════════════════════════════════════════

class SemanticChunker:
    """
    Embedding-based semantic chunking.

    Algorithm:
        1. Split text into sentences
        2. Group sentences into small windows
        3. Embed each window via OpenAI API
        4. Compute cosine similarity between adjacent windows
        5. Split wherever similarity drops below threshold
        6. Merge undersized chunks with their neighbour

    This produces chunks that are topically coherent, improving
    retrieval quality vs. fixed/sentence splitting.

    Args:
        openai_client       : Initialized OpenAI client.
        embedding_model     : OpenAI embedding model name.
        similarity_threshold: Cosine similarity below which a split is inserted (0–1).
        sentence_group_size : Number of sentences per embedding window.
        max_chunk_size      : Hard cap on characters per chunk.
        min_chunk_size      : Minimum characters — smaller chunks merged.
        cost_tracker        : Optional CostTracker to record embedding cost.
    """

    def __init__(
        self,
        openai_client,
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.65,
        sentence_group_size: int = 3,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 150,
        cost_tracker=None,
    ):
        self.client = openai_client
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.sentence_group_size = sentence_group_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.cost_tracker = cost_tracker

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        text = text.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= self.sentence_group_size:
            # Not enough sentences to embed — fall back to one chunk
            return [Chunk(text=text, index=0, char_start=0, char_end=len(text), metadata=metadata or {})]

        # Group sentences into windows
        groups = [
            " ".join(sentences[i: i + self.sentence_group_size])
            for i in range(0, len(sentences), self.sentence_group_size)
        ]

        # Embed all groups in a single batch call
        embeddings = self._embed_batch(groups)

        # Detect split points where cosine similarity < threshold
        split_indices = {0}  # always start a new chunk at 0
        for i in range(1, len(embeddings)):
            sim = self._cosine(embeddings[i - 1], embeddings[i])
            if sim < self.similarity_threshold:
                split_indices.add(i)

        # Build chunk text spans
        raw_chunks: list[str] = []
        current_groups: list[str] = []
        for i, group in enumerate(groups):
            if i in split_indices and current_groups:
                raw_chunks.append(" ".join(current_groups))
                current_groups = []
            current_groups.append(group)
        if current_groups:
            raw_chunks.append(" ".join(current_groups))

        # Enforce max_chunk_size by splitting oversized chunks further
        final_texts: list[str] = []
        for c in raw_chunks:
            if len(c) <= self.max_chunk_size:
                final_texts.append(c)
            else:
                # Split oversized chunks by sentences
                fallback = TextChunker(
                    strategy="sentence",
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=100,
                    min_chunk_size=self.min_chunk_size,
                )
                for sub in fallback.chunk(c):
                    final_texts.append(sub.text)

        # Build Chunk objects, filtering undersized
        result: list[Chunk] = []
        char_pos = 0
        for i, chunk_text in enumerate(final_texts):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                result.append(Chunk(
                    text=chunk_text.strip(),
                    index=i,
                    char_start=char_pos,
                    char_end=char_pos + len(chunk_text),
                    metadata=metadata or {},
                ))
            char_pos += len(chunk_text) + 1

        logger.info(f"SemanticChunker: {len(sentences)} sentences → {len(result)} chunks "
                    f"(threshold={self.similarity_threshold})")
        return result

    # ─── Helpers ──────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        from src.utils.retry import embedding_retry

        @embedding_retry
        def _call():
            return self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )

        response = _call()

        if self.cost_tracker:
            self.cost_tracker.record(
                model=self.embedding_model,
                operation="semantic_chunking",
                usage=response.usage,
            )

        return [item.embedding for item in response.data]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
