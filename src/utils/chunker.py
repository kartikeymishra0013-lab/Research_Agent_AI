"""
Text chunking utilities with multiple strategies.
Supports fixed-size, sentence-aware, and semantic chunking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

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
        # Rough token estimate: ~4 chars per token
        self.token_estimate = len(self.text) // 4


class TextChunker:
    """
    Flexible text chunker supporting multiple strategies.

    Strategies:
        - "fixed"     : Fixed character size with overlap
        - "sentence"  : Sentence-boundary-aware chunks
        - "paragraph" : Paragraph-boundary chunks
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
            "fixed": self._fixed_chunks,
            "sentence": self._sentence_chunks,
            "paragraph": self._paragraph_chunks,
        }
        if strategy not in _dispatch:
            raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose from {list(_dispatch)}")
        self._chunker = _dispatch[strategy]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Chunk text using the configured strategy."""
        text = text.strip()
        if not text:
            return []

        raw_chunks = self._chunker(text)
        result = []
        for i, (chunk_text, start, end) in enumerate(raw_chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            result.append(Chunk(
                text=chunk_text.strip(),
                index=i,
                char_start=start,
                char_end=end,
                metadata=metadata or {},
            ))

        logger.debug(f"Chunked text into {len(result)} chunks using strategy={self.strategy!r}")
        return result

    # ─── Strategies ───────────────────────────────────────────────

    def _fixed_chunks(self, text: str) -> List[tuple]:
        """Fixed-size chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append((text[start:end], start, end))
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sentence_chunks(self, text: str) -> List[tuple]:
        """Sentence-aware chunking — accumulates sentences until size limit."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        current_start = 0
        char_pos = 0

        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            if len(candidate) > self.chunk_size and current:
                end = current_start + len(current)
                chunks.append((current, current_start, end))
                # overlap: keep last `chunk_overlap` chars
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current_start = end - len(overlap_text)
                current = (overlap_text + " " + sentence).strip()
            else:
                current = candidate
            char_pos += len(sentence) + 1

        if current:
            chunks.append((current, current_start, current_start + len(current)))
        return chunks

    def _paragraph_chunks(self, text: str) -> List[tuple]:
        """Paragraph-boundary chunking."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current = ""
        current_start = 0
        char_pos = 0

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
