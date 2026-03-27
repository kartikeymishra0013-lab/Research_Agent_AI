"""
ChromaDB vector store for semantic search over document chunks.
Supports embedding, upsert, and similarity search operations.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaStore:
    """
    Manages ChromaDB collections for document chunk storage and retrieval.

    Supports:
        - Embedding generation via OpenAI text-embedding-3-small
        - Chunk upsert with metadata
        - Semantic similarity search
        - Collection management (create, delete, list)
    """

    def __init__(
        self,
        host: str = "chromadb",
        port: int = 8000,
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
        openai_client=None,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.openai_client = openai_client
        self._client = None
        self._collection = None

    def connect(self):
        """Establish connection to ChromaDB."""
        import chromadb
        self._client = chromadb.HttpClient(host=self.host, port=self.port)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Connected to ChromaDB at {self.host}:{self.port}, collection={self.collection_name!r}")

    @property
    def collection(self):
        if self._collection is None:
            self.connect()
        return self._collection

    def add_chunks(self, chunks: list, doc_metadata: Optional[dict] = None) -> int:
        """
        Embed and store a list of Chunk objects.

        Args:
            chunks      : List of Chunk objects from TextChunker.
            doc_metadata: Optional document-level metadata to attach to each chunk.

        Returns:
            Number of chunks stored.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embed_texts(texts)

        ids = []
        documents = []
        metadatas = []
        embeds = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = self._make_id(chunk.text)
            meta = {
                "chunk_index": chunk.index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_estimate": chunk.token_estimate,
                **(doc_metadata or {}),
                **{k: str(v) for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
            }
            ids.append(chunk_id)
            documents.append(chunk.text)
            metadatas.append(meta)
            embeds.append(embedding)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeds,
        )
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB collection {self.collection_name!r}")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search over stored chunks.

        Args:
            query  : Natural language query string.
            top_k  : Number of results to return.
            where  : Optional metadata filter dict.

        Returns:
            List of result dicts with keys: id, text, metadata, distance, score.
        """
        query_embedding = self._embed_texts([query])[0]

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            output.append({
                "id": results["ids"][0][i],
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "score": round(1 - dist, 4),  # cosine similarity score
            })

        logger.info(f"Search returned {len(output)} results for query={query!r}")
        return output

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self.collection.count()

    def delete_collection(self):
        """Delete and recreate the collection (destructive)."""
        self._client.delete_collection(self.collection_name)
        self._collection = None
        logger.warning(f"Deleted collection {self.collection_name!r}")

    def list_collections(self) -> list[str]:
        """List all collection names in ChromaDB."""
        return [c.name for c in self._client.list_collections()]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI embedding model."""
        if self.openai_client is None:
            raise RuntimeError("OpenAI client required for embedding generation")

        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _make_id(text: str) -> str:
        """Generate a deterministic ID for a chunk based on its content."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
