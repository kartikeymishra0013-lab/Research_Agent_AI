"""
ChromaDB vector store for semantic search over document chunks.
Upgrades: embedding_retry on all embedding calls, CostTracker integration.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from src.utils.logger import get_logger
from src.utils.retry import embedding_retry

logger = get_logger(__name__)


class ChromaStore:
    def __init__(
        self,
        host: str = "chromadb",
        port: int = 8000,
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
        openai_client=None,
        cost_tracker=None,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.openai_client = openai_client
        self.cost_tracker = cost_tracker
        self._client = None
        self._collection = None

    def connect(self):
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
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self._embed_texts(texts)
        ids, documents, metadatas, embeds = [], [], [], []
        for chunk, embedding in zip(chunks, embeddings):
            meta = {
                "chunk_index": chunk.index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_estimate": chunk.token_estimate,
                **(doc_metadata or {}),
                **{k: str(v) for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
            }
            ids.append(self._make_id(chunk.text))
            documents.append(chunk.text)
            metadatas.append(meta)
            embeds.append(embedding)
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeds)
        logger.info(f"Stored {len(chunks)} chunks in ChromaDB collection {self.collection_name!r}")
        return len(chunks)

    def search(self, query: str, top_k: int = 10, where: Optional[dict] = None) -> list[dict[str, Any]]:
        query_embedding = self._embed_texts([query])[0]
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        return [
            {
                "id": results["ids"][0][i],
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "score": round(1 - dist, 4),
            }
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ))
        ]

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self):
        self._client.delete_collection(self.collection_name)
        self._collection = None
        logger.warning(f"Deleted collection {self.collection_name!r}")

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.openai_client is None:
            raise RuntimeError("OpenAI client required for embedding generation")

        @embedding_retry
        def _call():
            return self.openai_client.embeddings.create(model=self.embedding_model, input=texts)

        response = _call()
        if self.cost_tracker:
            self.cost_tracker.record(self.embedding_model, "chroma_embedding", response.usage)
        return [item.embedding for item in response.data]

    @staticmethod
    def _make_id(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
