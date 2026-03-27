"""
GPT-4o function-calling tool definitions for the Document Intelligence Agent.
Upgrades: evaluate_extraction tool, CostTracker in all handlers, retry on summary.
"""
from __future__ import annotations

import json
from typing import Any

from src.utils.retry import with_retry

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "extract_structured_data",
            "description": "Extract structured fields from document text according to a schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":        {"type": "string", "description": "Document text to extract from."},
                    "schema_type": {"type": "string", "enum": ["research_paper", "patent", "default"],
                                   "description": "Schema type to use."},
                },
                "required": ["text", "schema_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "embed_and_store_chunks",
            "description": "Embed text chunks and store in ChromaDB for semantic search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunks": {"type": "array", "items": {"type": "string"}, "description": "Text chunks to embed."},
                    "doc_id": {"type": "string", "description": "Document identifier."},
                },
                "required": ["chunks", "doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_and_store_knowledge_graph",
            "description": "Extract entities and relationships from text and store in Neo4j.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":   {"type": "string", "description": "Text to extract from."},
                    "doc_id": {"type": "string", "description": "Document identifier."},
                },
                "required": ["text", "doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_summary",
            "description": "Generate a structured summary of the document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":     {"type": "string", "description": "Document text to summarize."},
                    "doc_type": {"type": "string", "description": "Document type (pdf, txt, etc.)"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search ChromaDB for chunks similar to a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {"type": "integer", "description": "Results to return (default 5).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_extraction",
            "description": (
                "Score confidence of a previous extraction result (0-1 per field). "
                "Call after extract_structured_data to validate quality."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_text":    {"type": "string", "description": "Original source text."},
                    "extracted_data": {"type": "object", "description": "Result from extract_structured_data."},
                },
                "required": ["source_text", "extracted_data"],
            },
        },
    },
]


class ToolExecutor:
    def __init__(self, schema_extractor=None, relationship_extractor=None,
                 chroma_store=None, neo4j_store=None, openai_client=None, cost_tracker=None):
        self.schema_extractor = schema_extractor
        self.relationship_extractor = relationship_extractor
        self.chroma_store = chroma_store
        self.neo4j_store = neo4j_store
        self.openai_client = openai_client
        self.cost_tracker = cost_tracker
        self._handlers = {
            "extract_structured_data":           self._handle_extract_structured,
            "embed_and_store_chunks":            self._handle_embed_store,
            "extract_and_store_knowledge_graph": self._handle_graph,
            "generate_summary":                  self._handle_summary,
            "semantic_search":                   self._handle_search,
            "evaluate_extraction":               self._handle_evaluate,
        }

    def execute(self, tool_name: str, arguments: dict) -> dict:
        handler = self._handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(**arguments)
        except Exception as e:
            return {"error": str(e), "tool": tool_name}

    def _handle_extract_structured(self, text: str, schema_type: str = "default") -> dict:
        if not self.schema_extractor:
            return {"error": "SchemaExtractor not initialized"}
        self.schema_extractor.schema_type = schema_type
        self.schema_extractor.schema = self.schema_extractor._load_schema(schema_type)
        extracted = self.schema_extractor.extract(text)
        return {"status": "ok", "schema_type": schema_type,
                "extracted_fields": len(extracted), "data": extracted}

    def _handle_embed_store(self, chunks: list[str], doc_id: str) -> dict:
        if not self.chroma_store:
            return {"error": "ChromaStore not initialized"}
        from src.utils.chunker import Chunk
        objs = [Chunk(text=t, index=i, char_start=0, char_end=len(t), metadata={"doc_id": doc_id})
                for i, t in enumerate(chunks)]
        count = self.chroma_store.add_chunks(objs, doc_metadata={"doc_id": doc_id})
        return {"status": "ok", "chunks_stored": count, "collection": self.chroma_store.collection_name}

    def _handle_graph(self, text: str, doc_id: str) -> dict:
        if not self.relationship_extractor:
            return {"error": "RelationshipExtractor not initialized"}
        graph_data = self.relationship_extractor.extract(text, doc_id=doc_id)
        storage = (self.neo4j_store.store_graph(graph_data["entities"], graph_data["relationships"], doc_id=doc_id)
                   if self.neo4j_store else {"warning": "Neo4jStore not initialized"})
        return {"status": "ok", "entities_extracted": len(graph_data["entities"]),
                "relationships_extracted": len(graph_data["relationships"]),
                "storage": storage, "graph_data": graph_data}

    def _handle_summary(self, text: str, doc_type: str = "") -> dict:
        if not self.openai_client:
            return {"error": "OpenAI client not initialized"}
        prompt = (f"Summarize this {doc_type} document. Include:\n"
                  "1. Main Topic  2. Key Contributions  3. Methodology  4. Findings  5. Keywords\n\n"
                  f"Text:\n{text[:6000]}")

        @with_retry(max_attempts=3, min_wait=1, max_wait=30)
        def _call():
            return self.openai_client.chat.completions.create(
                model="gpt-4o", temperature=0.3,
                messages=[{"role": "system", "content": "You are an expert scientific document summarizer."},
                          {"role": "user", "content": prompt}],
            )

        response = _call()
        if self.cost_tracker:
            self.cost_tracker.record("gpt-4o", "document_summary", response.usage)
        return {"status": "ok", "summary": response.choices[0].message.content}

    def _handle_search(self, query: str, top_k: int = 5) -> dict:
        if not self.chroma_store:
            return {"error": "ChromaStore not initialized"}
        return {"status": "ok", "query": query, "results": self.chroma_store.search(query, top_k=top_k)}

    def _handle_evaluate(self, source_text: str, extracted_data: dict) -> dict:
        if not self.openai_client:
            return {"error": "OpenAI client not initialized"}
        fields = {k: v for k, v in extracted_data.items() if not k.startswith("_") and k != "source"}
        if not fields:
            return {"status": "ok", "confidence": {}, "overall_quality": "no_fields"}
        prompt = (
            "Rate each extracted field 0.0-1.0 and give overall_quality (high/medium/low).\n"
            'Return ONLY JSON: {"confidence": {"field": score}, "overall_quality": "high", "low_confidence_fields": []}\n\n'
            f"SOURCE:\n{source_text[:2000]}\n\nEXTRACTED:\n{json.dumps(fields, indent=2, default=str)[:2000]}"
        )

        @with_retry(max_attempts=2, min_wait=1, max_wait=20)
        def _call():
            return self.openai_client.chat.completions.create(
                model="gpt-4o", temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )

        response = _call()
        if self.cost_tracker:
            self.cost_tracker.record("gpt-4o", "evaluate_extraction", response.usage)
        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            result = {"confidence": {}, "overall_quality": "unknown"}
        return {"status": "ok", **result}
