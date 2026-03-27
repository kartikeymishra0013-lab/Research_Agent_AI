"""
GPT-4o function-calling tool definitions for the Document Intelligence Agent.

Enhancements over baseline:
  - evaluate_extraction tool — GPT-4o rates confidence of extracted fields
  - CostTracker integrated into all tool handlers
  - Summary handler uses retry
"""
from __future__ import annotations

import json
from typing import Any, Optional

from src.utils.retry import with_retry

# ─── Tool Schemas (passed to OpenAI API) ──────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "extract_structured_data",
            "description": (
                "Extract structured fields from document text according to a schema. "
                "Use this to convert raw text into a structured JSON record with typed fields "
                "like title, authors, methods, results, keywords, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The document text or chunk to extract data from.",
                    },
                    "schema_type": {
                        "type": "string",
                        "enum": ["research_paper", "patent", "default"],
                        "description": "The schema to use for extraction. Choose based on document type.",
                    },
                },
                "required": ["text", "schema_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "embed_and_store_chunks",
            "description": (
                "Embed document text chunks and store them in ChromaDB for semantic search. "
                "Always use this to enable future document retrieval and similarity search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of text chunks to embed and store.",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Unique identifier for the source document.",
                    },
                },
                "required": ["chunks", "doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_and_store_knowledge_graph",
            "description": (
                "Extract named entities (people, organizations, technologies, concepts) "
                "and their relationships from text, then store in the Neo4j knowledge graph. "
                "Use this for documents with rich factual content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to extract entities and relationships from.",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Document identifier for provenance tracking.",
                    },
                },
                "required": ["text", "doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_summary",
            "description": (
                "Generate a structured summary of the document including: "
                "main topic, key contributions, methodology, findings, and implications. "
                "Always call this as part of processing each document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Full document text or representative sections to summarize.",
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Type of document (pdf, docx, url, etc.)",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Search the ChromaDB vector store for document chunks similar to a query. "
                "Use this to find relevant context or verify extracted information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5).",
                        "default": 5,
                    },
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
                "Score the quality and confidence of a previous extraction result. "
                "Returns a per-field confidence score (0–1) and an overall quality verdict. "
                "Call this after extract_structured_data to validate key fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_text": {
                        "type": "string",
                        "description": "The original source text that was extracted from.",
                    },
                    "extracted_data": {
                        "type": "object",
                        "description": "The JSON object returned by extract_structured_data.",
                    },
                },
                "required": ["source_text", "extracted_data"],
            },
        },
    },
]


# ─── Tool Executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes agent tool calls by routing to the appropriate handler.

    Each handler receives the parsed arguments and returns a JSON-serializable result.

    Args:
        schema_extractor       : SchemaExtractor instance (optional).
        relationship_extractor : RelationshipExtractor instance (optional).
        chroma_store           : ChromaStore instance (optional).
        neo4j_store            : Neo4jStore instance (optional).
        openai_client          : Initialized OpenAI client (optional).
        cost_tracker           : CostTracker for recording API costs (optional).
    """

    def __init__(
        self,
        schema_extractor=None,
        relationship_extractor=None,
        chroma_store=None,
        neo4j_store=None,
        openai_client=None,
        cost_tracker=None,
    ):
        self.schema_extractor = schema_extractor
        self.relationship_extractor = relationship_extractor
        self.chroma_store = chroma_store
        self.neo4j_store = neo4j_store
        self.openai_client = openai_client
        self.cost_tracker = cost_tracker

        self._handlers = {
            "extract_structured_data":        self._handle_extract_structured,
            "embed_and_store_chunks":         self._handle_embed_store,
            "extract_and_store_knowledge_graph": self._handle_graph,
            "generate_summary":               self._handle_summary,
            "semantic_search":                self._handle_search,
            "evaluate_extraction":            self._handle_evaluate,
        }

    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Route and execute a tool call."""
        handler = self._handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(**arguments)
        except Exception as e:
            return {"error": str(e), "tool": tool_name}

    # ─── Handlers ─────────────────────────────────────────────────────────────

    def _handle_extract_structured(self, text: str, schema_type: str = "default") -> dict:
        if not self.schema_extractor:
            return {"error": "SchemaExtractor not initialized"}
        self.schema_extractor.schema_type = schema_type
        self.schema_extractor.schema = self.schema_extractor._load_schema(schema_type)
        extracted = self.schema_extractor.extract(text)
        return {
            "status": "ok",
            "schema_type": schema_type,
            "extracted_fields": len(extracted),
            "data": extracted,
        }

    def _handle_embed_store(self, chunks: list[str], doc_id: str) -> dict:
        if not self.chroma_store:
            return {"error": "ChromaStore not initialized"}
        from src.utils.chunker import Chunk
        chunk_objs = [
            Chunk(text=t, index=i, char_start=0, char_end=len(t), metadata={"doc_id": doc_id})
            for i, t in enumerate(chunks)
        ]
        count = self.chroma_store.add_chunks(chunk_objs, doc_metadata={"doc_id": doc_id})
        return {
            "status": "ok",
            "chunks_stored": count,
            "collection": self.chroma_store.collection_name,
        }

    def _handle_graph(self, text: str, doc_id: str) -> dict:
        if not self.relationship_extractor:
            return {"error": "RelationshipExtractor not initialized"}
        graph_data = self.relationship_extractor.extract(text, doc_id=doc_id)
        if self.neo4j_store:
            storage_summary = self.neo4j_store.store_graph(
                graph_data["entities"], graph_data["relationships"], doc_id=doc_id
            )
        else:
            storage_summary = {
                "warning": "Neo4jStore not initialized — graph data extracted but not persisted"
            }
        return {
            "status": "ok",
            "entities_extracted": len(graph_data["entities"]),
            "relationships_extracted": len(graph_data["relationships"]),
            "storage": storage_summary,
            "graph_data": graph_data,
        }

    def _handle_summary(self, text: str, doc_type: str = "") -> dict:
        if not self.openai_client:
            return {"error": "OpenAI client not initialized"}

        prompt = f"""Provide a comprehensive structured summary of this {doc_type} document.

Include these sections:
1. **Main Topic**: One sentence overview
2. **Key Contributions**: Bullet list of main contributions
3. **Methodology**: How was the work done?
4. **Key Findings / Results**: What were the outcomes?
5. **Implications**: Why does this matter?
6. **Keywords**: 5-10 relevant keywords

Document text:
{text[:6000]}"""

        @with_retry(max_attempts=3, min_wait=1, max_wait=30)
        def _call():
            return self.openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert scientific document summarizer. Be concise and precise.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

        response = _call()

        if self.cost_tracker:
            self.cost_tracker.record(
                model="gpt-4o",
                operation="document_summary",
                usage=response.usage,
            )

        summary = response.choices[0].message.content
        return {"status": "ok", "summary": summary}

    def _handle_search(self, query: str, top_k: int = 5) -> dict:
        if not self.chroma_store:
            return {"error": "ChromaStore not initialized"}
        results = self.chroma_store.search(query, top_k=top_k)
        return {"status": "ok", "query": query, "results": results}

    def _handle_evaluate(self, source_text: str, extracted_data: dict) -> dict:
        """
        Score each extracted field for groundedness in the source text.
        Returns per-field confidence scores and an overall quality verdict.
        """
        if not self.openai_client:
            return {"error": "OpenAI client not initialized"}

        # Strip internal metadata keys from evaluation
        fields_to_eval = {
            k: v for k, v in extracted_data.items()
            if not k.startswith("_") and k not in ("source",)
        }

        if not fields_to_eval:
            return {"status": "ok", "confidence": {}, "overall_quality": "no_fields"}

        prompt = f"""You are a quality-assurance agent for document extraction.

Rate each extracted field with a confidence score (0.0–1.0):
  1.0 = directly and clearly stated in source text
  0.5 = implied or partially supported
  0.0 = not found / likely hallucinated

Also provide an "overall_quality" verdict: "high" (avg > 0.7), "medium" (0.4–0.7), or "low" (< 0.4).

Return ONLY a JSON object:
{{
  "confidence": {{"field_name": 0.95, ...}},
  "overall_quality": "high" | "medium" | "low",
  "low_confidence_fields": ["field_name", ...]
}}

SOURCE TEXT (first 2000 chars):
{source_text[:2000]}

EXTRACTED DATA:
{json.dumps(fields_to_eval, indent=2, default=str)[:3000]}"""

        @with_retry(max_attempts=2, min_wait=1, max_wait=20)
        def _call():
            return self.openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )

        response = _call()

        if self.cost_tracker:
            self.cost_tracker.record(
                model="gpt-4o",
                operation="evaluate_extraction",
                usage=response.usage,
            )

        try:
            eval_result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            eval_result = {"confidence": {}, "overall_quality": "unknown"}

        return {"status": "ok", **eval_result}
