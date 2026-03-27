"""
GPT-4o function-calling tool definitions for the Document Intelligence Agent.

Each tool is defined as an OpenAI function schema and has a corresponding
Python implementation that the agent can invoke.
"""
from __future__ import annotations

import json
from typing import Any

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
]


# ─── Tool Executor ─────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes agent tool calls by routing to the appropriate handler.

    Each handler receives the parsed arguments and returns a JSON-serializable result.
    """

    def __init__(
        self,
        schema_extractor=None,
        relationship_extractor=None,
        chroma_store=None,
        neo4j_store=None,
        openai_client=None,
    ):
        self.schema_extractor = schema_extractor
        self.relationship_extractor = relationship_extractor
        self.chroma_store = chroma_store
        self.neo4j_store = neo4j_store
        self.openai_client = openai_client

        # Dispatch table
        self._handlers = {
            "extract_structured_data": self._handle_extract_structured,
            "embed_and_store_chunks": self._handle_embed_store,
            "extract_and_store_knowledge_graph": self._handle_graph,
            "generate_summary": self._handle_summary,
            "semantic_search": self._handle_search,
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

    def _handle_extract_structured(self, text: str, schema_type: str = "default") -> dict:
        if not self.schema_extractor:
            return {"error": "SchemaExtractor not initialized"}
        self.schema_extractor.schema_type = schema_type
        self.schema_extractor.schema = self.schema_extractor._load_schema(schema_type)
        extracted = self.schema_extractor.extract(text)
        return {"status": "ok", "schema_type": schema_type, "extracted_fields": len(extracted), "data": extracted}

    def _handle_embed_store(self, chunks: list[str], doc_id: str) -> dict:
        if not self.chroma_store:
            return {"error": "ChromaStore not initialized"}
        from src.utils.chunker import Chunk
        chunk_objs = [
            Chunk(text=t, index=i, char_start=0, char_end=len(t), metadata={"doc_id": doc_id})
            for i, t in enumerate(chunks)
        ]
        count = self.chroma_store.add_chunks(chunk_objs, doc_metadata={"doc_id": doc_id})
        return {"status": "ok", "chunks_stored": count, "collection": self.chroma_store.collection_name}

    def _handle_graph(self, text: str, doc_id: str) -> dict:
        if not self.relationship_extractor:
            return {"error": "RelationshipExtractor not initialized"}
        graph_data = self.relationship_extractor.extract(text, doc_id=doc_id)
        if self.neo4j_store:
            summary = self.neo4j_store.store_graph(
                graph_data["entities"], graph_data["relationships"], doc_id=doc_id
            )
        else:
            summary = {"warning": "Neo4jStore not initialized — graph data extracted but not persisted"}
        return {
            "status": "ok",
            "entities_extracted": len(graph_data["entities"]),
            "relationships_extracted": len(graph_data["relationships"]),
            "storage": summary,
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

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are an expert scientific document summarizer. Be concise and precise."},
                {"role": "user", "content": prompt},
            ],
        )
        summary = response.choices[0].message.content
        return {"status": "ok", "summary": summary}

    def _handle_search(self, query: str, top_k: int = 5) -> dict:
        if not self.chroma_store:
            return {"error": "ChromaStore not initialized"}
        results = self.chroma_store.search(query, top_k=top_k)
        return {"status": "ok", "query": query, "results": results}
