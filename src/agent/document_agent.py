"""
GPT-4o Document Intelligence Agent.

An agentic loop that autonomously decides which tools to call
to process a document — extraction, embedding, graph construction, and summarization.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from openai import OpenAI

from src.agent.tools import TOOL_SCHEMAS, ToolExecutor
from src.ingestion.base_loader import Document
from src.utils.chunker import TextChunker
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_ITERATIONS = 20  # Safety cap on agentic loop


class DocumentAgent:
    """
    GPT-4o powered agent that processes documents through an autonomous tool-calling loop.

    The agent is given a document and pipeline configuration, then decides which
    tools to invoke (extraction, embedding, graph, summary) and in what order.

    Args:
        client      : Initialized OpenAI client.
        executor    : ToolExecutor with all storage backends connected.
        model       : OpenAI model to use (default: gpt-4o).
        temperature : Model temperature (default: 0.1 for determinism).
        pipeline_mode: Controls which capabilities to enable.
    """

    def __init__(
        self,
        client: OpenAI,
        executor: ToolExecutor,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        pipeline_mode: str = "full",
    ):
        self.client = client
        self.executor = executor
        self.model = model
        self.temperature = temperature
        self.pipeline_mode = pipeline_mode
        self.chunker = TextChunker(strategy="sentence", chunk_size=1500, chunk_overlap=200)

    def process(self, document: Document) -> dict[str, Any]:
        """
        Process a single document through the agent loop.

        Args:
            document: Loaded Document object from any ingestion loader.

        Returns:
            A result dict containing all agent outputs.
        """
        logger.info(f"Agent processing document: {document.source!r} ({document.doc_type})")

        # Prepare context
        doc_id = self._make_doc_id(document.source)
        chunks = self.chunker.chunk(document.content, metadata={"source": document.source, "doc_id": doc_id})
        chunk_texts = [c.text for c in chunks]
        preview_text = document.content[:5000]

        logger.info(f"Document chunked into {len(chunks)} chunks")

        system_prompt = self._build_system_prompt(document, doc_id)
        user_message = self._build_user_message(document, doc_id, chunk_texts, preview_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        tool_results = {}
        iterations = 0

        # Agentic tool-calling loop
        while iterations < MAX_ITERATIONS:
            iterations += 1
            logger.debug(f"Agent iteration {iterations}")

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                messages=messages,
            )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Append assistant message to history
            messages.append(message.model_dump(exclude_none=True))

            if finish_reason == "stop":
                # Agent is done
                logger.info(f"Agent completed in {iterations} iterations")
                break

            if finish_reason == "tool_calls" and message.tool_calls:
                # Execute all requested tool calls
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    logger.info(f"Calling tool: {tool_name}({list(args.keys())})")
                    result = self.executor.execute(tool_name, args)
                    tool_results[tool_name] = result

                    # Append tool result to message history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str)[:4000],
                    })
            else:
                # Unexpected finish reason
                logger.warning(f"Unexpected finish_reason: {finish_reason}")
                break

        final_message = message.content if hasattr(message, "content") and message.content else ""

        return {
            "doc_id": doc_id,
            "source": document.source,
            "doc_type": document.doc_type,
            "metadata": document.metadata,
            "word_count": document.word_count,
            "chunk_count": len(chunks),
            "iterations": iterations,
            "tool_results": tool_results,
            "agent_summary": final_message,
            "tools_called": list(tool_results.keys()),
        }

    def _build_system_prompt(self, doc: Document, doc_id: str) -> str:
        mode_instructions = {
            "full": (
                "You must call ALL of the following tools for this document:\n"
                "1. extract_structured_data — with the most appropriate schema\n"
                "2. embed_and_store_chunks — pass ALL text chunks for semantic search\n"
                "3. extract_and_store_knowledge_graph — extract entities and relationships\n"
                "4. generate_summary — create a comprehensive summary\n"
                "After all tools complete, write a brief final status message."
            ),
            "extract_only": (
                "Call extract_structured_data and generate_summary only. "
                "Do NOT call embedding or graph tools."
            ),
            "search_only": (
                "Call embed_and_store_chunks to index the document for search. "
                "Do NOT call extraction or graph tools."
            ),
            "graph_only": (
                "Call extract_and_store_knowledge_graph to build the knowledge graph. "
                "Do NOT call other tools."
            ),
        }
        instructions = mode_instructions.get(self.pipeline_mode, mode_instructions["full"])

        return f"""You are a Scientific Document Intelligence Agent powered by GPT-4o.

Your job is to process scientific documents and extract structured knowledge using the provided tools.

Document being processed:
- ID: {doc_id}
- Type: {doc.doc_type}
- Title: {doc.metadata.get("title", "Unknown")}
- Size: {doc.word_count} words

Pipeline mode: {self.pipeline_mode.upper()}
{instructions}

Important guidelines:
- Choose schema_type based on document content (research_paper for academic docs, patent for patents)
- Pass the full chunk list to embed_and_store_chunks for comprehensive coverage
- Be systematic — don't skip tools unless the pipeline mode restricts it
- After completing all tool calls, provide a brief completion message
"""

    def _build_user_message(
        self, doc: Document, doc_id: str, chunk_texts: list[str], preview: str
    ) -> str:
        chunks_preview = "\n---\n".join(chunk_texts[:3])
        return f"""Process this document now.

Document ID: {doc_id}
Source: {doc.source}
Total chunks: {len(chunk_texts)}

Document preview (first 5000 chars):
{preview}

First 3 chunks for context:
{chunks_preview}

All chunks to embed: {json.dumps(chunk_texts)}

Begin processing — call all required tools in sequence."""

    @staticmethod
    def _make_doc_id(source: str) -> str:
        """Generate a short deterministic doc ID from the source path/URL."""
        import hashlib
        from pathlib import Path
        name = Path(source).stem if not source.startswith("http") else source
        hash_suffix = hashlib.md5(source.encode()).hexdigest()[:8]
        import re
        clean = re.sub(r"[^\w]", "_", name)[:30]
        return f"{clean}_{hash_suffix}"
