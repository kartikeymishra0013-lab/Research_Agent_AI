"""
GPT-4o Document Intelligence Agent.

Enhancements over baseline:
  - CostTracker integration — records token usage for every API call
  - SemanticChunker support — selectable via config chunking.strategy = "semantic"
  - Retry wrapper on the agent loop's main API call
  - Max-iterations safety cap remains at 20
"""
from __future__ import annotations

import json
from typing import Any, Optional

from openai import OpenAI

from src.agent.tools import TOOL_SCHEMAS, ToolExecutor
from src.ingestion.base_loader import Document
from src.utils.chunker import TextChunker
from src.utils.logger import get_logger
from src.utils.retry import with_retry

logger = get_logger(__name__)

MAX_ITERATIONS = 20


class DocumentAgent:
    """
    GPT-4o powered agent that processes documents through an autonomous tool-calling loop.

    Args:
        client        : Initialized OpenAI client.
        executor      : ToolExecutor with all storage backends wired.
        model         : OpenAI model (default: gpt-4o).
        temperature   : Model temperature (default: 0.1 for near-determinism).
        pipeline_mode : Controls which tools are invoked.
        chunking_cfg  : Optional chunking configuration dict
                        (keys: strategy, chunk_size, chunk_overlap, min_chunk_size,
                               similarity_threshold, sentence_group_size).
        cost_tracker  : Optional CostTracker for recording API costs.
    """

    def __init__(
        self,
        client: OpenAI,
        executor: ToolExecutor,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        pipeline_mode: str = "full",
        chunking_cfg: Optional[dict] = None,
        cost_tracker=None,
    ):
        self.client = client
        self.executor = executor
        self.model = model
        self.temperature = temperature
        self.pipeline_mode = pipeline_mode
        self.cost_tracker = cost_tracker
        self.chunker = self._build_chunker(client, chunking_cfg or {})

    def process(self, document: Document) -> dict[str, Any]:
        """
        Process a single document through the agentic tool-calling loop.

        Args:
            document: Loaded Document object from any ingestion loader.

        Returns:
            Result dict containing all agent outputs and cost summary.
        """
        logger.info(f"Agent processing: {document.source!r} ({document.doc_type})")

        doc_id = self._make_doc_id(document.source)
        chunks = self.chunker.chunk(
            document.content,
            metadata={"source": document.source, "doc_id": doc_id},
        )
        chunk_texts = [c.text for c in chunks]
        preview_text = document.content[:5000]

        logger.info(f"Document chunked into {len(chunks)} chunks")

        system_prompt = self._build_system_prompt(document, doc_id)
        user_message = self._build_user_message(document, doc_id, chunk_texts, preview_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        tool_results: dict[str, Any] = {}
        iterations = 0
        message = None

        # ── Agentic tool-calling loop ──────────────────────────────────────────
        while iterations < MAX_ITERATIONS:
            iterations += 1
            logger.debug(f"Agent iteration {iterations}")

            @with_retry(max_attempts=3, min_wait=1, max_wait=30)
            def _call_agent():
                return self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    messages=messages,
                )

            response = _call_agent()

            if self.cost_tracker:
                self.cost_tracker.record(
                    model=self.model,
                    operation=f"agent_loop_iter_{iterations}",
                    usage=response.usage,
                )

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            messages.append(message.model_dump(exclude_none=True))

            if finish_reason == "stop":
                logger.info(f"Agent completed in {iterations} iterations")
                break

            if finish_reason == "tool_calls" and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    logger.info(f"Calling tool: {tool_name}({list(args.keys())})")
                    result = self.executor.execute(tool_name, args)
                    tool_results[tool_name] = result

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str)[:4000],
                    })
            else:
                logger.warning(f"Unexpected finish_reason: {finish_reason}")
                break

        final_message = (
            message.content
            if message is not None and hasattr(message, "content") and message.content
            else ""
        )

        result: dict[str, Any] = {
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

        if self.cost_tracker:
            result["cost_summary"] = self.cost_tracker.summary()

        return result

    # ─── Prompt builders ──────────────────────────────────────────────────────

    def _build_system_prompt(self, doc: Document, doc_id: str) -> str:
        mode_instructions = {
            "full": (
                "You must call ALL of the following tools for this document:\n"
                "1. extract_structured_data — with the most appropriate schema\n"
                "2. evaluate_extraction — pass the source text and extraction result to score quality\n"
                "3. embed_and_store_chunks — pass ALL text chunks for semantic search\n"
                "4. extract_and_store_knowledge_graph — extract entities and relationships\n"
                "5. generate_summary — create a comprehensive summary\n"
                "After all tools complete, write a brief final status message."
            ),
            "extract_only": (
                "Call extract_structured_data, evaluate_extraction, and generate_summary only. "
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

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_chunker(client, cfg: dict):
        """Build the appropriate chunker from config."""
        strategy = cfg.get("strategy", "sentence")
        chunk_size = cfg.get("chunk_size", 1500)
        chunk_overlap = cfg.get("chunk_overlap", 200)
        min_chunk_size = cfg.get("min_chunk_size", 100)

        if strategy == "semantic":
            from src.utils.chunker import SemanticChunker
            return SemanticChunker(
                openai_client=client,
                similarity_threshold=cfg.get("similarity_threshold", 0.65),
                sentence_group_size=cfg.get("sentence_group_size", 3),
                max_chunk_size=chunk_size,
                min_chunk_size=min_chunk_size,
            )

        return TextChunker(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )

    @staticmethod
    def _make_doc_id(source: str) -> str:
        """Generate a short deterministic doc ID from the source path/URL."""
        import hashlib
        import re
        from pathlib import Path

        name = Path(source).stem if not source.startswith("http") else source
        hash_suffix = hashlib.md5(source.encode()).hexdigest()[:8]
        clean = re.sub(r"[^\w]", "_", name)[:30]
        return f"{clean}_{hash_suffix}"
