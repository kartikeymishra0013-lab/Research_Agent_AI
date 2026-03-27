"""
GPT-4o Document Intelligence Agent.
Upgrades: CostTracker integration, SemanticChunker support, retry on agent loop calls.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from src.agent.tools import TOOL_SCHEMAS, ToolExecutor
from src.ingestion.base_loader import Document
from src.utils.chunker import TextChunker
from src.utils.logger import get_logger
from src.utils.retry import with_retry

logger = get_logger(__name__)
MAX_ITERATIONS = 20


class DocumentAgent:
    def __init__(
        self,
        client,
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
        logger.info(f"Agent processing: {document.source!r} ({document.doc_type})")
        doc_id = self._make_doc_id(document.source)
        chunks = self.chunker.chunk(document.content, metadata={"source": document.source, "doc_id": doc_id})
        chunk_texts = [c.text for c in chunks]
        logger.info(f"Document chunked into {len(chunks)} chunks")

        messages = [
            {"role": "system", "content": self._build_system_prompt(document, doc_id)},
            {"role": "user",   "content": self._build_user_message(document, doc_id, chunk_texts, document.content[:5000])},
        ]

        tool_results: dict[str, Any] = {}
        iterations = 0
        message = None

        while iterations < MAX_ITERATIONS:
            iterations += 1
            logger.debug(f"Agent iteration {iterations}")

            @with_retry(max_attempts=3, min_wait=1, max_wait=30)
            def _call_agent():
                return self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    tools=TOOL_SCHEMAS, tool_choice="auto", messages=messages,
                )

            response = _call_agent()
            if self.cost_tracker:
                self.cost_tracker.record(self.model, f"agent_loop_iter_{iterations}", response.usage)

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            messages.append(message.model_dump(exclude_none=True))

            if finish_reason == "stop":
                logger.info(f"Agent completed in {iterations} iterations")
                break

            if finish_reason == "tool_calls" and message.tool_calls:
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    logger.info(f"Calling tool: {tool_name}({list(args.keys())})")
                    result = self.executor.execute(tool_name, args)
                    tool_results[tool_name] = result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
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

    def _build_system_prompt(self, doc: Document, doc_id: str) -> str:
        mode_instructions = {
            "full": (
                "Call ALL tools in order:\n"
                "1. extract_structured_data\n2. evaluate_extraction\n"
                "3. embed_and_store_chunks\n4. extract_and_store_knowledge_graph\n5. generate_summary\n"
                "Then write a brief completion message."
            ),
            "extract_only": "Call extract_structured_data, evaluate_extraction, generate_summary only.",
            "search_only":  "Call embed_and_store_chunks only.",
            "graph_only":   "Call extract_and_store_knowledge_graph only.",
        }
        instructions = mode_instructions.get(self.pipeline_mode, mode_instructions["full"])
        return (
            f"You are a Scientific Document Intelligence Agent.\n\n"
            f"Document: {doc.metadata.get('title','Unknown')} | ID: {doc_id} | {doc.word_count} words\n"
            f"Mode: {self.pipeline_mode.upper()}\n\n{instructions}\n\n"
            "Choose schema_type based on document content (research_paper for academic, patent for patents)."
        )

    def _build_user_message(self, doc: Document, doc_id: str, chunk_texts: list[str], preview: str) -> str:
        return (
            f"Process this document.\n\nDocument ID: {doc_id}\nSource: {doc.source}\n"
            f"Total chunks: {len(chunk_texts)}\n\nDocument preview:\n{preview}\n\n"
            f"First 3 chunks:\n{'---'.join(chunk_texts[:3])}\n\n"
            f"All chunks to embed: {json.dumps(chunk_texts)}\n\nBegin processing."
        )

    @staticmethod
    def _build_chunker(client, cfg: dict):
        strategy = cfg.get("strategy", "sentence")
        if strategy == "semantic":
            from src.utils.chunker import SemanticChunker
            return SemanticChunker(
                openai_client=client,
                similarity_threshold=cfg.get("similarity_threshold", 0.65),
                sentence_group_size=cfg.get("sentence_group_size", 3),
                max_chunk_size=cfg.get("chunk_size", 1500),
                min_chunk_size=cfg.get("min_chunk_size", 100),
            )
        return TextChunker(
            strategy=strategy,
            chunk_size=cfg.get("chunk_size", 1500),
            chunk_overlap=cfg.get("chunk_overlap", 200),
            min_chunk_size=cfg.get("min_chunk_size", 100),
        )

    @staticmethod
    def _make_doc_id(source: str) -> str:
        import hashlib, re
        from pathlib import Path
        name = Path(source).stem if not source.startswith("http") else source
        suffix = hashlib.md5(source.encode()).hexdigest()[:8]
        clean = re.sub(r'[^\w]', '_', name)[:30]
        return f"{clean}_{suffix}"
