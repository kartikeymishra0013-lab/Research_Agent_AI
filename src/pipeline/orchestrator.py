"""
Pipeline Orchestrator — the top-level coordinator for document processing.

Enhancements over baseline:
  - Parallel directory processing via ThreadPoolExecutor (configurable max_workers)
  - Deduplication via ProcessingRegistry (skip already-processed, --force flag)
  - CostTracker threaded through all components; cost summary saved in run result
  - SemanticChunker supported via config chunking.strategy = "semantic"
"""
from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import yaml
from openai import OpenAI

from src.agent.document_agent import DocumentAgent
from src.agent.tools import ToolExecutor
from src.extraction.relationship_extractor import RelationshipExtractor
from src.extraction.schema_extractor import SchemaExtractor
from src.ingestion import get_loader
from src.ingestion.base_loader import Document
from src.storage.chroma_store import ChromaStore
from src.storage.json_store import JsonStore
from src.storage.neo4j_store import Neo4jStore
from src.utils.cost_tracker import CostTracker
from src.utils.logger import get_logger
from src.utils.registry import ProcessingRegistry

logger = get_logger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"


class PipelineOrchestrator:
    """
    End-to-end pipeline orchestrator for Scientific Document Intelligence.

    Responsibilities:
        - Load and validate configuration
        - Initialize all service connections
        - Process single documents or entire directories (parallel)
        - Skip already-processed documents (deduplication via registry)
        - Track API cost across all pipeline stages
        - Handle errors gracefully with per-document fault isolation
        - Persist all results to configured output stores
    """

    def __init__(
        self,
        config_path: str | Path = CONFIG_PATH,
        config_overrides: dict = None,
    ):
        self.config = self._load_config(config_path, config_overrides or {})
        self.run_id = str(uuid.uuid4())[:8]

        # Shared cost tracker — passed to all components
        cost_cfg = self.config.get("cost_tracking", {})
        self.cost_tracker = CostTracker(
            warn_threshold_usd=cost_cfg.get("warn_threshold_usd", 1.0)
        )

        # Processing registry for deduplication
        reg_cfg = self.config.get("registry", {})
        registry_path = reg_cfg.get("path", "data/output/registry.json")
        self.registry = ProcessingRegistry(registry_path=registry_path)

        # Services (lazily initialized)
        self._openai_client: Optional[OpenAI] = None
        self._chroma_store: Optional[ChromaStore] = None
        self._neo4j_store: Optional[Neo4jStore] = None
        self._json_store: Optional[JsonStore] = None
        self._agent: Optional[DocumentAgent] = None

        logger.info(
            f"Pipeline initialized | run_id={self.run_id} "
            f"| mode={self.config['pipeline']['mode']}"
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def process_file(self, file_path: str, force: bool = False) -> dict:
        """Process a single document file."""
        return self._process_source(file_path, force=force)

    def process_url(self, url: str, force: bool = False) -> dict:
        """Process a document from a URL."""
        return self._process_source(url, force=force)

    def process_directory(
        self,
        dir_path: str,
        glob_pattern: str = "**/*",
        force: bool = False,
    ) -> dict:
        """
        Process all supported documents in a directory using parallel workers.

        Args:
            dir_path     : Directory path to scan.
            glob_pattern : Glob pattern to filter files.
            force        : If True, re-process already-indexed documents.

        Returns:
            Aggregated run result dict.
        """
        directory = Path(dir_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        supported_exts = {".pdf", ".docx", ".doc", ".txt", ".md", ".rst"}
        files = [
            f for f in directory.glob(glob_pattern)
            if f.is_file() and f.suffix.lower() in supported_exts
        ]

        if not files:
            logger.warning(f"No supported files found in {dir_path}")
            return {"results": [], "total": 0, "succeeded": 0, "failed": 0, "skipped": 0}

        # Filter already-processed (unless force=True)
        to_process = []
        skipped = []
        for f in files:
            if self.registry.is_processed(str(f), force=force):
                skipped.append(str(f))
            else:
                to_process.append(f)

        if skipped:
            logger.info(f"Skipping {len(skipped)} already-processed documents (use --force to override)")

        if not to_process:
            return {
                "run_id": self.run_id,
                "total": len(files),
                "succeeded": 0,
                "failed": 0,
                "skipped": len(skipped),
                "results": [],
            }

        logger.info(f"Processing {len(to_process)} documents with parallel workers")

        # Parallel processing
        proc_cfg = self.config.get("processing", {})
        max_workers = proc_cfg.get("max_workers", 4)

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_file = {
                pool.submit(self._process_source, str(f), force=force): str(f)
                for f in to_process
            }
            for future in as_completed(future_to_file):
                file_path_str = future_to_file[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "source": file_path_str,
                        "status": "error",
                        "error": str(e),
                    }
                results.append(result)

        succeeded = sum(1 for r in results if r.get("status") == "success")
        failed = len(results) - succeeded

        aggregated = {
            "run_id": self.run_id,
            "total": len(files),
            "processed": len(to_process),
            "skipped": len(skipped),
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
            "cost_summary": self.cost_tracker.summary(),
        }

        self.json_store.save_pipeline_result(aggregated, self.run_id)
        logger.info(
            f"Run {self.run_id} complete — "
            f"{succeeded}/{len(to_process)} ok, {failed} failed, {len(skipped)} skipped | "
            f"cost=${self.cost_tracker.total_cost():.4f}"
        )
        return aggregated

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
        """Run semantic search over the indexed document corpus."""
        return self.chroma_store.search(query, top_k=top_k, where=filters)

    def get_stats(self) -> dict:
        """Return current stats for all connected services."""
        stats = {
            "run_id": self.run_id,
            "pipeline_mode": self.config["pipeline"]["mode"],
            "registry": self.registry.stats(),
        }
        try:
            stats["chromadb_chunks"] = self.chroma_store.count()
        except Exception as e:
            stats["chromadb_error"] = str(e)
        try:
            stats["neo4j"] = self.neo4j_store.get_stats()
        except Exception as e:
            stats["neo4j_error"] = str(e)
        stats["output_files"] = self.json_store.list_outputs()
        stats["cost_summary"] = self.cost_tracker.summary()
        return stats

    # ─── Internal Processing ──────────────────────────────────────────────────

    def _process_source(self, source: str, force: bool = False) -> dict:
        start_time = time.time()
        logger.info(f"Processing source: {source}")

        try:
            # 1. Load document
            loader_cfg = self.config.get("loaders", {})
            loader = get_loader(source, config=loader_cfg)
            document: Document = loader.load(source)

            # 2. Run agent
            result = self.agent.process(document)

            # 3. Persist outputs
            self._persist_results(result, source)

            # 4. Mark as processed in registry
            self.registry.mark_processed(
                source=source,
                doc_id=result.get("doc_id", ""),
                status="success",
                metadata={
                    "word_count": document.word_count,
                    "page_count": document.metadata.get("page_count"),
                    "doc_type": document.doc_type,
                },
            )

            elapsed = round(time.time() - start_time, 2)
            result["status"] = "success"
            result["elapsed_seconds"] = elapsed
            logger.info(f"Processed {source!r} in {elapsed}s ✓")
            return result

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"Failed to process {source!r}: {e}", exc_info=True)

            # Mark as failed so it can be retried without --force
            try:
                self.registry.mark_processed(source=source, doc_id="", status="error")
            except Exception:
                pass

            return {
                "source": source,
                "status": "error",
                "error": str(e),
                "elapsed_seconds": elapsed,
            }

    def _persist_results(self, result: dict, source: str):
        """Persist agent results to configured output stores."""
        tool_results = result.get("tool_results", {})
        schema_type = self.config["pipeline"].get("schema_type", "default")

        if "extract_structured_data" in tool_results:
            extracted = tool_results["extract_structured_data"].get("data", {})
            self.json_store.save_extraction(extracted, source, schema_type=schema_type)
            self.json_store.append_to_dataset(
                {"doc_id": result["doc_id"], **extracted},
                dataset_name="extractions",
            )

        if "generate_summary" in tool_results:
            summary = tool_results["generate_summary"].get("summary", "")
            self.json_store.save_summary_report(
                summary, source, metadata=result.get("metadata")
            )

        if "extract_and_store_knowledge_graph" in tool_results:
            graph_data = tool_results["extract_and_store_knowledge_graph"].get("graph_data", {})
            if graph_data:
                self.json_store.save_graph_data(graph_data, source)

    # ─── Service Initialization ───────────────────────────────────────────────

    @property
    def openai_client(self) -> OpenAI:
        if self._openai_client is None:
            api_key = os.environ.get("OPENAI_API_KEY") or self.config.get("openai", {}).get("api_key")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or config.yaml")
            self._openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        return self._openai_client

    @property
    def chroma_store(self) -> ChromaStore:
        if self._chroma_store is None:
            cfg = self.config.get("chromadb", {})
            self._chroma_store = ChromaStore(
                host=cfg.get("host", "chromadb"),
                port=cfg.get("port", 8000),
                collection_name=cfg.get("collection", "documents"),
                embedding_model=cfg.get(
                    "embedding_model",
                    self.config.get("openai", {}).get("embedding_model", "text-embedding-3-small"),
                ),
                openai_client=self.openai_client,
                cost_tracker=self.cost_tracker,
            )
            self._chroma_store.connect()
        return self._chroma_store

    @property
    def neo4j_store(self) -> Neo4jStore:
        if self._neo4j_store is None:
            cfg = self.config.get("neo4j", {})
            self._neo4j_store = Neo4jStore(
                uri=cfg.get("uri", os.environ.get("NEO4J_URI", "bolt://neo4j:7687")),
                username=cfg.get("username", os.environ.get("NEO4J_USERNAME", "neo4j")),
                password=cfg.get("password", os.environ.get("NEO4J_PASSWORD", "password")),
            )
            self._neo4j_store.connect()
        return self._neo4j_store

    @property
    def json_store(self) -> JsonStore:
        if self._json_store is None:
            output_dir = self.config.get("output", {}).get("dir", "data/output")
            self._json_store = JsonStore(output_dir=output_dir)
        return self._json_store

    @property
    def agent(self) -> DocumentAgent:
        if self._agent is None:
            pipeline_cfg = self.config.get("pipeline", {})
            chunking_cfg = self.config.get("chunking", {})

            schema_extractor = SchemaExtractor(
                client=self.openai_client,
                schema_type=pipeline_cfg.get("schema_type", "default"),
                model=pipeline_cfg.get("model", "gpt-4o"),
                cost_tracker=self.cost_tracker,
            )
            relationship_extractor = RelationshipExtractor(
                client=self.openai_client,
                model=pipeline_cfg.get("model", "gpt-4o"),
                cost_tracker=self.cost_tracker,
            )

            executor = ToolExecutor(
                schema_extractor=schema_extractor,
                relationship_extractor=relationship_extractor,
                chroma_store=self.chroma_store,
                neo4j_store=self.neo4j_store,
                openai_client=self.openai_client,
                cost_tracker=self.cost_tracker,
            )

            self._agent = DocumentAgent(
                client=self.openai_client,
                executor=executor,
                model=pipeline_cfg.get("model", "gpt-4o"),
                temperature=pipeline_cfg.get("temperature", 0.1),
                pipeline_mode=pipeline_cfg.get("mode", "full"),
                chunking_cfg=chunking_cfg,
                cost_tracker=self.cost_tracker,
            )
        return self._agent

    # ─── Config Loading ───────────────────────────────────────────────────────

    @staticmethod
    def _load_config(config_path: Path, overrides: dict) -> dict:
        if not Path(config_path).exists():
            logger.warning(f"Config not found at {config_path}, using defaults")
            return _default_config()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        _deep_merge(config, overrides)
        config = _resolve_env_vars(config)
        return config


def _default_config() -> dict:
    return {
        "pipeline": {"mode": "full", "model": "gpt-4o", "temperature": 0.1, "schema_type": "default"},
        "chromadb": {"host": "chromadb", "port": 8000, "collection": "documents"},
        "neo4j": {"uri": "bolt://neo4j:7687", "username": "neo4j", "password": "password"},
        "output": {"dir": "data/output"},
        "processing": {"max_workers": 4, "force_reprocess": False},
        "registry": {"path": "data/output/registry.json"},
        "cost_tracking": {"warn_threshold_usd": 1.0},
        "chunking": {
            "strategy": "sentence",
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "min_chunk_size": 100,
        },
    }


def _deep_merge(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${VAR_NAME} placeholders in config values."""
    import re
    if isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), obj
        )
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj
