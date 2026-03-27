"""
Pipeline Orchestrator.
Upgrades: parallel processing (ThreadPoolExecutor), deduplication (ProcessingRegistry),
CostTracker threaded through all components, SemanticChunker support.
"""
from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import yaml

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
    def __init__(self, config_path: str | Path = CONFIG_PATH, config_overrides: dict = None):
        self.config = self._load_config(config_path, config_overrides or {})
        self.run_id = str(uuid.uuid4())[:8]

        cost_cfg = self.config.get("cost_tracking", {})
        self.cost_tracker = CostTracker(warn_threshold_usd=cost_cfg.get("warn_threshold_usd", 1.0))

        reg_cfg = self.config.get("registry", {})
        self.registry = ProcessingRegistry(registry_path=reg_cfg.get("path", "data/output/registry.json"))

        self._openai_client = None
        self._chroma_store: Optional[ChromaStore] = None
        self._neo4j_store: Optional[Neo4jStore] = None
        self._json_store: Optional[JsonStore] = None
        self._agent: Optional[DocumentAgent] = None

        logger.info(f"Pipeline initialized | run_id={self.run_id} | mode={self.config['pipeline']['mode']}")

    # ─── Public API ───────────────────────────────────────────────────────────

    def process_file(self, file_path: str, force: bool = False) -> dict:
        return self._process_source(file_path, force=force)

    def process_url(self, url: str, force: bool = False) -> dict:
        return self._process_source(url, force=force)

    def process_directory(self, dir_path: str, glob_pattern: str = "**/*", force: bool = False) -> dict:
        directory = Path(dir_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        supported = {".pdf", ".docx", ".doc", ".txt", ".md", ".rst"}
        files = [f for f in directory.glob(glob_pattern) if f.is_file() and f.suffix.lower() in supported]

        if not files:
            return {"results": [], "total": 0, "succeeded": 0, "failed": 0, "skipped": 0}

        to_process = [f for f in files if not self.registry.is_processed(str(f), force=force)]
        skipped = len(files) - len(to_process)
        if skipped:
            logger.info(f"Skipping {skipped} already-processed documents")

        if not to_process:
            return {"run_id": self.run_id, "total": len(files), "succeeded": 0, "failed": 0,
                    "skipped": skipped, "results": []}

        max_workers = self.config.get("processing", {}).get("max_workers", 4)
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self._process_source, str(f), force=force): str(f) for f in to_process}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append({"source": futures[future], "status": "error", "error": str(e)})

        succeeded = sum(1 for r in results if r.get("status") == "success")
        aggregated = {
            "run_id": self.run_id,
            "total": len(files),
            "processed": len(to_process),
            "skipped": skipped,
            "succeeded": succeeded,
            "failed": len(results) - succeeded,
            "results": results,
            "cost_summary": self.cost_tracker.summary(),
        }
        self.json_store.save_pipeline_result(aggregated, self.run_id)
        logger.info(f"Run {self.run_id}: {succeeded}/{len(to_process)} ok | cost=${self.cost_tracker.total_cost():.4f}")
        return aggregated

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
        return self.chroma_store.search(query, top_k=top_k, where=filters)

    def get_stats(self) -> dict:
        stats = {"run_id": self.run_id, "pipeline_mode": self.config["pipeline"]["mode"],
                 "registry": self.registry.stats()}
        try: stats["chromadb_chunks"] = self.chroma_store.count()
        except Exception as e: stats["chromadb_error"] = str(e)
        try: stats["neo4j"] = self.neo4j_store.get_stats()
        except Exception as e: stats["neo4j_error"] = str(e)
        stats["output_files"] = self.json_store.list_outputs()
        stats["cost_summary"] = self.cost_tracker.summary()
        return stats

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _process_source(self, source: str, force: bool = False) -> dict:
        start = time.time()
        logger.info(f"Processing: {source}")
        try:
            loader = get_loader(source, config=self.config.get("loaders", {}))
            document: Document = loader.load(source)
            result = self.agent.process(document)
            self._persist_results(result, source)
            self.registry.mark_processed(source, result.get("doc_id", ""), "success",
                                          metadata={"word_count": document.word_count,
                                                    "doc_type": document.doc_type})
            elapsed = round(time.time() - start, 2)
            result.update(status="success", elapsed_seconds=elapsed)
            logger.info(f"Processed {source!r} in {elapsed}s")
            return result
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            logger.error(f"Failed to process {source!r}: {e}", exc_info=True)
            try: self.registry.mark_processed(source, "", "error")
            except Exception: pass
            return {"source": source, "status": "error", "error": str(e), "elapsed_seconds": elapsed}

    def _persist_results(self, result: dict, source: str):
        tool_results = result.get("tool_results", {})
        schema_type = self.config["pipeline"].get("schema_type", "default")
        if "extract_structured_data" in tool_results:
            extracted = tool_results["extract_structured_data"].get("data", {})
            self.json_store.save_extraction(extracted, source, schema_type=schema_type)
            self.json_store.append_to_dataset({"doc_id": result["doc_id"], **extracted}, dataset_name="extractions")
        if "generate_summary" in tool_results:
            self.json_store.save_summary_report(
                tool_results["generate_summary"].get("summary", ""), source, metadata=result.get("metadata"))
        if "extract_and_store_knowledge_graph" in tool_results:
            graph_data = tool_results["extract_and_store_knowledge_graph"].get("graph_data", {})
            if graph_data:
                self.json_store.save_graph_data(graph_data, source)

    # ─── Service properties ───────────────────────────────────────────────────

    @property
    def openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY") or self.config.get("openai", {}).get("api_key")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self._openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized")
        return self._openai_client

    @property
    def chroma_store(self) -> ChromaStore:
        if self._chroma_store is None:
            cfg = self.config.get("chromadb", {})
            self._chroma_store = ChromaStore(
                host=cfg.get("host", "chromadb"), port=cfg.get("port", 8000),
                collection_name=cfg.get("collection", "documents"),
                embedding_model=cfg.get("embedding_model", "text-embedding-3-small"),
                openai_client=self.openai_client, cost_tracker=self.cost_tracker,
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
            self._json_store = JsonStore(output_dir=self.config.get("output", {}).get("dir", "data/output"))
        return self._json_store

    @property
    def agent(self) -> DocumentAgent:
        if self._agent is None:
            pcfg = self.config.get("pipeline", {})
            self._agent = DocumentAgent(
                client=self.openai_client,
                executor=ToolExecutor(
                    schema_extractor=SchemaExtractor(
                        client=self.openai_client,
                        schema_type=pcfg.get("schema_type", "default"),
                        model=pcfg.get("model", "gpt-4o"),
                        cost_tracker=self.cost_tracker,
                    ),
                    relationship_extractor=RelationshipExtractor(
                        client=self.openai_client,
                        model=pcfg.get("model", "gpt-4o"),
                        cost_tracker=self.cost_tracker,
                    ),
                    chroma_store=self.chroma_store,
                    neo4j_store=self.neo4j_store,
                    openai_client=self.openai_client,
                    cost_tracker=self.cost_tracker,
                ),
                model=pcfg.get("model", "gpt-4o"),
                temperature=pcfg.get("temperature", 0.1),
                pipeline_mode=pcfg.get("mode", "full"),
                chunking_cfg=self.config.get("chunking", {}),
                cost_tracker=self.cost_tracker,
            )
        return self._agent

    @staticmethod
    def _load_config(config_path: Path, overrides: dict) -> dict:
        if not Path(config_path).exists():
            logger.warning(f"Config not found at {config_path}, using defaults")
            return _default_config()
        with open(config_path) as f:
            config = yaml.safe_load(f)
        _deep_merge(config, overrides)
        return _resolve_env_vars(config)


def _default_config() -> dict:
    return {
        "pipeline": {"mode": "full", "model": "gpt-4o", "temperature": 0.1, "schema_type": "default"},
        "chromadb": {"host": "chromadb", "port": 8000, "collection": "documents"},
        "neo4j":    {"uri": "bolt://neo4j:7687", "username": "neo4j", "password": "password"},
        "output":   {"dir": "data/output"},
        "processing": {"max_workers": 4, "force_reprocess": False},
        "registry": {"path": "data/output/registry.json"},
        "cost_tracking": {"warn_threshold_usd": 1.0},
        "chunking": {"strategy": "sentence", "chunk_size": 1500, "chunk_overlap": 200, "min_chunk_size": 100},
    }


def _deep_merge(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_env_vars(obj: Any) -> Any:
    import re
    if isinstance(obj, str):
        def _replace(m):
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else m.group(0))
        return re.sub(r"\$\{(\w+)(?::-(.*?))?\}", _replace, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj
