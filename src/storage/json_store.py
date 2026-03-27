"""
JSON/JSONL output store for structured dataset serialization.
Supports both pretty-printed JSON reports and streaming JSONL datasets.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class JsonStore:
    """
    Manages structured JSON/JSONL output for extracted documents.

    Output formats:
        - JSONL  : One JSON object per line (streaming datasets, large corpora)
        - JSON   : Pretty-printed single file (summaries, reports)
        - Dataset: Aggregated dataset file combining all document extractions
    """

    def __init__(self, output_dir: str = "data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_extraction(
        self,
        extracted: dict,
        source: str,
        schema_type: str = "default",
        pretty: bool = True,
    ) -> Path:
        """
        Save a single document's structured extraction to JSON.

        Args:
            extracted   : Dict of extracted fields.
            source      : Original source path/URL.
            schema_type : Schema used for extraction.
            pretty      : Whether to pretty-print the output.

        Returns:
            Path to saved file.
        """
        doc_name = self._source_to_name(source)
        filename = f"{doc_name}_{schema_type}_extraction.json"
        output_path = self.output_dir / filename

        record = {
            "_meta": {
                "source": source,
                "schema_type": schema_type,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            },
            **extracted,
        }

        self._write_json(record, output_path, pretty=pretty)
        logger.info(f"Saved extraction to {output_path}")
        return output_path

    def append_to_dataset(self, record: dict, dataset_name: str = "dataset") -> Path:
        """
        Append a record to a JSONL dataset file (streaming append).

        Args:
            record       : Dict to append.
            dataset_name : Base name for the dataset file.

        Returns:
            Path to the dataset file.
        """
        output_path = self.output_dir / f"{dataset_name}.jsonl"
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output_path

    def save_summary_report(
        self,
        summary: str,
        source: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Save a generated summary report."""
        doc_name = self._source_to_name(source)
        filename = f"{doc_name}_summary.json"
        output_path = self.output_dir / filename

        record = {
            "source": source,
            "summary": summary,
            "metadata": metadata or {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._write_json(record, output_path, pretty=True)
        logger.info(f"Saved summary report to {output_path}")
        return output_path

    def save_graph_data(self, graph_data: dict, source: str) -> Path:
        """Save extracted knowledge graph entities and relationships."""
        doc_name = self._source_to_name(source)
        filename = f"{doc_name}_graph.json"
        output_path = self.output_dir / filename

        self._write_json(graph_data, output_path, pretty=True)
        logger.info(f"Saved graph data to {output_path}")
        return output_path

    def save_pipeline_result(self, result: dict, run_id: str) -> Path:
        """Save the complete pipeline run result."""
        filename = f"pipeline_run_{run_id}.json"
        output_path = self.output_dir / filename
        self._write_json(result, output_path, pretty=True)
        logger.info(f"Saved pipeline result to {output_path}")
        return output_path

    def load_dataset(self, dataset_name: str = "dataset") -> list[dict]:
        """Load all records from a JSONL dataset."""
        path = self.output_dir / f"{dataset_name}.jsonl"
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def list_outputs(self) -> list[str]:
        """List all output files in the output directory."""
        return [f.name for f in self.output_dir.iterdir() if f.is_file()]

    @staticmethod
    def _write_json(data: Any, path: Path, pretty: bool = True):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(data, f, ensure_ascii=False, default=str)

    @staticmethod
    def _source_to_name(source: str) -> str:
        """Convert a source path/URL to a safe filename stem."""
        import re
        if source.startswith("http"):
            from urllib.parse import urlparse
            parsed = urlparse(source)
            name = parsed.netloc + parsed.path
        else:
            name = Path(source).stem
        return re.sub(r"[^\w\-]", "_", name)[:60]
