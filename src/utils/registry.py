"""
Content-hash deduplication registry for the pipeline.

Persists a JSON file mapping content-hash → processing record so that
already-processed documents are skipped on subsequent runs.

Usage:
    registry = ProcessingRegistry("data/output/registry.json")
    if not registry.is_processed(source):
        # ... process ...
        registry.mark_processed(source, doc_id, status="success")
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingRegistry:
    """
    Thread-safe, file-backed registry that tracks which documents have
    already been processed, using content-hash keys to detect duplicates.

    Hash strategy:
        - Files : MD5 of binary file content  (catches renames of same file)
        - URLs  : MD5 of the URL string       (stable across restarts)

    Args:
        registry_path : Path to the JSON persistence file.
    """

    def __init__(self, registry_path: str = "data/output/registry.json"):
        self._path = Path(registry_path)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._load()

    # ─── Public API ───────────────────────────────────────────────────────────

    def is_processed(self, source: str, force: bool = False) -> bool:
        """
        Return True if ``source`` has been successfully processed before.

        Args:
            source : File path or URL.
            force  : If True, always return False (treat as unprocessed).
        """
        if force:
            return False
        h = self._hash(source)
        with self._lock:
            record = self._data.get(h)
        return record is not None and record.get("status") == "success"

    def mark_processed(
        self,
        source: str,
        doc_id: str,
        status: str = "success",
        metadata: Optional[dict] = None,
    ):
        """
        Record that ``source`` has been processed.

        Args:
            source   : File path or URL.
            doc_id   : Pipeline-assigned document identifier.
            status   : "success" or "error".
            metadata : Optional extra data to store (word_count, page_count, …).
        """
        import datetime
        h = self._hash(source)
        record = {
            "source": source,
            "doc_id": doc_id,
            "status": status,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        with self._lock:
            self._data[h] = record
            self._save()
        logger.debug(f"Registry: marked {source!r} as {status!r} (hash={h[:8]}…)")

    def get_record(self, source: str) -> Optional[dict]:
        """Return the stored record for a source, or None if not found."""
        h = self._hash(source)
        with self._lock:
            return self._data.get(h)

    def stats(self) -> dict:
        """Return summary statistics for the registry."""
        with self._lock:
            total   = len(self._data)
            success = sum(1 for r in self._data.values() if r.get("status") == "success")
            error   = sum(1 for r in self._data.values() if r.get("status") == "error")
        return {"total": total, "success": success, "error": error}

    def clear(self):
        """Remove all entries from the registry and persist the empty state."""
        with self._lock:
            self._data.clear()
            self._save()
        logger.warning("ProcessingRegistry cleared")

    def list_all(self) -> list[dict]:
        """Return all stored records as a list."""
        with self._lock:
            return list(self._data.values())

    # ─── Internals ────────────────────────────────────────────────────────────

    def _hash(self, source: str) -> str:
        """
        Compute a stable content key for *source*:
            - If it looks like a URL → MD5 of the URL string
            - If it's a file that exists → MD5 of the file's binary content
            - Otherwise → MD5 of the source string
        """
        path = Path(source)
        if source.startswith(("http://", "https://")):
            return hashlib.md5(source.encode()).hexdigest()
        if path.exists() and path.is_file():
            try:
                return hashlib.md5(path.read_bytes()).hexdigest()
            except OSError:
                pass
        return hashlib.md5(source.encode()).hexdigest()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.debug(f"Registry loaded: {len(self._data)} entries from {self._path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load registry from {self._path}: {e} — starting fresh")
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)
