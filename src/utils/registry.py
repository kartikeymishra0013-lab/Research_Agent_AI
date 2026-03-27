"""
Document processing registry for deduplication.

Tracks processed documents by content hash so the pipeline
never re-processes the same file twice (unless --force is passed).
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingRegistry:
    """
    Content-hash-based registry persisted as a JSON file.

    - Files are identified by MD5 hash of their binary content
    - URLs are identified by MD5 hash of the URL string
    - Thread-safe for parallel document processing
    """

    def __init__(self, registry_path: str = "data/output/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._registry: dict = self._load()

    # ─── Public API ───────────────────────────────────────────────

    def is_processed(self, source: str, force: bool = False) -> bool:
        """
        Check if source has already been successfully processed.

        Args:
            source : File path or URL string.
            force  : If True, always returns False (re-process everything).
        """
        if force:
            return False
        file_hash = self._hash(source)
        record = self._registry.get(file_hash)
        if record and record.get("status") == "success":
            logger.info(f"Skipping already-processed source: {source}")
            return True
        return False

    def mark_processed(
        self,
        source: str,
        doc_id: str,
        status: str = "success",
        metadata: Optional[dict] = None,
    ):
        """
        Register a source as processed.

        Args:
            source   : Original file path or URL.
            doc_id   : Pipeline-generated document identifier.
            status   : "success" or "error".
            metadata : Optional extra info (page_count, word_count, etc.)
        """
        file_hash = self._hash(source)
        record = {
            "source": source,
            "doc_id": doc_id,
            "status": status,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        with self._lock:
            self._registry[file_hash] = record
            self._save()

        logger.debug(f"Registered {status}: {source} → {doc_id}")

    def get_record(self, source: str) -> Optional[dict]:
        """Return the registry record for a source, or None."""
        return self._registry.get(self._hash(source))

    def all_records(self) -> list[dict]:
        """Return all registry records."""
        return list(self._registry.values())

    def stats(self) -> dict:
        records = self.all_records()
        return {
            "total": len(records),
            "success": sum(1 for r in records if r.get("status") == "success"),
            "error": sum(1 for r in records if r.get("status") == "error"),
        }

    def clear(self):
        """Remove all registry entries (allows full reprocessing)."""
        with self._lock:
            self._registry.clear()
            self._save()
        logger.warning("Processing registry cleared")

    # ─── Internals ────────────────────────────────────────────────

    def _hash(self, source: str) -> str:
        """
        Generate a stable hash for a source.
        - Files: MD5 of binary file content (true content dedup)
        - URLs:  MD5 of URL string
        """
        if source.startswith("http://") or source.startswith("https://"):
            return hashlib.md5(source.encode("utf-8")).hexdigest()

        path = Path(source)
        if path.exists() and path.is_file():
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()

        # Fallback to path string hash
        return hashlib.md5(source.encode("utf-8")).hexdigest()

    def _load(self) -> dict:
        if not self.registry_path.exists():
            return {}
        try:
            with open(self.registry_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Could not load registry from {self.registry_path}, starting fresh")
            return {}

    def _save(self):
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2, default=str)
        except OSError as e:
            logger.error(f"Failed to save registry: {e}")
