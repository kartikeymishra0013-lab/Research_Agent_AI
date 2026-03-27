"""
Thread-safe token usage and USD cost tracker for OpenAI API calls.

Usage:
    tracker = CostTracker(warn_threshold_usd=1.0)
    cost = tracker.record("gpt-4o", "schema_extraction", response.usage)
    summary = tracker.summary()
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# USD per token (as of mid-2024; update as OpenAI pricing changes)
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "input":  2.50  / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input":  0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "gpt-4-turbo": {
        "input":  10.00 / 1_000_000,
        "output": 30.00 / 1_000_000,
    },
    "gpt-3.5-turbo": {
        "input":  0.50 / 1_000_000,
        "output": 1.50 / 1_000_000,
    },
    "text-embedding-3-small": {
        "input":  0.02 / 1_000_000,
        "output": 0.0,
    },
    "text-embedding-3-large": {
        "input":  0.13 / 1_000_000,
        "output": 0.0,
    },
    "text-embedding-ada-002": {
        "input":  0.10 / 1_000_000,
        "output": 0.0,
    },
}


class CostTracker:
    """
    Thread-safe tracker for OpenAI token usage and USD costs.

    Records per-model and per-operation breakdowns.  Logs a warning
    when total spend exceeds ``warn_threshold_usd``.

    Args:
        warn_threshold_usd : Emit a WARNING log when total cost exceeds this value.
    """

    def __init__(self, warn_threshold_usd: float = 1.0):
        self._lock = threading.Lock()
        self._warn_threshold = warn_threshold_usd
        self._warned = False

        # Nested dicts: model → {input_tokens, output_tokens, cost_usd}
        self._by_model: dict[str, dict[str, float]] = {}
        # Operation → {calls, tokens, cost_usd}
        self._by_operation: dict[str, dict[str, float]] = {}
        self._total_cost: float = 0.0
        self._total_tokens: int = 0

    def record(self, model: str, operation: str, usage: Any) -> float:
        """
        Record token usage from an OpenAI API response and return the USD cost.

        Args:
            model     : Model name string (e.g. "gpt-4o").
            operation : Human-readable label for the operation (e.g. "schema_extraction").
            usage     : OpenAI ``CompletionUsage`` or ``Usage`` object with
                        ``prompt_tokens``, ``completion_tokens`` / ``total_tokens``.

        Returns:
            float : Cost in USD for this single call.
        """
        prompt_tokens     = getattr(usage, "prompt_tokens",     0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens      = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        # Look up pricing; fall back to gpt-4o rates if model is unknown
        price = PRICING.get(model) or PRICING.get(self._normalize_model(model)) or PRICING["gpt-4o"]
        cost = (prompt_tokens * price["input"]) + (completion_tokens * price["output"])

        with self._lock:
            # By model
            m = self._by_model.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            m["input_tokens"]  += prompt_tokens
            m["output_tokens"] += completion_tokens
            m["cost_usd"]      += cost

            # By operation
            op = self._by_operation.setdefault(operation, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            op["calls"]    += 1
            op["tokens"]   += total_tokens
            op["cost_usd"] += cost

            self._total_cost   += cost
            self._total_tokens += total_tokens

            if not self._warned and self._total_cost >= self._warn_threshold:
                self._warned = True
                logger.warning(
                    f"CostTracker: total spend ${self._total_cost:.4f} exceeded "
                    f"threshold ${self._warn_threshold:.2f}"
                )

        logger.debug(
            f"[cost] {operation} via {model}: "
            f"{prompt_tokens}↑ {completion_tokens}↓ tokens → ${cost:.6f}"
        )
        return cost

    def total_cost(self) -> float:
        """Return total USD cost recorded so far."""
        with self._lock:
            return self._total_cost

    def summary(self) -> dict:
        """
        Return a summary dict with:
            total_cost_usd, total_tokens, by_model (dict), by_operation (dict)
        """
        with self._lock:
            return {
                "total_cost_usd": round(self._total_cost, 8),
                "total_tokens":   self._total_tokens,
                "by_model": {
                    model: {
                        "input_tokens":  d["input_tokens"],
                        "output_tokens": d["output_tokens"],
                        "cost_usd":      round(d["cost_usd"], 8),
                    }
                    for model, d in self._by_model.items()
                },
                "by_operation": {
                    op: {
                        "calls":    d["calls"],
                        "tokens":   d["tokens"],
                        "cost_usd": round(d["cost_usd"], 8),
                    }
                    for op, d in self._by_operation.items()
                },
            }

    def reset(self):
        """Reset all counters (useful between test runs)."""
        with self._lock:
            self._by_model.clear()
            self._by_operation.clear()
            self._total_cost = 0.0
            self._total_tokens = 0
            self._warned = False

    @staticmethod
    def _normalize_model(model: str) -> str:
        """Strip version suffixes for fallback lookup (e.g. 'gpt-4o-2024-05-13' → 'gpt-4o')."""
        for key in PRICING:
            if model.startswith(key):
                return key
        return model
