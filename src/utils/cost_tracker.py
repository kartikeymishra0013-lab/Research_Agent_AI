"""
Token usage and cost tracker for all OpenAI API calls.

Thread-safe — safe to use in parallel document processing.
Pass a CostTracker instance through the pipeline and call
tracker.record() after every OpenAI response.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ── OpenAI Pricing (USD per token, as of 2025) ────────────────────────────────
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":                    {"input": 2.50  / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini":               {"input": 0.15  / 1_000_000, "output": 0.60  / 1_000_000},
    "gpt-4-turbo":               {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    "gpt-3.5-turbo":             {"input": 0.50  / 1_000_000, "output": 1.50  / 1_000_000},
    "text-embedding-3-small":    {"input": 0.02  / 1_000_000, "output": 0.0},
    "text-embedding-3-large":    {"input": 0.13  / 1_000_000, "output": 0.0},
    "text-embedding-ada-002":    {"input": 0.10  / 1_000_000, "output": 0.0},
}


@dataclass
class UsageRecord:
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "operation": self.operation,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "timestamp": self.timestamp,
        }


class CostTracker:
    """
    Thread-safe cost tracker.

    Usage:
        tracker = CostTracker()
        response = client.chat.completions.create(...)
        tracker.record(model="gpt-4o", operation="extract", usage=response.usage)

        print(tracker.summary())
    """

    def __init__(self, warn_threshold_usd: float = 1.0):
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self.warn_threshold_usd = warn_threshold_usd

    def record(self, model: str, operation: str, usage: Any) -> float:
        """
        Record token usage from an OpenAI response.

        Args:
            model     : Model name (e.g. "gpt-4o")
            operation : Human-readable description (e.g. "schema_extraction")
            usage     : OpenAI usage object with prompt_tokens, completion_tokens

        Returns:
            Cost in USD for this call.
        """
        input_tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)

        pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])

        record = UsageRecord(
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        with self._lock:
            self._records.append(record)
            total = self.total_cost()

        if total >= self.warn_threshold_usd:
            import logging
            logging.getLogger(__name__).warning(
                f"Pipeline cost has reached ${total:.4f} (threshold: ${self.warn_threshold_usd})"
            )

        return cost

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self._records)

    def total_tokens(self) -> dict[str, int]:
        return {
            "input":  sum(r.input_tokens for r in self._records),
            "output": sum(r.output_tokens for r in self._records),
            "total":  sum(r.input_tokens + r.output_tokens for r in self._records),
        }

    def summary(self) -> dict:
        by_model: dict[str, dict] = {}
        by_operation: dict[str, dict] = {}

        for r in self._records:
            # By model
            m = by_model.setdefault(r.model, {
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0
            })
            m["input_tokens"]  += r.input_tokens
            m["output_tokens"] += r.output_tokens
            m["cost_usd"]      += r.cost_usd
            m["calls"]         += 1

            # By operation
            o = by_operation.setdefault(r.operation, {
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0
            })
            o["input_tokens"]  += r.input_tokens
            o["output_tokens"] += r.output_tokens
            o["cost_usd"]      += r.cost_usd
            o["calls"]         += 1

        return {
            "total_cost_usd":    round(self.total_cost(), 6),
            "total_tokens":      self.total_tokens(),
            "call_count":        len(self._records),
            "by_model":          {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in by_model.items()},
            "by_operation":      {k: {**v, "cost_usd": round(v["cost_usd"], 6)} for k, v in by_operation.items()},
        }

    def records(self) -> list[dict]:
        return [r.to_dict() for r in self._records]

    def reset(self):
        with self._lock:
            self._records.clear()
