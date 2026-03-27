"""
Entity and relationship extractor for knowledge graph construction.

Enhancements over baseline:
  - Retry with exponential backoff via `with_retry`
  - Cost tracking via CostTracker
  - Chunk-level provenance (chunk_id injected into properties)
"""
from __future__ import annotations

import json
from typing import Any, Optional

from openai import OpenAI

from src.utils.logger import get_logger
from src.utils.retry import with_retry

logger = get_logger(__name__)


ENTITY_TYPES = [
    "Person", "Organization", "Technology", "Method", "Dataset",
    "Concept", "Chemical", "Gene", "Disease", "Location",
    "Publication", "Patent", "Material", "Metric",
]

RELATIONSHIP_TYPES = [
    "USES", "DEVELOPED_BY", "RELATED_TO", "APPLIED_TO", "PART_OF",
    "CITES", "IMPROVES", "COMPARES_WITH", "MEASURES", "CAUSES",
    "AUTHORED_BY", "AFFILIATED_WITH", "PUBLISHED_IN", "FUNDED_BY",
]


class RelationshipExtractor:
    """
    Extracts entities and relationships from text for knowledge graph ingestion.

    Returns a dict with:
        - entities: list of {id, label, type, properties}
        - relationships: list of {source, target, type, properties}

    Args:
        client             : Initialized OpenAI client.
        model              : OpenAI model (default: gpt-4o).
        temperature        : Model temperature (default: 0.0).
        entity_types       : Override the default entity type list.
        relationship_types : Override the default relationship type list.
        cost_tracker       : Optional CostTracker for recording API costs.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        cost_tracker=None,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.entity_types = entity_types or ENTITY_TYPES
        self.relationship_types = relationship_types or RELATIONSHIP_TYPES
        self.cost_tracker = cost_tracker

    def extract(
        self,
        text: str,
        doc_id: str = "",
        chunk_id: Optional[str] = None,
    ) -> dict[str, list]:
        """
        Extract entities and relationships from text.

        Args:
            text     : Source text to extract from.
            doc_id   : Document identifier for provenance tracking.
            chunk_id : Optional chunk identifier for fine-grained provenance.

        Returns:
            {"entities": [...], "relationships": [...]}
        """
        if not text.strip():
            return {"entities": [], "relationships": []}

        logger.debug(f"Extracting graph entities from {len(text)} chars")

        @with_retry(max_attempts=3, min_wait=1, max_wait=30)
        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self._user_prompt(text, doc_id)},
                ],
            )

        response = _call()

        if self.cost_tracker:
            self.cost_tracker.record(
                model=self.model,
                operation="relationship_extraction",
                usage=response.usage,
            )

        raw = response.choices[0].message.content
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relationship extraction response: {e}")
            return {"entities": [], "relationships": []}

        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        # Inject provenance (doc_id + optional chunk_id)
        for e in entities:
            props = e.setdefault("properties", {})
            if doc_id:
                props["doc_id"] = doc_id
            if chunk_id:
                props["chunk_id"] = chunk_id

        for r in relationships:
            props = r.setdefault("properties", {})
            if doc_id:
                props["doc_id"] = doc_id
            if chunk_id:
                props["chunk_id"] = chunk_id

        logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")
        return {"entities": entities, "relationships": relationships}

    # ─── Prompt builders ──────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        return f"""You are a knowledge graph extraction specialist for scientific documents.

Extract entities and relationships from the provided text.

ENTITY TYPES (use exact strings): {", ".join(self.entity_types)}

RELATIONSHIP TYPES (use exact strings): {", ".join(self.relationship_types)}

Return a JSON object with this exact structure:
{{
  "entities": [
    {{
      "id": "unique_slug_no_spaces",
      "label": "Human Readable Name",
      "type": "EntityType",
      "properties": {{"description": "brief description", "aliases": []}}
    }}
  ],
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "RELATIONSHIP_TYPE",
      "properties": {{"context": "brief context for this relationship"}}
    }}
  ]
}}

Rules:
- Entity IDs must be unique, lowercase, use underscores (e.g. "bert_model")
- Only use entity/relationship types from the provided lists
- Only extract what is explicitly stated — do not infer
- Return valid JSON only, no markdown fences
"""

    def _user_prompt(self, text: str, doc_id: str) -> str:
        ctx = f" (Document ID: {doc_id})" if doc_id else ""
        return f"Extract entities and relationships from this document text{ctx}:\n\n{text[:4000]}"
