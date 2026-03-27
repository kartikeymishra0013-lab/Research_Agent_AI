"""
Entity and relationship extractor for knowledge graph construction.
Upgrades: retry, cost tracking, chunk_id provenance.
"""
from __future__ import annotations

import json
from typing import Optional

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
    def __init__(
        self,
        client,
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

    def extract(self, text: str, doc_id: str = "", chunk_id: Optional[str] = None) -> dict[str, list]:
        if not text.strip():
            return {"entities": [], "relationships": []}

        @with_retry(max_attempts=3, min_wait=1, max_wait=30)
        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user",   "content": self._user_prompt(text, doc_id)},
                ],
            )

        response = _call()
        if self.cost_tracker:
            self.cost_tracker.record(self.model, "relationship_extraction", response.usage)

        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relationship extraction response: {e}")
            return {"entities": [], "relationships": []}

        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        for e in entities:
            props = e.setdefault("properties", {})
            if doc_id:   props["doc_id"]   = doc_id
            if chunk_id: props["chunk_id"] = chunk_id
        for r in relationships:
            props = r.setdefault("properties", {})
            if doc_id:   props["doc_id"]   = doc_id
            if chunk_id: props["chunk_id"] = chunk_id

        logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")
        return {"entities": entities, "relationships": relationships}

    def _system_prompt(self) -> str:
        return (
            "You are a knowledge graph extraction specialist for scientific documents.\n\n"
            f"ENTITY TYPES (use exact strings): {', '.join(self.entity_types)}\n\n"
            f"RELATIONSHIP TYPES (use exact strings): {', '.join(self.relationship_types)}\n\n"
            "Return a JSON object:\n"
            '{"entities": [{"id": "slug", "label": "Name", "type": "EntityType", "properties": {}}], '
            '"relationships": [{"source": "id", "target": "id", "type": "REL_TYPE", "properties": {}}]}\n\n'
            "Rules:\n"
            "- IDs: lowercase, underscores (e.g. bert_model)\n"
            "- Only use listed types\n"
            "- Only extract explicitly stated facts\n"
            "- Return valid JSON only, no markdown"
        )

    def _user_prompt(self, text: str, doc_id: str) -> str:
        ctx = f" (Document ID: {doc_id})" if doc_id else ""
        return f"Extract entities and relationships{ctx}:\n\n{text[:4000]}"
