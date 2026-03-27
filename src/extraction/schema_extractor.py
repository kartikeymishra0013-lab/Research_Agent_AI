"""
Schema-aware structured data extractor using GPT-4o.

Transforms raw document text into structured JSON records
according to configurable domain schemas (research_paper, patent, default).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from src.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMAS_DIR = Path(__file__).parent.parent.parent / "config" / "schemas"


class SchemaExtractor:
    """
    Extracts structured data from text using a GPT-4o prompt
    driven by a JSON schema definition.

    Usage:
        extractor = SchemaExtractor(client=openai_client, schema_type="research_paper")
        result = extractor.extract(chunk_text)
    """

    def __init__(
        self,
        client: OpenAI,
        schema_type: str = "default",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        custom_schema: Optional[dict] = None,
    ):
        self.client = client
        self.schema_type = schema_type
        self.model = model
        self.temperature = temperature
        self.schema = custom_schema or self._load_schema(schema_type)

    def extract(self, text: str, doc_metadata: Optional[dict] = None) -> dict[str, Any]:
        """
        Extract structured fields from a text chunk.

        Args:
            text         : The text to extract from.
            doc_metadata : Optional document-level context (title, source, etc.)

        Returns:
            A dict matching the schema fields.
        """
        if not text.strip():
            return {}

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(text, doc_metadata)

        logger.debug(f"Extracting with schema={self.schema_type!r}, model={self.model}")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content
        try:
            extracted = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction response as JSON: {e}")
            extracted = {"raw_response": raw}

        # Inject metadata fields
        if doc_metadata:
            extracted.setdefault("source", doc_metadata.get("source", ""))
            extracted.setdefault("title", doc_metadata.get("title", ""))

        logger.debug(f"Extracted {len(extracted)} fields")
        return extracted

    def _load_schema(self, schema_type: str) -> dict:
        schema_path = SCHEMAS_DIR / f"{schema_type}.json"
        if not schema_path.exists():
            logger.warning(f"Schema {schema_type!r} not found, using default")
            schema_path = SCHEMAS_DIR / "default.json"
        with open(schema_path) as f:
            return json.load(f)

    def _build_system_prompt(self) -> str:
        fields_desc = "\n".join(
            f"  - {name}: {meta.get('description', '')} (type: {meta.get('type', 'string')})"
            for name, meta in self.schema.get("fields", {}).items()
        )
        return f"""You are a precise scientific document intelligence extractor.
Your task is to extract structured information from the provided document text.

Extract the following fields and return a valid JSON object:
{fields_desc}

Rules:
- Extract ONLY what is explicitly present in the text
- Use null for fields that cannot be determined
- For list fields, return an array (empty array [] if none found)
- Be precise and factual — do not infer or hallucinate
- Return valid JSON only, no markdown fences
"""

    def _build_user_prompt(self, text: str, doc_metadata: Optional[dict]) -> str:
        context = ""
        if doc_metadata:
            context = f"\nDocument context: {json.dumps({k: v for k, v in doc_metadata.items() if isinstance(v, str)}, indent=2)}\n"
        return f"{context}\nDocument text:\n{text[:4000]}"
