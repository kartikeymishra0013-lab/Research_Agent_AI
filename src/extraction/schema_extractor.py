"""
Schema-aware structured data extractor using GPT-4o.
Upgrades: few-shot examples, self-evaluation confidence scoring, retry, cost tracking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger
from src.utils.retry import with_retry

logger = get_logger(__name__)

SCHEMAS_DIR = Path(__file__).parent.parent.parent / "config" / "schemas"

FEW_SHOT_EXAMPLES: dict[str, str] = {
    "research_paper": """
EXAMPLE INPUT: "We propose BERT by Jacob Devlin et al. from Google AI Language, achieving GLUE 80.5%."
EXAMPLE OUTPUT: {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "authors": ["Jacob Devlin"], "affiliations": ["Google AI Language"], "keywords": ["BERT", "NLP", "transformers"], "results": "GLUE 80.5%", "venue": null}
""",
    "patent": """
EXAMPLE INPUT: "US Patent 10,123,456. Method for NLP queries. Inventors: John Smith. Assignee: TechCorp."
EXAMPLE OUTPUT: {"patent_number": "US 10,123,456", "title": "Method for NLP queries", "inventors": ["John Smith"], "assignee": "TechCorp", "ipc_classification": [], "filing_date": null}
""",
    "default": """
EXAMPLE INPUT: "Q3 2024 Report — AI Division. Authors: Research Team. Key finding: 94% accuracy."
EXAMPLE OUTPUT: {"title": "Q3 2024 Report — AI Division", "authors": ["Research Team"], "summary": "Phase 1 completed with 94% accuracy", "keywords": ["AI", "accuracy"], "findings": ["94% accuracy"]}
""",
}


class SchemaExtractor:
    """
    Extracts structured data from text using GPT-4o with few-shot examples,
    self-evaluation confidence scoring, retry, and cost tracking.
    """

    def __init__(
        self,
        client,
        schema_type: str = "default",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        custom_schema: Optional[dict] = None,
        enable_eval: bool = True,
        cost_tracker=None,
    ):
        self.client = client
        self.schema_type = schema_type
        self.model = model
        self.temperature = temperature
        self.schema = custom_schema or self._load_schema(schema_type)
        self.enable_eval = enable_eval
        self.cost_tracker = cost_tracker

    def extract(self, text: str, doc_metadata: Optional[dict] = None) -> dict[str, Any]:
        if not text.strip():
            return {}

        @with_retry(max_attempts=3, min_wait=1, max_wait=30)
        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user",   "content": self._build_user_prompt(text, doc_metadata)},
                ],
            )

        response = _call()
        if self.cost_tracker:
            self.cost_tracker.record(self.model, "schema_extraction", response.usage)

        raw = response.choices[0].message.content
        try:
            extracted = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction response: {e}")
            extracted = {"raw_response": raw}

        if doc_metadata:
            extracted.setdefault("source", doc_metadata.get("source", ""))
            extracted.setdefault("title",  doc_metadata.get("title", ""))

        if self.enable_eval and extracted and "raw_response" not in extracted:
            try:
                extracted["_confidence"] = self._evaluate_confidence(text, extracted)
            except Exception as e:
                logger.warning(f"Confidence evaluation failed: {e}")

        logger.debug(f"Extracted {len(extracted)} fields")
        return extracted

    def _evaluate_confidence(self, original_text: str, extracted: dict) -> dict[str, float]:
        fields = {k: v for k, v in extracted.items() if not k.startswith("_") and k != "source"}
        if not fields:
            return {}

        prompt = (
            "Rate each extracted field 0.0-1.0 (1.0=directly stated, 0.5=implied, 0.0=not found).\n"
            "Return ONLY JSON: {\"field_name\": score, ...}\n\n"
            f"SOURCE TEXT:\n{original_text[:2000]}\n\n"
            f"EXTRACTED:\n{json.dumps(fields, indent=2, default=str)[:2000]}"
        )

        @with_retry(max_attempts=2, min_wait=1, max_wait=20)
        def _call():
            return self.client.chat.completions.create(
                model=self.model, temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )

        response = _call()
        if self.cost_tracker:
            self.cost_tracker.record(self.model, "extraction_evaluation", response.usage)

        scores = json.loads(response.choices[0].message.content)
        return {k: max(0.0, min(1.0, float(v))) for k, v in scores.items() if isinstance(v, (int, float))}

    def _load_schema(self, schema_type: str) -> dict:
        path = SCHEMAS_DIR / f"{schema_type}.json"
        if not path.exists():
            logger.warning(f"Schema {schema_type!r} not found, using default")
            path = SCHEMAS_DIR / "default.json"
        with open(path) as f:
            return json.load(f)

    def _build_system_prompt(self) -> str:
        fields_desc = "\n".join(
            f"  - {name}: {meta.get('description','')} (type: {meta.get('type','string')})"
            for name, meta in self.schema.get("fields", {}).items()
        )
        few_shot = FEW_SHOT_EXAMPLES.get(self.schema_type, FEW_SHOT_EXAMPLES["default"])
        return (
            "You are a precise scientific document intelligence extractor.\n"
            "Extract the following fields and return a valid JSON object:\n"
            f"{fields_desc}\n\n"
            "Rules:\n"
            "- Extract ONLY what is explicitly present\n"
            "- Use null for absent fields; [] for empty lists\n"
            "- Be precise, do not hallucinate\n"
            "- Return valid JSON only, no markdown\n"
            f"\n{few_shot}"
        )

    def _build_user_prompt(self, text: str, doc_metadata: Optional[dict]) -> str:
        ctx = ""
        if doc_metadata:
            ctx = f"\nDocument context: {json.dumps({k:v for k,v in doc_metadata.items() if isinstance(v,str)}, indent=2)}\n"
        return f"{ctx}\nDocument text:\n{text[:4000]}"
