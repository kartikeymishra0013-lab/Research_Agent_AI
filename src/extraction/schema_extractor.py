"""
Schema-aware structured data extractor using GPT-4o.

Enhancements over baseline:
  - Few-shot worked examples in system prompt for higher accuracy
  - Self-evaluation: GPT-4o rates confidence (0–1) for each extracted field
  - Retry with exponential backoff via `with_retry`
  - Cost tracking via CostTracker
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from src.utils.logger import get_logger
from src.utils.retry import with_retry

logger = get_logger(__name__)

SCHEMAS_DIR = Path(__file__).parent.parent.parent / "config" / "schemas"

# ── Few-shot examples injected into the system prompt ────────────────────────
# One compact example per schema type keeps the prompt focused and accurate.
FEW_SHOT_EXAMPLES: dict[str, str] = {
    "research_paper": """
EXAMPLE INPUT:
"We propose BERT: Bidirectional Encoder Representations from Transformers. BERT obtains
new state-of-the-art results on eleven NLP tasks, including GLUE score of 80.5% and
SQuAD 2.0 test F1 of 83.1. Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.
Google AI Language. Published at NAACL 2019. Code: https://github.com/google-research/bert"

EXAMPLE OUTPUT:
{
  "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
  "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
  "affiliations": ["Google AI Language"],
  "abstract": null,
  "keywords": ["BERT", "NLP", "transformers", "language model", "pre-training"],
  "methodology": "Bidirectional pre-training using masked language model and next sentence prediction",
  "datasets": ["GLUE", "SQuAD 2.0"],
  "results": "GLUE 80.5%, SQuAD 2.0 test F1 83.1%",
  "contributions": ["State-of-the-art on 11 NLP tasks", "Bidirectional transformer pre-training"],
  "code_url": "https://github.com/google-research/bert",
  "venue": "NAACL 2019"
}
""",
    "patent": """
EXAMPLE INPUT:
"US Patent 10,123,456. Method and system for processing natural language queries using
neural networks. Inventors: John Smith, Jane Doe. Assignee: TechCorp Inc.
IPC: G06F 40/56. Filed: 2021-03-15. The invention relates to transformer-based NLP
for enterprise search applications."

EXAMPLE OUTPUT:
{
  "patent_number": "US 10,123,456",
  "title": "Method and system for processing natural language queries using neural networks",
  "inventors": ["John Smith", "Jane Doe"],
  "assignee": "TechCorp Inc.",
  "ipc_classification": ["G06F 40/56"],
  "filing_date": "2021-03-15",
  "technical_field": "Natural language processing using neural networks",
  "abstract": null,
  "claims": [],
  "prior_art": [],
  "applications": ["Enterprise search"]
}
""",
    "default": """
EXAMPLE INPUT:
"Q3 2024 Progress Report — AI Research Division. Authors: Research Team.
Organization: ACME Labs. Date: October 2024.
Summary: We completed Phase 1 of the neural document pipeline project,
achieving 94% accuracy on extraction benchmarks. Key finding: semantic
chunking outperforms fixed-size chunking by 18%."

EXAMPLE OUTPUT:
{
  "title": "Q3 2024 Progress Report — AI Research Division",
  "document_type": "report",
  "authors": ["Research Team"],
  "date": "October 2024",
  "organization": "ACME Labs",
  "summary": "Phase 1 of neural document pipeline completed with 94% extraction accuracy.",
  "main_topics": ["document pipeline", "neural extraction", "semantic chunking"],
  "key_entities": ["ACME Labs", "AI Research Division"],
  "keywords": ["AI", "document pipeline", "semantic chunking", "accuracy"],
  "findings": ["94% accuracy on extraction benchmarks", "Semantic chunking outperforms fixed-size by 18%"],
  "recommendations": []
}
""",
}


class SchemaExtractor:
    """
    Extracts structured data from text using a GPT-4o prompt
    driven by a JSON schema definition.

    Args:
        client        : Initialized OpenAI client.
        schema_type   : Schema name matching a file in config/schemas/.
        model         : OpenAI model (default: gpt-4o).
        temperature   : Model temperature (default: 0.0 for determinism).
        custom_schema : Override schema dict instead of loading from file.
        enable_eval   : If True, run a second GPT-4o call to score field confidence.
        cost_tracker  : Optional CostTracker for recording API costs.

    Usage:
        extractor = SchemaExtractor(client=openai_client, schema_type="research_paper",
                                    cost_tracker=tracker)
        result = extractor.extract(chunk_text)
        # result["_confidence"] contains per-field scores if enable_eval=True
    """

    def __init__(
        self,
        client: OpenAI,
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
        """
        Extract structured fields from a text chunk.

        Args:
            text         : The text to extract from.
            doc_metadata : Optional document-level context (title, source, etc.)

        Returns:
            A dict matching the schema fields, with an optional ``_confidence``
            sub-dict (field → 0–1 score) when ``enable_eval=True``.
        """
        if not text.strip():
            return {}

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(text, doc_metadata)

        logger.debug(f"Extracting with schema={self.schema_type!r}, model={self.model}")

        @with_retry(max_attempts=3, min_wait=1, max_wait=30)
        def _call_extraction():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        response = _call_extraction()

        if self.cost_tracker:
            self.cost_tracker.record(
                model=self.model,
                operation="schema_extraction",
                usage=response.usage,
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

        # Self-evaluation: score each field's confidence
        if self.enable_eval and extracted and "raw_response" not in extracted:
            try:
                confidence = self._evaluate_confidence(text, extracted)
                extracted["_confidence"] = confidence
            except Exception as e:
                logger.warning(f"Confidence evaluation failed: {e}")

        logger.debug(f"Extracted {len(extracted)} fields")
        return extracted

    def _evaluate_confidence(self, original_text: str, extracted: dict) -> dict[str, float]:
        """
        Ask GPT-4o to score how confident it is that each extracted field
        is accurate and well-grounded in the source text (0.0 = unsupported,
        1.0 = directly stated).

        Returns a dict mapping field name → confidence score.
        """
        # Exclude internal / metadata keys from evaluation
        fields_to_eval = {
            k: v for k, v in extracted.items()
            if not k.startswith("_") and k not in ("source",)
        }

        if not fields_to_eval:
            return {}

        prompt = f"""You are a quality-assurance agent for document extraction.

Given the SOURCE TEXT and EXTRACTED DATA below, rate each extracted field with a
confidence score between 0.0 and 1.0:
  1.0 = field value is directly and clearly stated in the source text
  0.5 = field value is implied or partially supported
  0.0 = field value is not found in the source text (likely hallucinated)

Respond with ONLY a JSON object mapping each field name to its score.
Example: {{"title": 1.0, "authors": 0.9, "abstract": 0.5, "code_url": 0.0}}

SOURCE TEXT (first 2000 chars):
{original_text[:2000]}

EXTRACTED DATA:
{json.dumps(fields_to_eval, indent=2, default=str)[:3000]}"""

        @with_retry(max_attempts=2, min_wait=1, max_wait=20)
        def _call_eval():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )

        response = _call_eval()

        if self.cost_tracker:
            self.cost_tracker.record(
                model=self.model,
                operation="extraction_evaluation",
                usage=response.usage,
            )

        raw = response.choices[0].message.content
        scores = json.loads(raw)
        # Clamp all scores to [0, 1]
        return {k: max(0.0, min(1.0, float(v))) for k, v in scores.items() if isinstance(v, (int, float))}

    # ─── Schema & prompt builders ─────────────────────────────────────────────

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
        few_shot = FEW_SHOT_EXAMPLES.get(self.schema_type, FEW_SHOT_EXAMPLES["default"])

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

{few_shot}"""

    def _build_user_prompt(self, text: str, doc_metadata: Optional[dict]) -> str:
        context = ""
        if doc_metadata:
            context = (
                f"\nDocument context: "
                f"{json.dumps({k: v for k, v in doc_metadata.items() if isinstance(v, str)}, indent=2)}\n"
            )
        return f"{context}\nDocument text:\n{text[:4000]}"
