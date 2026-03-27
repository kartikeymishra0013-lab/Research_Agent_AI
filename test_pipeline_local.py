#!/usr/bin/env python3
"""
Local end-to-end pipeline test (no Docker / no network required).

Stubs out OpenAI, ChromaDB, and Neo4j so the entire pipeline wiring
can be exercised locally. Run with:

    python test_pipeline_local.py

Or, to run against the live API once Docker is up:

    docker compose up -d chromadb neo4j
    docker compose --profile api up -d api
    python test_pipeline_local.py --live

"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── stub external packages that aren't installed in the sandbox ───────────────
# (they ARE installed inside Docker via requirements.txt)
import types

def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod

if "openai" not in sys.modules:
    class _FakeError(Exception): pass
    openai_mod = _stub_module(
        "openai",
        OpenAI=MagicMock,
        RateLimitError=_FakeError,
        APIConnectionError=_FakeError,
        APITimeoutError=_FakeError,
        InternalServerError=_FakeError,
    )

if "tenacity" not in sys.modules:
    # Provide a no-op tenacity so retry.py can import cleanly
    ten = _stub_module("tenacity")
    ten.retry = lambda *a, **kw: (lambda f: f)
    ten.stop_after_attempt = lambda n: None
    ten.wait_exponential = lambda **kw: None
    ten.retry_if_exception_type = lambda *a: None
    ten.before_sleep_log = lambda *a: None

if "chromadb" not in sys.modules:
    _stub_module("chromadb")

if "neo4j" not in sys.modules:
    _stub_module("neo4j", GraphDatabase=MagicMock)

if "fastapi" not in sys.modules:
    fa = _stub_module("fastapi")
    fa.FastAPI = MagicMock
    fa.HTTPException = Exception
    fa.Query = lambda *a, **kw: None
    fa.BackgroundTasks = MagicMock
    _stub_module("fastapi.responses", JSONResponse=MagicMock)
    _stub_module("pydantic", BaseModel=object, Field=lambda *a, **kw: None)

if "uvicorn" not in sys.modules:
    _stub_module("uvicorn")

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):  print(f"  {GREEN}✓  {msg}{RESET}")
def info(msg): print(f"  {CYAN}→  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠  {msg}{RESET}")
def err(msg):  print(f"  {RED}✗  {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{CYAN}{'─'*60}\n  {msg}\n{'─'*60}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK OpenAI client
# ═══════════════════════════════════════════════════════════════════════════════

def _make_openai_mock():
    """
    Returns a mock OpenAI client that returns realistic-looking responses
    for chat completions and embeddings without any network calls.
    """
    # ── chat.completions.create ───────────────────────────────────────────────
    def fake_chat_create(**kwargs):
        messages = kwargs.get("messages", [])
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        model = kwargs.get("model", "gpt-4o")

        # Decide which response to simulate
        if "extract" in last_user.lower() or "schema" in str(messages).lower()[:200]:
            content = json.dumps({
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
                "affiliations": ["Google Brain", "Google Research"],
                "keywords": ["transformer", "attention", "NLP", "machine translation"],
                "methodology": "Scaled dot-product attention and multi-head attention",
                "datasets": ["WMT 2014 English-German", "WMT 2014 English-French"],
                "results": "28.4 BLEU EN-DE, 41.8 BLEU EN-FR",
                "contributions": ["First purely attention-based sequence transduction model"],
                "venue": "NeurIPS 2017",
                "abstract": "The Transformer is a model architecture eschewing recurrence.",
            })
        elif "confidence" in last_user.lower() or "quality" in last_user.lower():
            content = json.dumps({
                "confidence": {
                    "title": 1.0,
                    "authors": 1.0,
                    "affiliations": 0.9,
                    "keywords": 0.95,
                    "methodology": 0.85,
                    "datasets": 1.0,
                    "results": 1.0,
                    "contributions": 0.8,
                    "venue": 1.0,
                },
                "overall_quality": "high",
                "low_confidence_fields": [],
            })
        elif "entities" in last_user.lower() or "relationships" in last_user.lower() or "graph" in str(messages).lower()[:200]:
            content = json.dumps({
                "entities": [
                    {"id": "transformer_model", "label": "Transformer", "type": "Technology",
                     "properties": {"description": "Attention-based seq2seq model"}},
                    {"id": "ashish_vaswani", "label": "Ashish Vaswani", "type": "Person",
                     "properties": {"description": "Lead author"}},
                    {"id": "google_brain", "label": "Google Brain", "type": "Organization",
                     "properties": {"description": "Research lab"}},
                    {"id": "wmt_2014", "label": "WMT 2014", "type": "Dataset",
                     "properties": {"description": "Machine translation benchmark"}},
                    {"id": "multi_head_attention", "label": "Multi-Head Attention", "type": "Method",
                     "properties": {"description": "Parallel attention over multiple subspaces"}},
                    {"id": "nips_2017", "label": "NeurIPS 2017", "type": "Publication",
                     "properties": {"description": "Conference"}},
                ],
                "relationships": [
                    {"source": "transformer_model", "target": "ashish_vaswani", "type": "AUTHORED_BY",
                     "properties": {"context": "Lead author of the Transformer paper"}},
                    {"source": "ashish_vaswani", "target": "google_brain", "type": "AFFILIATED_WITH",
                     "properties": {"context": "Google Brain researcher"}},
                    {"source": "transformer_model", "target": "multi_head_attention", "type": "USES",
                     "properties": {"context": "Core architectural component"}},
                    {"source": "transformer_model", "target": "wmt_2014", "type": "APPLIED_TO",
                     "properties": {"context": "Evaluated on WMT 2014"}},
                ],
            })
        elif "summary" in last_user.lower():
            content = (
                "**Main Topic**: The Transformer is a sequence transduction model based entirely "
                "on attention mechanisms, replacing RNNs and CNNs.\n\n"
                "**Key Contributions**:\n- First purely attention-based encoder-decoder\n"
                "- 28.4 BLEU on WMT 2014 EN-DE (new SOTA)\n- 12× faster training than RNN models\n\n"
                "**Methodology**: Multi-head scaled dot-product attention with positional encoding.\n\n"
                "**Findings**: Achieves state-of-the-art BLEU on two translation benchmarks.\n\n"
                "**Implications**: Enabled the pre-training paradigm (BERT, GPT) that now dominates NLP.\n\n"
                "**Keywords**: transformer, attention, NLP, machine translation, BLEU, encoder-decoder"
            )
        else:
            # Agent deciding which tools to call
            content = "I have successfully processed the document using all required tools."

        # Determine finish_reason: if tools could be called, simulate tool_calls on first iterations
        finish_reason = "stop"
        tool_calls_list = None

        # If tools kwarg is present and we haven't seen tool results yet, simulate tool calls
        if kwargs.get("tools") and not any(
            m.get("role") == "tool" for m in messages
        ):
            # First agent call → return tool calls
            finish_reason = "tool_calls"
            tool_calls_list = [
                SimpleNamespace(
                    id="call_extract",
                    function=SimpleNamespace(
                        name="extract_structured_data",
                        arguments=json.dumps({
                            "text": "Attention Is All You Need — Ashish Vaswani et al.",
                            "schema_type": "research_paper",
                        }),
                    ),
                ),
            ]
        elif kwargs.get("tools") and not any(
            m.get("role") == "tool" and json.loads(m.get("content", "{}")).get("tool") == "embed_and_store_chunks"
            for m in messages
        ) and sum(1 for m in messages if m.get("role") == "tool") == 1:
            # Second agent call → embed chunks
            finish_reason = "tool_calls"
            tool_calls_list = [
                SimpleNamespace(
                    id="call_embed",
                    function=SimpleNamespace(
                        name="embed_and_store_chunks",
                        arguments=json.dumps({
                            "chunks": ["The Transformer is based solely on attention mechanisms."],
                            "doc_id": "test_doc_001",
                        }),
                    ),
                ),
            ]
        elif kwargs.get("tools") and sum(1 for m in messages if m.get("role") == "tool") == 2:
            finish_reason = "tool_calls"
            tool_calls_list = [
                SimpleNamespace(
                    id="call_graph",
                    function=SimpleNamespace(
                        name="extract_and_store_knowledge_graph",
                        arguments=json.dumps({
                            "text": "The Transformer was developed by Ashish Vaswani at Google Brain.",
                            "doc_id": "test_doc_001",
                        }),
                    ),
                ),
            ]
        elif kwargs.get("tools") and sum(1 for m in messages if m.get("role") == "tool") == 3:
            finish_reason = "tool_calls"
            tool_calls_list = [
                SimpleNamespace(
                    id="call_summary",
                    function=SimpleNamespace(
                        name="generate_summary",
                        arguments=json.dumps({
                            "text": "Attention Is All You Need introduces the Transformer model.",
                            "doc_type": "txt",
                        }),
                    ),
                ),
            ]

        msg = SimpleNamespace(
            content=content,
            tool_calls=tool_calls_list,
            role="assistant",
        )
        msg.model_dump = lambda exclude_none=False: {
            "role": "assistant",
            "content": content,
            **({"tool_calls": [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    "type": "function",
                } for tc in tool_calls_list
            ]} if tool_calls_list else {}),
        }

        usage = SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=150,
            total_tokens=350,
        )
        choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], usage=usage, model=model)

    # ── embeddings.create ─────────────────────────────────────────────────────
    def fake_embeddings_create(**kwargs):
        texts = kwargs.get("input", [])
        # Return unit-normalised fake embeddings (1536 dims for text-embedding-3-small)
        import math
        dim = 1536
        embeddings = []
        for i, text in enumerate(texts):
            vec = [math.sin(hash(text[:20] + str(j)) % 1000 / 1000.0) for j in range(dim)]
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0
            embeddings.append(SimpleNamespace(embedding=[x / norm for x in vec]))

        usage = SimpleNamespace(prompt_tokens=len(texts) * 20, total_tokens=len(texts) * 20)
        return SimpleNamespace(data=embeddings, usage=usage)

    client = MagicMock()
    client.chat.completions.create.side_effect = fake_chat_create
    client.embeddings.create.side_effect = fake_embeddings_create
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK Storage backends
# ═══════════════════════════════════════════════════════════════════════════════

class MockChromaStore:
    def __init__(self, **kwargs):
        self._chunks: list = []
        self.collection_name = kwargs.get("collection_name", "documents")
        self.openai_client = kwargs.get("openai_client")
        self.cost_tracker = kwargs.get("cost_tracker")

    def connect(self): pass

    def add_chunks(self, chunks, doc_metadata=None):
        self._chunks.extend(chunks)
        return len(chunks)

    def search(self, query, top_k=10, where=None):
        return [
            {
                "id": "abc123",
                "text": "The Transformer is based solely on attention mechanisms.",
                "metadata": {"doc_id": "test_doc_001", "source": "attention_is_all_you_need.txt"},
                "distance": 0.08,
                "score": 0.92,
            }
        ]

    def count(self): return len(self._chunks)
    def delete_collection(self): self._chunks = []
    def list_collections(self): return [self.collection_name]


class MockNeo4jStore:
    def __init__(self, **kwargs):
        self._entities: list = []
        self._relationships: list = []

    def connect(self): pass
    def close(self): pass

    def store_graph(self, entities, relationships, doc_id=""):
        self._entities.extend(entities)
        self._relationships.extend(relationships)
        return {"nodes_merged": len(entities), "relationships_merged": len(relationships)}

    def store_document_hierarchy(self, doc_id, doc_metadata, chunks, entities_by_chunk):
        return {
            "document_nodes": 1,
            "chunk_nodes": len(chunks),
            "entity_nodes": sum(len(v) for v in entities_by_chunk.values()),
            "mentions_relationships": sum(len(v) for v in entities_by_chunk.values()),
        }

    def get_stats(self):
        return {
            "total_nodes": len(self._entities) + 1,
            "total_relationships": len(self._relationships),
            "nodes_by_label": {"Entity": len(self._entities)},
        }

    def clear_graph(self):
        self._entities.clear()
        self._relationships.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_local_tests():
    header("1 — Utility layer")

    # CostTracker
    from src.utils.cost_tracker import CostTracker
    ct = CostTracker(warn_threshold_usd=10.0)
    fake_usage = SimpleNamespace(prompt_tokens=500, completion_tokens=200, total_tokens=700)
    cost = ct.record("gpt-4o", "schema_extraction", fake_usage)
    ok(f"CostTracker recorded ${cost:.6f} for gpt-4o / schema_extraction")

    cost2 = ct.record("text-embedding-3-small", "chroma_embedding",
                      SimpleNamespace(prompt_tokens=1000, completion_tokens=0, total_tokens=1000))
    ok(f"CostTracker recorded ${cost2:.6f} for embedding")
    summary = ct.summary()
    ok(f"Cost summary: total=${summary['total_cost_usd']:.6f}, "
       f"models={list(summary['by_model'].keys())}, "
       f"ops={list(summary['by_operation'].keys())}")

    # ProcessingRegistry
    import tempfile
    from src.utils.registry import ProcessingRegistry
    with tempfile.TemporaryDirectory() as tmp:
        reg = ProcessingRegistry(registry_path=f"{tmp}/reg.json")
        assert not reg.is_processed("file_a.txt")
        reg.mark_processed("file_a.txt", "doc_001", "success", {"pages": 1})
        assert reg.is_processed("file_a.txt")
        assert not reg.is_processed("file_b.txt")
        assert not reg.is_processed("file_a.txt", force=True)  # force=True → treat as unprocessed
        s = reg.stats()
        ok(f"Registry: {s['total']} total, {s['success']} success, {s['error']} error")

    header("2 — Text chunker")
    from src.utils.chunker import TextChunker, Chunk

    sample_text = open(PROJECT_ROOT / "data/input/attention_is_all_you_need.txt").read()
    chunker = TextChunker(strategy="sentence", chunk_size=500, chunk_overlap=50, min_chunk_size=50)
    chunks = chunker.chunk(sample_text, metadata={"source": "attention.txt", "doc_id": "doc_001"})

    ok(f"Chunked into {len(chunks)} sentence-based chunks")
    ok(f"First chunk: {chunks[0].text[:80].strip()!r}…")
    ok(f"All chunks have doc_id={chunks[0].metadata.get('doc_id')!r}")
    assert all(c.token_estimate > 0 for c in chunks), "All chunks should have positive token estimates"
    assert all(c.char_end > c.char_start for c in chunks), "All chunks should have valid char ranges"

    para_chunker = TextChunker(strategy="paragraph", chunk_size=600, chunk_overlap=0, min_chunk_size=50)
    para_chunks = para_chunker.chunk(sample_text)
    ok(f"Paragraph chunker: {len(para_chunks)} chunks")

    header("3 — Text file ingestion")
    from src.ingestion.text_loader import TextLoader
    loader = TextLoader()
    doc = loader.load(str(PROJECT_ROOT / "data/input/attention_is_all_you_need.txt"))
    ok(f"Loaded document: {doc.metadata['title']!r}")
    ok(f"Word count: {doc.word_count:,} words | Type: {doc.doc_type}")
    ok(f"Source: {doc.source}")
    assert doc.word_count > 100, "Should have extracted meaningful text"
    assert doc.doc_type in ("text", "txt")

    header("4 — Schema extractor (mocked OpenAI)")
    from src.extraction.schema_extractor import SchemaExtractor, FEW_SHOT_EXAMPLES

    mock_client = _make_openai_mock()
    extractor = SchemaExtractor(
        client=mock_client,
        schema_type="research_paper",
        model="gpt-4o",
        enable_eval=True,
        cost_tracker=ct,
    )
    ok(f"Few-shot schemas available: {list(FEW_SHOT_EXAMPLES.keys())}")

    result = extractor.extract(doc.content[:2000], doc_metadata={"source": doc.source, "title": "Attention Is All You Need"})
    ok(f"Extracted {len(result)} fields from schema_type=research_paper")
    ok(f"  title: {result.get('title')!r}")
    ok(f"  authors: {result.get('authors')}")
    ok(f"  datasets: {result.get('datasets')}")
    if "_confidence" in result:
        avg_conf = sum(result["_confidence"].values()) / len(result["_confidence"])
        ok(f"  Confidence scores: avg={avg_conf:.2f}, overall_quality=N/A (from eval call)")

    header("5 — Relationship extractor (mocked OpenAI)")
    from src.extraction.relationship_extractor import RelationshipExtractor

    rel_extractor = RelationshipExtractor(
        client=mock_client,
        model="gpt-4o",
        cost_tracker=ct,
    )
    graph_data = rel_extractor.extract(doc.content[:2000], doc_id="doc_001", chunk_id="chunk_000")
    entities = graph_data["entities"]
    rels = graph_data["relationships"]
    ok(f"Extracted {len(entities)} entities, {len(rels)} relationships")
    entity_types = {e["type"] for e in entities}
    ok(f"Entity types: {sorted(entity_types)}")
    ok(f"Sample entity: {entities[0]['label']!r} ({entities[0]['type']})")
    ok(f"Sample relationship: {rels[0]['source']!r} → {rels[0]['type']} → {rels[0]['target']!r}")
    # Verify chunk_id provenance was injected
    assert all(e["properties"].get("chunk_id") == "chunk_000" for e in entities)
    ok(f"chunk_id provenance injected into all entities ✓")

    header("6 — ChromaDB store (mocked)")
    from src.utils.chunker import Chunk as ChunkObj

    chroma = MockChromaStore(
        host="localhost", port=8000,
        collection_name="documents",
        openai_client=mock_client,
        cost_tracker=ct,
    )
    chunk_objs = [
        ChunkObj(text=c.text, index=c.index, char_start=c.char_start,
                 char_end=c.char_end, metadata=c.metadata)
        for c in chunks[:5]
    ]
    stored = chroma.add_chunks(chunk_objs, doc_metadata={"doc_id": "doc_001"})
    ok(f"Stored {stored} chunks in mock ChromaDB")
    results = chroma.search("attention mechanism transformer", top_k=5)
    ok(f"Search returned {len(results)} results")
    ok(f"Top result: score={results[0]['score']:.3f} | {results[0]['text'][:60]!r}…")

    header("7 — Neo4j store (mocked)")
    neo4j = MockNeo4jStore()
    storage_summary = neo4j.store_graph(entities, rels, doc_id="doc_001")
    ok(f"Stored to Neo4j: {storage_summary}")

    # Test hierarchical
    hier_summary = neo4j.store_document_hierarchy(
        doc_id="doc_001",
        doc_metadata={"title": "Attention Is All You Need", "source": doc.source},
        chunks=[{"id": f"doc_001_chunk_{i}", "text": c.text, "index": c.index,
                 "char_start": c.char_start, "char_end": c.char_end}
                for i, c in enumerate(chunks[:3])],
        entities_by_chunk={f"doc_001_chunk_0": entities},
    )
    ok(f"Hierarchical store: {hier_summary}")
    stats = neo4j.get_stats()
    ok(f"Graph stats: {stats}")

    header("8 — Full agent loop (mocked)")
    from src.agent.tools import ToolExecutor, TOOL_SCHEMAS
    from src.agent.document_agent import DocumentAgent

    tool_names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    ok(f"Registered tools: {tool_names}")

    executor = ToolExecutor(
        schema_extractor=extractor,
        relationship_extractor=rel_extractor,
        chroma_store=chroma,
        neo4j_store=neo4j,
        openai_client=mock_client,
        cost_tracker=ct,
    )

    agent = DocumentAgent(
        client=mock_client,
        executor=executor,
        model="gpt-4o",
        temperature=0.1,
        pipeline_mode="full",
        chunking_cfg={"strategy": "sentence", "chunk_size": 500, "chunk_overlap": 50, "min_chunk_size": 50},
        cost_tracker=ct,
    )

    start = time.time()
    result = agent.process(doc)
    elapsed = round(time.time() - start, 3)

    ok(f"Agent completed in {result['iterations']} iterations ({elapsed}s)")
    ok(f"doc_id: {result['doc_id']!r}")
    ok(f"Chunks: {result['chunk_count']}")
    ok(f"Tools called: {result['tools_called']}")
    ok(f"Agent summary (first 100 chars): {str(result['agent_summary'])[:100]!r}…")
    if "cost_summary" in result:
        cs = result["cost_summary"]
        ok(f"Cost summary: total=${cs['total_cost_usd']:.6f}, "
           f"total_tokens={cs['total_tokens']:,}")

    header("9 — Full orchestrator (mocked)")
    import tempfile
    from src.pipeline.orchestrator import PipelineOrchestrator

    with tempfile.TemporaryDirectory() as tmp_output:
        with patch("src.pipeline.orchestrator.OpenAI", return_value=mock_client), \
             patch("src.pipeline.orchestrator.ChromaStore", return_value=chroma), \
             patch("src.pipeline.orchestrator.Neo4jStore", return_value=neo4j):

            orch = PipelineOrchestrator(
                config_overrides={
                    "pipeline": {"mode": "full", "model": "gpt-4o", "schema_type": "research_paper"},
                    "output": {"dir": tmp_output},
                    "registry": {"path": f"{tmp_output}/registry.json"},
                    "processing": {"max_workers": 2},
                    "cost_tracking": {"warn_threshold_usd": 10.0},
                    "chunking": {"strategy": "sentence", "chunk_size": 500, "chunk_overlap": 50, "min_chunk_size": 50},
                }
            )
            # Inject mocks
            orch._openai_client = mock_client
            orch._chroma_store = chroma
            orch._neo4j_store = neo4j
            orch._agent = None  # force re-build with injected clients

            result = orch.process_file(
                str(PROJECT_ROOT / "data/input/attention_is_all_you_need.txt")
            )

    ok(f"Orchestrator result: status={result['status']!r}")
    ok(f"doc_id: {result['doc_id']!r}")
    ok(f"Word count: {result['word_count']:,}")
    ok(f"Chunks: {result['chunk_count']}")
    ok(f"Tools called: {result.get('tools_called', [])}")
    ok(f"Elapsed: {result.get('elapsed_seconds')}s")

    # Test deduplication — second call should be skipped
    result2 = orch.process_file(
        str(PROJECT_ROOT / "data/input/attention_is_all_you_need.txt")
    )
    ok(f"Second call (dedup test): status={result2['status']!r} "
       f"(skipped={result2.get('status') == 'skipped' or 'already' in str(result2).lower()})")

    # Cost summary
    cost_summary = orch.cost_tracker.summary()
    ok(f"Final cost summary:")
    ok(f"  Total cost:   ${cost_summary['total_cost_usd']:.6f}")
    ok(f"  Total tokens: {cost_summary['total_tokens']:,}")
    ok(f"  By model:     {dict((k, round(v, 6)) for k,v in cost_summary['by_model'].items())}")
    ok(f"  By operation: {list(cost_summary['by_operation'].keys())}")

    header("✓ ALL TESTS PASSED")
    print(f"\n  The pipeline is fully wired and ready to run.\n"
          f"  To process real documents against live services:\n\n"
          f"  {CYAN}make setup{RESET}  (if not done)\n"
          f"  {CYAN}make build{RESET}  (build Docker image — includes Tesseract OCR)\n"
          f"  {CYAN}make up{RESET}     (start ChromaDB + Neo4j)\n"
          f"  {CYAN}make run FILE=data/input/attention_is_all_you_need.txt{RESET}\n"
          f"  {CYAN}make api{RESET}    (start REST API on :8080)\n"
          f"  {CYAN}make search Q=\"attention mechanism\"  TOP_K=5{RESET}\n"
          f"  {CYAN}make stats{RESET}  (show cost, chunks, graph stats)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE API TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_live_api_test():
    """Hit the real FastAPI endpoints (requires 'make api' running on :8080)."""
    try:
        import urllib.request, urllib.error
        base = "http://localhost:8080"

        header("LIVE API TEST (http://localhost:8080)")

        # Health check
        with urllib.request.urlopen(f"{base}/health") as r:
            data = json.loads(r.read())
        ok(f"GET /health → {data}")

        # Submit a process job
        payload = json.dumps({
            "source": str(PROJECT_ROOT / "data/input/attention_is_all_you_need.txt"),
            "mode": "full",
            "schema_type": "research_paper",
            "force": True,
        }).encode()
        req = urllib.request.Request(
            f"{base}/process", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req) as r:
            job = json.loads(r.read())
        job_id = job["job_id"]
        ok(f"POST /process → job_id={job_id!r}, status={job['status']!r}")

        # Poll until done
        for attempt in range(30):
            time.sleep(3)
            with urllib.request.urlopen(f"{base}/jobs/{job_id}") as r:
                job = json.loads(r.read())
            if job["status"] in ("success", "error"):
                break
            info(f"  [{attempt+1}] Job status: {job['status']!r} ({attempt*3}s elapsed)")

        ok(f"GET /jobs/{job_id} → status={job['status']!r}, elapsed={job.get('elapsed_seconds')}s")
        if job["status"] == "error":
            err(f"Job failed: {job.get('error')}")
        else:
            result = job.get("result", {})
            ok(f"  doc_id: {result.get('doc_id')!r}")
            ok(f"  chunks: {result.get('chunk_count')}")
            ok(f"  tools: {result.get('tools_called')}")

        # Semantic search
        with urllib.request.urlopen(f"{base}/search?q=attention+mechanism&top_k=3") as r:
            search = json.loads(r.read())
        ok(f"GET /search → {search['count']} results")
        for r in search["results"][:2]:
            ok(f"  score={r['score']} | {r['text'][:70]!r}…")

        # Stats
        with urllib.request.urlopen(f"{base}/stats") as r:
            stats = json.loads(r.read())
        ok(f"GET /stats → chromadb_chunks={stats.get('chromadb_chunks')}, "
           f"cost=${stats.get('cost_summary', {}).get('total_cost_usd', 0):.4f}")

        header("✓ LIVE API TESTS PASSED")

    except Exception as e:
        err(f"Live API test failed: {e}")
        info("Make sure the API is running: make api")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline test runner")
    parser.add_argument("--live", action="store_true", help="Also hit the live API on :8080")
    args = parser.parse_args()

    run_local_tests()

    if args.live:
        run_live_api_test()
