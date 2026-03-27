"""
FastAPI REST API for the Scientific Document Intelligence Pipeline.

Endpoints:
  POST /process          — submit a file path or URL for processing (background job)
  GET  /jobs/{job_id}    — poll a background job for status / result
  GET  /search           — semantic search over indexed chunks
  GET  /stats            — pipeline + storage statistics
  POST /clear            — clear ChromaDB + Neo4j + registry (destructive)
  GET  /health           — liveness check

Run locally:
  uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload

Or via Makefile:
  make api-local
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Scientific Document Intelligence Pipeline",
    description=(
        "REST API for processing PDFs, patents, and research documents into "
        "structured datasets, knowledge graphs, and semantic search indexes."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Shared orchestrator (single instance — lazily initialized on first request)
_orchestrator: Optional[PipelineOrchestrator] = None

# Background job registry: job_id → {status, result, error, submitted_at}
_jobs: dict[str, dict[str, Any]] = {}
_thread_pool = ThreadPoolExecutor(max_workers=8)


def _get_orchestrator() -> PipelineOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator


# ── Request / Response models ─────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    source: str = Field(
        ...,
        description="Absolute file path or HTTP/HTTPS URL to process.",
        examples=["data/input/paper.pdf", "https://arxiv.org/pdf/1706.03762"],
    )
    mode: Optional[str] = Field(
        default=None,
        description="Pipeline mode override: full | extract_only | search_only | graph_only",
    )
    schema_type: Optional[str] = Field(
        default=None,
        description="Schema override: default | research_paper | patent",
    )
    force: bool = Field(
        default=False,
        description="Re-process even if already indexed.",
    )


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | running | success | error
    submitted_at: float
    completed_at: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query.")
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[dict] = Field(default=None, description="Metadata filter dict.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — returns 200 OK when the API is ready."""
    return {"status": "ok", "timestamp": time.time()}


@app.post("/process", response_model=JobStatusResponse, tags=["Pipeline"])
def process_document(req: ProcessRequest):
    """
    Submit a document for background processing.

    Returns a job_id that can be polled via GET /jobs/{job_id}.
    """
    job_id = str(uuid.uuid4())[:12]
    submitted_at = time.time()

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "submitted_at": submitted_at,
        "completed_at": None,
        "elapsed_seconds": None,
        "result": None,
        "error": None,
    }

    def _run_job():
        _jobs[job_id]["status"] = "running"
        start = time.time()
        try:
            orchestrator = _get_orchestrator()

            # Apply per-request config overrides
            overrides: dict[str, Any] = {}
            if req.mode:
                overrides.setdefault("pipeline", {})["mode"] = req.mode
            if req.schema_type:
                overrides.setdefault("pipeline", {})["schema_type"] = req.schema_type

            if overrides:
                # Merge overrides into the orchestrator config for this request
                from src.pipeline.orchestrator import _deep_merge
                _deep_merge(orchestrator.config, overrides)
                orchestrator._agent = None  # force re-build with new config

            result = orchestrator._process_source(req.source, force=req.force)
            elapsed = round(time.time() - start, 2)
            _jobs[job_id].update(
                status=result.get("status", "success"),
                completed_at=time.time(),
                elapsed_seconds=elapsed,
                result=result,
            )
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            _jobs[job_id].update(
                status="error",
                completed_at=time.time(),
                elapsed_seconds=elapsed,
                error=str(e),
            )

    _thread_pool.submit(_run_job)

    return JobStatusResponse(
        job_id=job_id,
        status="queued",
        submitted_at=submitted_at,
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Pipeline"])
def get_job_status(job_id: str):
    """Poll a background processing job for status and result."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(**job)


@app.get("/search", tags=["Search"])
def semantic_search(
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(default=10, ge=1, le=100, description="Number of results"),
):
    """
    Run semantic search over all indexed document chunks.

    Returns the top-k most similar chunks with metadata and similarity scores.
    """
    try:
        orchestrator = _get_orchestrator()
        results = orchestrator.search(q, top_k=top_k)
        return {
            "query": q,
            "top_k": top_k,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["System"])
def get_stats():
    """Return pipeline statistics: chunk counts, graph size, output files, and API cost."""
    try:
        orchestrator = _get_orchestrator()
        return orchestrator.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear", tags=["System"])
def clear_all(confirm: bool = Query(False, description="Must be true to proceed")):
    """
    Clear ChromaDB collection, Neo4j graph, and processing registry.

    **Destructive** — pass ?confirm=true to execute.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Pass ?confirm=true to clear all data. This action is irreversible.",
        )
    try:
        orchestrator = _get_orchestrator()
        orchestrator.chroma_store.delete_collection()
        orchestrator.neo4j_store.clear_graph()
        orchestrator.registry.clear()
        return {"status": "cleared", "message": "ChromaDB, Neo4j, and registry have been cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs", tags=["Pipeline"])
def list_jobs(
    limit: int = Query(default=20, ge=1, le=200),
    status: Optional[str] = Query(default=None, description="Filter by status"),
):
    """List recent background jobs."""
    all_jobs = list(_jobs.values())
    if status:
        all_jobs = [j for j in all_jobs if j["status"] == status]
    # Sort by submission time, most recent first
    all_jobs.sort(key=lambda j: j["submitted_at"], reverse=True)
    return {
        "total": len(all_jobs),
        "jobs": all_jobs[:limit],
    }
