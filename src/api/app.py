"""
FastAPI REST API for the Scientific Document Intelligence Pipeline.

Endpoints:
  POST /process       — submit a document for background processing
  GET  /jobs/{job_id} — poll job status / result
  GET  /jobs          — list all jobs
  GET  /search        — semantic search
  GET  /stats         — pipeline + storage statistics
  POST /clear         — clear all data (destructive)
  GET  /health        — liveness check

Run locally:  uvicorn src.api.app:app --host 0.0.0.0 --port 8080 --reload
Via Makefile: make api-local
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.pipeline.orchestrator import PipelineOrchestrator
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Scientific Document Intelligence Pipeline",
    description="REST API for processing PDFs, patents, and research documents.",
    version="2.0.0",
)

_orchestrator: Optional[PipelineOrchestrator] = None
_jobs: dict[str, dict[str, Any]] = {}
_pool = ThreadPoolExecutor(max_workers=8)


def _get_orchestrator() -> PipelineOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = PipelineOrchestrator()
    return _orchestrator


class ProcessRequest(BaseModel):
    source: str = Field(..., description="File path or HTTP/HTTPS URL to process.")
    mode: Optional[str]        = Field(None, description="full | extract_only | search_only | graph_only")
    schema_type: Optional[str] = Field(None, description="default | research_paper | patent")
    force: bool                = Field(False, description="Re-process even if already indexed.")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    submitted_at: float
    completed_at: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str]   = None


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/process", response_model=JobStatusResponse, tags=["Pipeline"])
def process_document(req: ProcessRequest):
    job_id = str(uuid.uuid4())[:12]
    submitted_at = time.time()
    _jobs[job_id] = {"job_id": job_id, "status": "queued", "submitted_at": submitted_at,
                     "completed_at": None, "elapsed_seconds": None, "result": None, "error": None}

    def _run():
        _jobs[job_id]["status"] = "running"
        start = time.time()
        try:
            orch = _get_orchestrator()
            if req.mode or req.schema_type:
                from src.pipeline.orchestrator import _deep_merge
                overrides: dict = {}
                if req.mode:        overrides.setdefault("pipeline", {})["mode"]        = req.mode
                if req.schema_type: overrides.setdefault("pipeline", {})["schema_type"] = req.schema_type
                _deep_merge(orch.config, overrides)
                orch._agent = None
            result = orch._process_source(req.source, force=req.force)
            _jobs[job_id].update(status=result.get("status","success"),
                                 completed_at=time.time(),
                                 elapsed_seconds=round(time.time()-start,2),
                                 result=result)
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            _jobs[job_id].update(status="error", completed_at=time.time(),
                                 elapsed_seconds=round(time.time()-start,2), error=str(e))

    _pool.submit(_run)
    return JobStatusResponse(job_id=job_id, status="queued", submitted_at=submitted_at)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Pipeline"])
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(**job)


@app.get("/jobs", tags=["Pipeline"])
def list_jobs(limit: int = Query(20, ge=1, le=200),
              status: Optional[str] = Query(None)):
    jobs = list(_jobs.values())
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    jobs.sort(key=lambda j: j["submitted_at"], reverse=True)
    return {"total": len(jobs), "jobs": jobs[:limit]}


@app.get("/search", tags=["Search"])
def semantic_search(q: str = Query(..., description="Natural language query"),
                    top_k: int = Query(10, ge=1, le=100)):
    try:
        results = _get_orchestrator().search(q, top_k=top_k)
        return {"query": q, "top_k": top_k, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["System"])
def get_stats():
    try:
        return _get_orchestrator().get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear", tags=["System"])
def clear_all(confirm: bool = Query(False)):
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass ?confirm=true to clear all data.")
    try:
        orch = _get_orchestrator()
        orch.chroma_store.delete_collection()
        orch.neo4j_store.clear_graph()
        orch.registry.clear()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
