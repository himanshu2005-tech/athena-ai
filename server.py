"""
server.py — Athena FastAPI Bridge Server
=========================================
Exposes the engine.py RAG pipeline over HTTP with:
  • POST /query          — Non-streaming RAG response
  • GET  /query/stream   — SSE streaming token-by-token
  • GET  /stats          — Memory + model stats
  • POST /clear          — Wipe all memory
  • GET  /health         — Health check
  • CORS configured for React dev (localhost:5173 / localhost:3000)
"""

from __future__ import annotations

import json
import asyncio
import logging
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── Import your existing engine ──────────────────────────────────────────────
from engine import (
    rag_pipeline,
    clear_memory,
    get_memory_stats,
    get_memory_count,
)

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("athena.server")

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Athena RAG API",
    description="Ultra-production RAG engine — HyDE · RRF · FLARE · Self-RAG",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",   # CRA default
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    use_cot: bool = False
    use_flare: bool = True
    retrieval_k: int = Field(default=6, ge=0, le=20)
    max_search_results: int = Field(default=5, ge=1, le=10)
    conversation_history: list[dict] | None = None


class QueryResponse(BaseModel):
    response: str
    sources: list[str]
    used_web: bool
    chunk_count: int


class StatsResponse(BaseModel):
    vector_chunks: int
    bm25_chunks: int
    parent_docs: int
    embed_model: str
    rerank_model: str
    device: str


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/health")
async def health():
    """Quick liveness check."""
    return {"status": "ok", "chunks": get_memory_count()}


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return memory and model statistics."""
    return get_memory_stats()


@app.post("/clear")
async def clear():
    """Wipe all memory (ChromaDB + BM25 + parent store)."""
    try:
        clear_memory()
        return {"status": "cleared"}
    except Exception as exc:
        log.exception("clear_memory failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse)
async def query_blocking(req: QueryRequest):
    """
    Non-streaming RAG response.
    Runs the full pipeline (FLARE + Self-RAG) and returns when complete.
    """
    try:
        response, sources, used_web = await asyncio.to_thread(
            rag_pipeline,
            query=req.query,
            max_search_results=req.max_search_results,
            retrieval_k=req.retrieval_k,
            conversation_history=req.conversation_history,
            stream=False,
            use_flare=req.use_flare,
            use_cot=req.use_cot,
        )
        return QueryResponse(
            response=response,
            sources=sources,
            used_web=used_web,
            chunk_count=get_memory_count(),
        )
    except Exception as exc:
        log.exception("Blocking query failed — %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))


async def _sse_generator(req: QueryRequest) -> AsyncIterator[str]:
    """
    Wraps the synchronous token iterator from engine.py into an async SSE stream.
    Emits three event types:
      data: {"type": "token",  "content": "..."}
      data: {"type": "meta",   "sources": [...], "used_web": bool}
      data: {"type": "done"}
    """
    try:
        # Run blocking pipeline call in a thread pool
        token_iter, sources, used_web = await asyncio.to_thread(
            rag_pipeline,
            query=req.query,
            max_search_results=req.max_search_results,
            retrieval_k=req.retrieval_k,
            conversation_history=req.conversation_history,
            stream=True,
            use_flare=False,   # FLARE is non-streaming; disable for SSE path
            use_cot=req.use_cot,
        )

        # Stream tokens
        for token in token_iter:
            payload = json.dumps({"type": "token", "content": token})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0)   # yield control to event loop

        # Send metadata after all tokens
        meta = json.dumps({"type": "meta", "sources": sources, "used_web": used_web})
        yield f"data: {meta}\n\n"

    except Exception as exc:
        log.exception("SSE stream error — %s", req.query)
        err = json.dumps({"type": "error", "message": str(exc)})
        yield f"data: {err}\n\n"

    finally:
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    SSE streaming endpoint.
    Returns tokens as they are generated, then a final metadata event.
    """
    return StreamingResponse(
        _sse_generator(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )