"""HTTP API for jira-rag.

Run from the repo root:
    HF_HOME=.hf JIRA_RAG_DEVICE=cpu .venv/bin/uvicorn api:app --app-dir src/jira_rag --port 8000 --env-file .env

HF_HOME=.hf keeps the embedder + reranker weights inside the repo (git-ignored).
JIRA_RAG_DEVICE=cpu leaves the GPU to the 27B llama-server (port 8092), which the
parser needs; with the 27B stopped, JIRA_RAG_DEVICE=cuda is ~23x faster
(eval wall time 1253 s CPU vs 55 s GPU, README section 3).
"""
import os
import time
import urllib.error

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

import generate
import retrieve

STRATEGY = "summary_desc"  # chosen configuration, see README section 3

app = FastAPI(title="jira-rag", version="0.3.0")

API_KEY = os.environ.get("JIRA_RAG_API_KEY", "")  # from .env via uvicorn --env-file

# 30 requests/minute per API key; keyed by the X-API-Key header so that
# /v1/health (which carries no key) is never rate limited.
limiter = Limiter(key_func=lambda request: request.headers.get("X-API-Key", "no-key"))
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def require_api_key(request: Request) -> str:
    """Reject missing or wrong X-API-Key headers with 401."""
    if not API_KEY or request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return API_KEY


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500,
                          examples=["Which ZooKeeper bumps in Spark were done to fix known security holes?"])
    k: int = Field(default=10, ge=1, le=50)


class Hit(BaseModel):
    issue_key: str
    score: float


class SearchResponse(BaseModel):
    question: str
    where: str
    parse_error: bool  # True when the 27B parser failed and no filters were applied
    hits: list[Hit]
    seconds: dict[str, float]


@app.get("/v1/health")
def health():
    """Liveness check: answers as long as the process is up. No API key needed."""
    return {"status": "ok"}


@app.post("/v1/search", response_model=SearchResponse)
@limiter.limit("30/minute")
def search(req: SearchRequest, request: Request, _key: str = Depends(require_api_key)):
    """Parse the question into filters (27B), vector search, routed rerank."""
    results, form, where, stages = retrieve.search_rerank(req.question, STRATEGY, req.k)
    return SearchResponse(
        question=req.question,
        where=where,
        parse_error=bool(form.get("parse_error", False)),
        hits=[Hit(issue_key=key, score=float(score)) for key, score in results],
        seconds={name: round(sec, 3) for name, sec in stages.items()},
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500,
                          examples=["Which ZooKeeper bumps in Spark were done to fix known security holes?"])
    k: int = Field(default=10, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]           # cited keys that WERE in the context
    invented_citations: list[str]  # cited keys that were NOT in the context (should be empty)
    context_keys: list[str]
    parse_error: bool
    seconds: dict[str, float]


@app.post("/v1/ask", response_model=AskResponse)
@limiter.limit("30/minute")
def ask(req: AskRequest, request: Request, _key: str = Depends(require_api_key)):
    """Retrieve the top-k issues, then let the 27B answer from them with citations."""
    results, form, where, stages = retrieve.search_rerank(req.question, STRATEGY, req.k)
    keys = [key for key, _ in results]
    t = time.monotonic()
    contexts = generate.fetch_contexts(keys)
    try:
        text = generate.answer(req.question, contexts)
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"answer model unavailable: {exc}") from exc
    stages["generate"] = time.monotonic() - t
    cited = generate.cited_keys(text)
    return AskResponse(
        question=req.question,
        answer=text,
        citations=[c for c in cited if c in keys],
        invented_citations=[c for c in cited if c not in keys],
        context_keys=keys,
        parse_error=bool(form.get("parse_error", False)),
        seconds={name: round(sec, 3) for name, sec in stages.items()},
    )
