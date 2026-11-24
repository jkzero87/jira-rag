"""HTTP API for jira-rag.

Run from the repo root:
    HF_HOME=.hf JIRA_RAG_DEVICE=cpu .venv/bin/uvicorn api:app --app-dir src/jira_rag --port 8000 --env-file .env

HF_HOME=.hf keeps the embedder + reranker weights inside the repo (git-ignored).
JIRA_RAG_DEVICE=cpu leaves the GPU to the 27B llama-server (port 8092), which the
parser needs; with the 27B stopped, JIRA_RAG_DEVICE=cuda is ~23x faster
(eval wall time 1253 s CPU vs 55 s GPU, README section 3).
"""
import hashlib
import os
import secrets
import time
import urllib.error

from fastapi import Depends, FastAPI, HTTPException, Request
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import generate
import retrieve

STRATEGY = "summary_desc"  # chosen configuration, see README section 3

app = FastAPI(title="jira-rag", version="0.3.0")

API_KEY = os.environ.get("JIRA_RAG_API_KEY", "")  # from .env via uvicorn --env-file

# 30 requests/minute per valid API key; keyed by the sha256 of the X-API-Key
# header (hashed so the limiter's key space never holds the raw key). Failed
# authentication has its own per-IP budget (FAILED_AUTH_LIMIT below).
# /v1/health carries no key and is not rate limited.
limiter = Limiter(key_func=lambda request: hashlib.sha256(
    request.headers.get("X-API-Key", "no-key").encode()).hexdigest())
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Failed authentication (missing or wrong key): 10 per minute per client IP.
# The budget is checked BEFORE the key is validated, so an IP that used it up
# gets 429 until the window ends; requests with a valid key do not consume it.
FAILED_AUTH_LIMIT = parse("10/minute")
failed_auth_storage = MemoryStorage()
failed_auth_limiter = FixedWindowRateLimiter(failed_auth_storage)


def require_api_key(request: Request) -> str:
    """Rate-limit failed attempts per IP (429), then reject missing or wrong
    X-API-Key headers with 401. The compare is constant-time on UTF-8 bytes,
    so a non-ASCII header gives 401, not 500."""
    ip = get_remote_address(request)
    if not failed_auth_limiter.test(FAILED_AUTH_LIMIT, ip):
        raise HTTPException(status_code=429, detail="too many failed authentication attempts")
    provided = request.headers.get("X-API-Key", "").encode("utf-8")
    if not API_KEY or not secrets.compare_digest(provided, API_KEY.encode("utf-8")):
        failed_auth_limiter.hit(FAILED_AUTH_LIMIT, ip)
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
