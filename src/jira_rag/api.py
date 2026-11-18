"""HTTP API for jira-rag.

Run from the repo root:
    HF_HOME=.hf JIRA_RAG_DEVICE=cpu .venv/bin/uvicorn api:app --app-dir src/jira_rag --port 8000 --env-file .env

HF_HOME=.hf keeps the embedder + reranker weights inside the repo (git-ignored).
JIRA_RAG_DEVICE=cpu leaves the GPU to the 27B llama-server (port 8092), which the
parser needs; with the 27B stopped, JIRA_RAG_DEVICE=cuda is ~23x faster
(eval wall time 1253 s CPU vs 55 s GPU, README section 3).
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

import retrieve

STRATEGY = "summary_desc"  # chosen configuration, see README section 3

app = FastAPI(title="jira-rag", version="0.2.0")


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
    hits: list[Hit]
    seconds: dict[str, float]


@app.get("/health")
def health():
    """Liveness check: answers as long as the process is up."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Parse the question into filters (27B), vector search, routed rerank."""
    results, form, where, stages = retrieve.search_rerank(req.question, STRATEGY, req.k)
    return SearchResponse(
        question=req.question,
        where=where,
        hits=[Hit(issue_key=key, score=float(score)) for key, score in results],
        seconds={name: round(sec, 3) for name, sec in stages.items()},
    )
