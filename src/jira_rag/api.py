"""HTTP API for jira-rag.

Run from the repo root:
    .venv/bin/uvicorn api:app --app-dir src/jira_rag --port 8000
"""
from fastapi import FastAPI

app = FastAPI(title="jira-rag", version="0.1.0")


@app.get("/health")
def health():
    """Liveness check: answers as long as the process is up."""
    return {"status": "ok"}
