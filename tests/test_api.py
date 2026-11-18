"""HTTP-layer tests for the jira-rag API.

The real retrieve module loads torch + two models and talks to Postgres and
the 27B. Here it is replaced by a fake ("stunt double") so these tests check
only the API itself: routing, input validation and the response shape.
Run:  .venv/bin/python -m pytest -q
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

CALLS = []


def fake_search_rerank(question, strategy, k):
    CALLS.append((question, strategy, k))
    hits = [(f"SPARK-{i}", 1.0 / i) for i in range(1, k + 1)]
    return hits, {"parse_error": False}, "", {"parse": 0.1, "vector": 0.2, "rerank": 0.3}


fake = types.ModuleType("retrieve")
fake.search_rerank = fake_search_rerank
sys.modules["retrieve"] = fake  # must happen before "import api"

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_search_response_shape():
    r = client.post("/search", json={"question": "ZooKeeper security bumps", "k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [h["issue_key"] for h in body["hits"]] == ["SPARK-1", "SPARK-2", "SPARK-3"]
    assert body["parse_error"] is False
    assert set(body["seconds"]) == {"parse", "vector", "rerank"}
    assert CALLS[-1] == ("ZooKeeper security bumps", "summary_desc", 3)


def test_default_k_is_10():
    client.post("/search", json={"question": "default k please"})
    assert CALLS[-1][2] == 10


def test_parse_error_is_reported():
    fake.search_rerank = lambda q, s, k: ([], {"parse_error": True}, "", {})
    try:
        r = client.post("/search", json={"question": "anything at all"})
        assert r.status_code == 200
        assert r.json()["parse_error"] is True
    finally:
        fake.search_rerank = fake_search_rerank


def test_validation_rejects_bad_input():
    r = client.post("/search", json={"question": "", "k": 999})
    assert r.status_code == 422
    assert {e["loc"][-1] for e in r.json()["detail"]} == {"question", "k"}


import urllib.error  # noqa: E402

import generate  # noqa: E402


def test_ask_splits_real_and_invented_citations(monkeypatch):
    monkeypatch.setattr(generate, "fetch_contexts", lambda keys: [{"issue_key": k} for k in keys])
    monkeypatch.setattr(generate, "answer",
                        lambda q, ctx: "Fixed in [SPARK-1] and [SPARK-2], also [SPARK-999].")
    r = client.post("/ask", json={"question": "what was fixed?", "k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["context_keys"] == ["SPARK-1", "SPARK-2", "SPARK-3"]
    assert body["citations"] == ["SPARK-1", "SPARK-2"]
    assert body["invented_citations"] == ["SPARK-999"]
    assert "generate" in body["seconds"]


def test_ask_returns_503_when_model_is_down(monkeypatch):
    monkeypatch.setattr(generate, "fetch_contexts", lambda keys: [])

    def down(q, ctx):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(generate, "answer", down)
    r = client.post("/ask", json={"question": "is the model there?"})
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]


def test_ask_validation():
    r = client.post("/ask", json={"question": "ok question", "k": 21})
    assert r.status_code == 422
