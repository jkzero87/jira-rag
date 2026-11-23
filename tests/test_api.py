"""HTTP-layer tests for the jira-rag /v1 API (prefix, X-API-Key auth, rate limit).

The real retrieve module loads torch + two models and talks to Postgres and
the 27B. Here it is replaced by a fake ("stunt double") so these tests check
only the API itself: routing, auth and rate limiting.
Run:  .venv/bin/python -m pytest -q
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")  # JIRA_RAG_API_KEY must come from the same .env as production

import os

API_KEY = os.environ["JIRA_RAG_API_KEY"]
AUTH = {"X-API-Key": API_KEY}
SEARCH_JSON = {"question": "ZooKeeper security bumps", "k": 3}

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


def test_health_200_without_key():
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_missing_key_is_401():
    r = client.post("/v1/search", json=SEARCH_JSON)
    assert r.status_code == 401


def test_wrong_key_is_401():
    r = client.post("/v1/search", json=SEARCH_JSON, headers={"X-API-Key": "definitely-wrong"})
    assert r.status_code == 401


def test_good_key_is_200():
    r = client.post("/v1/search", json=SEARCH_JSON, headers=AUTH)
    assert r.status_code == 200, r.text
    assert [h["issue_key"] for h in r.json()["hits"]] == ["SPARK-1", "SPARK-2", "SPARK-3"]


def test_31st_request_in_same_minute_is_429():
    api.limiter.reset()  # deterministic window, independent of other tests
    statuses = [client.post("/v1/search", json=SEARCH_JSON, headers=AUTH).status_code
                for _ in range(31)]
    assert statuses[:30] == [200] * 30
    assert statuses[30] == 429
