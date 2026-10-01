"""parse() error handling and normalization, with the LLM call mocked.

No server needed: urllib.request.urlopen is replaced per test.
"""
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jira_rag import query_parse
from jira_rag.query_parse import ParserUnavailable, parse


def server_returning(content):
    """Fake urlopen whose chat/completions reply carries `content`."""
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def fake_urlopen(req, timeout=None):
        return io.BytesIO(body)

    return fake_urlopen


def test_server_down_raises(monkeypatch):
    calls = []

    def refused(req, timeout=None):
        calls.append(1)
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr(query_parse.urllib.request, "urlopen", refused)
    with pytest.raises(ParserUnavailable):
        parse("Which blocker bugs are still open?")
    assert len(calls) == 2  # first attempt + one retry


def test_invalid_json_fails_open(monkeypatch):
    monkeypatch.setattr(query_parse.urllib.request, "urlopen",
                        server_returning("not json at all"))
    form = parse("Which blocker bugs are still open?")
    assert form["parse_error"] is True
    assert form["priority"] is None and form["text_terms"] == []


def test_valid_output_normalized(monkeypatch):
    raw = {"priority": "Blocker", "issue_type": ["Bug", "Epic"], "open": True,
           "resolution": [], "created_from": None, "created_to": None,
           "text_terms": None}
    monkeypatch.setattr(query_parse.urllib.request, "urlopen",
                        server_returning("```json\n" + json.dumps(raw) + "\n```"))
    form = parse("Which blocker bugs are still open?")
    assert form["parse_error"] is False
    assert form["priority"] == ["Blocker"]   # bare string wrapped
    assert form["issue_type"] == ["Bug"]     # "Epic" dropped by the name guard
    assert form["resolution"] is None        # [] -> null
    assert form["text_terms"] == []          # null -> []
