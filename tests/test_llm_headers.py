"""Assert-based tests for query_parse.llm_headers().

No framework: run with  .venv/bin/python tests/test_llm_headers.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

from query_parse import llm_headers


def test_no_key_no_auth_header():
    os.environ.pop("LLAMA_API_KEY", None)
    h = llm_headers()
    assert h == {"Content-Type": "application/json"}, h


def test_blank_key_is_ignored():
    os.environ["LLAMA_API_KEY"] = "   "
    h = llm_headers()
    assert "Authorization" not in h, h


def test_key_sends_bearer():
    os.environ["LLAMA_API_KEY"] = "s3cret"
    h = llm_headers()
    assert h["Authorization"] == "Bearer s3cret", h
    assert h["Content-Type"] == "application/json", h


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    os.environ.pop("LLAMA_API_KEY", None)
