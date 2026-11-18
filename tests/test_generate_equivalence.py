"""The API's generation (src/jira_rag/generate.py) must build EXACTLY the same
prompt as the phase-3 eval (eval/generate_answers.py), or the eval numbers in
the README stop describing what /ask does.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))
sys.path.insert(0, str(ROOT / "eval"))

import generate  # noqa: E402
import generate_answers as ev  # noqa: E402

CONTEXTS = [
    {"issue_key": "SPARK-1", "summary": "Bump ZooKeeper", "status": "Resolved",
     "resolution": "Fixed", "priority": "Major", "issue_type": "Improvement",
     "created": "2019-09-01T10:00:00+00:00", "description": "x" * 2000},
    {"issue_key": "SPARK-2", "summary": "No description", "status": "Open",
     "resolution": None, "priority": None, "issue_type": "Bug",
     "created": None, "description": None},
]


def test_same_settings():
    assert generate.SYSTEM == ev.SYSTEM
    assert generate.CONTEXT_FIELDS == ev.CONTEXT_FIELDS
    assert generate.TEMPERATURE == ev.TEMPERATURE
    assert generate.MAX_TOKENS == ev.MAX_TOKENS
    assert generate.CITE_RE.pattern == ev.CITE_RE.pattern


def test_same_prompt_byte_for_byte():
    q = "Which ZooKeeper bumps fixed CVEs?"
    assert generate.build_user_prompt(q, CONTEXTS, 1500) == ev.build_user_prompt(q, CONTEXTS, 1500)


def test_default_desc_chars_matches_eval():
    q = "default cut"
    assert generate.build_user_prompt(q, CONTEXTS) == ev.build_user_prompt(q, CONTEXTS, 1500)
