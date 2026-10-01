"""The parser's few-shot examples must not contain any gold question.

A gold question inside SYSTEM_PROMPT shows the model the answer to an eval
item (G07 was one, until it was replaced with a non-gold example).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jira_rag.query_parse import SYSTEM_PROMPT


def gold_questions():
    lines = (ROOT / "eval" / "gold.jsonl").read_text().splitlines()
    return [json.loads(line)["question"] for line in lines if line.strip()]


def test_no_gold_question_in_prompt():
    leaked = [q for q in gold_questions() if q.lower() in SYSTEM_PROMPT.lower()]
    assert not leaked, leaked
