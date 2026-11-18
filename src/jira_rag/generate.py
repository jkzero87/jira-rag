"""Answer generation: retrieved issues -> cited answer from the 27B.

Library version of the phase-3 eval (eval/generate_answers.py), used by the
API's /ask. Prompt, context format and sampling settings MUST stay identical
to the eval; tests/test_generate_equivalence.py enforces it.
DB connection: libpq PG* environment variables (DSN "").
"""
import json
import re
import urllib.request

import psycopg2

import query_parse

SYSTEM = ("Answer the question using ONLY the Jira issues provided. "
          "Cite every claim with the issue key in square brackets, e.g. [SPARK-1234]. "
          "If the issues do not contain the answer, say so plainly and cite nothing. "
          "Be concise.")

CONTEXT_FIELDS = ["issue_key", "summary", "status", "resolution",
                  "priority", "issue_type", "created"]
CITE_RE = re.compile(r"SPARK-\d+")
TEMPERATURE = 0.0
MAX_TOKENS = 1024
TIMEOUT = 300
DESC_CHARS = 1500  # same as the phase-3 eval (README section 4)


def fetch_contexts(keys):
    """Context dicts for keys, in the given (ranked) order; missing keys are skipped."""
    if not keys:
        return []
    conn = psycopg2.connect("")
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT issue_key, summary, status, resolution, priority, "
            "issue_type, created, description "
            "FROM jira.issues WHERE issue_key = ANY(%s)", (list(keys),))
        rows = cur.fetchall()
    finally:
        conn.close()
    by_key = {}
    for r in rows:
        ctx = dict(zip(CONTEXT_FIELDS + ["description"], r))
        ctx["created"] = ctx["created"].isoformat() if ctx["created"] else None
        by_key[ctx["issue_key"]] = ctx
    return [by_key[k] for k in keys if k in by_key]


def build_user_prompt(question, contexts, desc_chars=DESC_CHARS):
    """Same format as eval/generate_answers.py. Returns (prompt, desc_chars_total)."""
    lines = [f"Question: {question}", "", "Jira issues:"]
    total = 0
    for i, ctx in enumerate(contexts, 1):
        parts = []
        for field in CONTEXT_FIELDS:
            value = ctx.get(field)
            if value is not None:
                parts.append(f"{field}={value}")
        lines.append(f"{i}. [{ctx['issue_key']}] " + "; ".join(parts))
        desc = ctx.get("description")
        if desc is not None:
            cut = desc[:desc_chars]
            total += len(cut)
            lines.append(f"description={cut}")
    lines.append("")
    lines.append("Answer with citations.")
    return "\n".join(lines), total


def answer(question, contexts):
    """One chat/completions call to the 27B. Raises urllib.error.URLError if unreachable."""
    user_prompt, _ = build_user_prompt(question, contexts)
    body = {
        "model": query_parse.MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(query_parse.LLAMA_URL, data=json.dumps(body).encode(),
                                 headers=query_parse.llm_headers())
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def cited_keys(text):
    """Unique SPARK keys in order of first appearance."""
    seen = []
    for key in CITE_RE.findall(text):
        if key not in seen:
            seen.append(key)
    return seen
