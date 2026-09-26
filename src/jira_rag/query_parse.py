"""Query parser: turns a natural-language question into a structured filter form.

Uses the local llama-server (OpenAI-compatible /v1/chat/completions) with
response_format json_schema so the output is guaranteed to match the form.

Form fields (all nullable unless noted):
  priority:      list of allowed priorities or null
  issue_type:    list of real issue_type values or null
  open:          true (resolution IS NULL) / false (resolution IS NOT NULL) / null
  resolution:    list of real resolution values or null
  created_from:  "YYYY-MM-DD" inclusive or null
  created_to:    "YYYY-MM-DD" exclusive or null
  text_terms:    list of words that must appear in the summary (case-insensitive), or []

NOTE on `open` vs `resolution`: the schema maps
  open=true   -> resolution IS NULL
  open=false  -> resolution IS NOT NULL
If a question names a specific resolution value (e.g. "Cannot Reproduce"),
fill `resolution` and leave `open` null (a named resolution is NOT "open").
"""

import json
import os
import re
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8092/v1/chat/completions")
MODEL = os.environ.get("LLAMA_MODEL", "default")
SNAPSHOT_DATE = "2026-09-18"

# Allowed values straight from the DB (distinct values in jira.issues).
PRIORITIES = ["Blocker", "Critical", "Major", "Minor", "Trivial"]
ISSUE_TYPES = [
    "Bug", "Improvement", "Sub-task", "New Feature", "Task", "Test",
    "Documentation", "Umbrella", "Question", "Wish", "Dependency upgrade",
    "Story", "Epic", "Brainstorming", "IT Help", "Request", "Planned Work",
    "Github Integration", "Technical task", "RTC", "New JIRA Project",
    "Blog - New Blog Request",
]
RESOLUTIONS = [
    "Fixed", "Incomplete", "Duplicate", "Won't Fix", "Not A Problem",
    "Invalid", "Cannot Reproduce", "Done", "Resolved", "Later",
    "Won't Do", "Not A Bug", "Auto Closed", "Implemented", "Abandoned",
    "Workaround", "Information Provided", "Works for Me", "Feedback Received",
]

# ---------------------------------------------------------------------------
# JSON Schema for response_format
# ---------------------------------------------------------------------------
_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "priority": {
            "type": ["array", "null"],
            "items": {"type": "string", "enum": PRIORITIES},
        },
        "issue_type": {
            "type": ["array", "null"],
            "items": {"type": "string", "enum": ISSUE_TYPES},
        },
        "open": {
            "type": ["boolean", "null"],
        },
        "resolution": {
            "type": ["array", "null"],
            "items": {"type": "string", "enum": RESOLUTIONS},
        },
        "created_from": {
            "type": ["string", "null"],
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
        },
        "created_to": {
            "type": ["string", "null"],
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
        },
        "text_terms": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["priority", "issue_type", "open", "resolution",
                 "created_from", "created_to", "text_terms"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are a filter extractor for a Jira issue database (Apache Spark project).
Given a user's question about issues, extract ONLY the explicit filters it states.

Rules:
- Fill a field ONLY when the question states it explicitly. Do not guess.
- "open" means "still open" / "unresolved" / "not yet resolved" -> open=true.
  "closed" / "resolved" / "fixed" -> open=false.
  If the question names a specific resolution value (e.g. "Cannot Reproduce"),
  put it in the `resolution` list and leave `open` null.
- A question about a symptom, behavior, or bug (e.g. "crashes", "fails",
  "wrong results", "can't handle it") gets an empty form: all fields null /
  empty list, text_terms=[].
- text_terms: ONLY when the question names a specific component or product
  (e.g. "Kubernetes", "Spark Connect", "ZooKeeper", "Parquet", "S3").
  Use the exact product name as the term. Do NOT put symptom words in text_terms.
- Dates: "filed in 2026" -> created_from="2026-01-01".
  "since June 2026" -> created_from="2026-06-01".
  "filed in 2025" -> created_from="2025-01-01", created_to="2026-01-01".
  created_to is exclusive.
- The data snapshot date is {SNAPSHOT_DATE}. Do not use today's date.
- Return only the JSON object matching the schema. No prose."""

_USER_TEMPLATE = "Question: {question}\nExtract the filter form."

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def _call_llm(messages, temperature=0.0, max_tokens=1024):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": _JSON_SCHEMA},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(LLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    content = result["choices"][0]["message"]["content"]
    return json.loads(content)

# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
def parse(question, *, retry=1):
    """Parse a natural-language question into a structured filter form.

    Returns a dict with keys: priority, issue_type, open, resolution,
    created_from, created_to, text_terms.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(question=question)},
    ]
    last_err = None
    for attempt in range(retry + 1):
        try:
            raw = _call_llm(messages)
            # Basic sanity checks
            if not isinstance(raw, dict):
                raise ValueError(f"expected dict, got {type(raw)}")
            # Strip unexpected keys (model sometimes hallucinates "issuetype" etc.);
            # fill missing required keys with their empty defaults.
            form = {}
            for key in ("priority", "issue_type", "open", "resolution",
                        "created_from", "created_to", "text_terms"):
                form[key] = raw.get(key)
            # Normalize: None for list fields (model sometimes returns [] for "none"),
            # and wrap bare strings (single-item lists) into lists.
            for key in ("priority", "issue_type", "resolution", "text_terms"):
                v = form[key]
                if isinstance(v, str):
                    form[key] = [v]
                elif v == []:
                    form[key] = None if key != "text_terms" else []
                elif v is None and key == "text_terms":
                    form[key] = []
            if not isinstance(form.get("text_terms"), list):
                form["text_terms"] = []
            return form
        except Exception as exc:
            last_err = exc
            if attempt == retry:
                raise
    # unreachable
    raise RuntimeError("parse failed after retries")

# ---------------------------------------------------------------------------
# to_sql
# ---------------------------------------------------------------------------
def to_sql(form):
    """Convert a filter form to a parameterized WHERE clause + params list.

    Returns (where_clause, params).
    where_clause is "" if no filters (caller must handle).

    Note: single-item lists from the model come through as bare strings,
    not lists — normalize first.
    """
    clauses = []
    params = []

    def as_list(v):
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v

    # priority
    prio = as_list(form.get("priority"))
    if prio:
        placeholders = ", ".join(["%s"] * len(prio))
        clauses.append(f"priority IN ({placeholders})")
        params.extend(prio)

    # issue_type
    itype = as_list(form.get("issue_type"))
    if itype:
        placeholders = ", ".join(["%s"] * len(itype))
        clauses.append(f"issue_type IN ({placeholders})")
        params.extend(itype)

    # open
    if form.get("open") is True:
        clauses.append("resolution IS NULL")
    elif form.get("open") is False:
        clauses.append("resolution IS NOT NULL")

    # resolution
    if form.get("resolution"):
        placeholders = ", ".join(["%s"] * len(form["resolution"]))
        clauses.append(f"resolution IN ({placeholders})")
        params.extend(form["resolution"])

    # created_from (inclusive)
    if form.get("created_from"):
        clauses.append("created >= %s::timestamp")
        params.append(form["created_from"])

    # created_to (exclusive)
    if form.get("created_to"):
        clauses.append("created < %s::timestamp")
        params.append(form["created_to"])

    # text_terms (case-insensitive substring on summary)
    if form.get("text_terms"):
        for term in form["text_terms"]:
            clauses.append("summary ILIKE %s")
            params.append(f"%{term}%")

    where = " AND ".join(clauses) if clauses else ""
    return where, params

# ---------------------------------------------------------------------------
# CLI: parse a single question and print the form + SQL
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: query_parse.py <question>", file=sys.stderr)
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    form = parse(q)
    print("FORM:")
    print(json.dumps(form, indent=2))
    where, params = to_sql(form)
    print(f"\nSQL WHERE: {where or '(no filters)'}")
    print(f"PARAMS: {params}")
