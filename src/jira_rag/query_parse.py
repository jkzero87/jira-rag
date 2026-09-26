"""Query parser: turns a natural-language question into a structured filter form.

Uses the local llama-server (OpenAI-compatible /v1/chat/completions) with
response_format {"type": "json_object", "schema": ...} so the output is
guaranteed to match the form (enforced via llama.cpp's JSON-schema grammar).

Fail-open semantics:
  - List values not in the DB's allowed enum lists are dropped (warning
    logged); if a list becomes empty, the field is set to null.
  - A parse failure after the retry returns the empty form (all null,
    text_terms=[]) with "parse_error": True in the form.
  - Thinking is OFF by default (enable_thinking=False).

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
import logging
import os
import re
import urllib.request

logger = logging.getLogger(__name__)

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
- Return only the JSON object matching the schema. No prose.

Examples:
- Question: "Which blocker bugs filed in March 2024 are still open?"
  -> {{"priority": ["Blocker"], "issue_type": ["Bug"], "open": true,
      "resolution": null, "created_from": "2024-03-01", "created_to": "2024-04-01",
      "text_terms": []}}
- Question: "Which sub-tasks and questions for the streaming module were filed in 2023?"
  -> {{"priority": null, "issue_type": ["Sub-task", "Question"], "open": null,
      "resolution": null, "created_from": "2023-01-01", "created_to": "2024-01-01",
      "text_terms": ["streaming"]}}
- Question: "Which major improvements filed since January 2025 are still open?"
  -> {{"priority": ["Major"], "issue_type": ["Improvement"], "open": true,
      "resolution": null, "created_from": "2025-01-01", "created_to": null,
      "text_terms": []}}
- Question: "My job fails with a segmentation fault when the driver sends a large shuffle block."
  -> {{"priority": null, "issue_type": null, "open": null, "resolution": null,
      "created_from": null, "created_to": null, "text_terms": []}}"""

_USER_TEMPLATE = "Question: {question}\nExtract the filter form."

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def _call_llm(messages, temperature=0.0, max_tokens=1024, enable_thinking=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": _JSON_SCHEMA},
    }
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    data = json.dumps(body).encode()
    req = urllib.request.Request(LLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    content = result["choices"][0]["message"]["content"]
    # Safety net: the model sometimes wraps the JSON in a fenced code block
    # (the json_schema grammar is not always enforced by llama-server).
    stripped = content.strip()
    if stripped.startswith("```"):
        # Drop the opening fence (``` or ```json) and a trailing fence.
        first_newline = stripped.find("\n")
        body = stripped[first_newline + 1:] if first_newline != -1 else stripped[3:]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        stripped = body.strip()
    return json.loads(stripped)

# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------
_EMPTY_FORM = {
    "priority": None, "issue_type": None, "open": None,
    "resolution": None, "created_from": None, "created_to": None,
    "text_terms": [],
}


def _apply_question_name_guard(question, form):
    """Keep a priority/issue_type value only if its name appears in the
    question (case-insensitive, singular or plural, whole word).

    For issue_type, "sub-task(s)" mentions are removed from the question
    BEFORE checking "task(s)", so a "sub-task" mention does NOT satisfy
    "task".  resolution, open and the date fields are NOT guarded.

    Dropped values log a warning; an emptied list becomes null.
    """
    q_low = question.lower()
    # Remove "sub-task"/"subtask" mentions so "task" checks are not fooled.
    q_nosub = q_low.replace("sub-task", "").replace("subtask", "")
    for key in ("priority", "issue_type"):
        v = form.get(key)
        if not v:
            continue
        kept = []
        for item in v:
            base = item.lower()
            singular = base[:-1] if base.endswith("s") else base
            # "Task" is matched against the sub-task-removed question so a
            # "sub-task" mention alone cannot satisfy "task".  Every other
            # type is matched against the raw (lowercased) question.
            check_in = q_nosub if base == "task" else q_low
            if re.search(r"\b" + re.escape(singular) + r"s?\b", check_in):
                kept.append(item)
            else:
                logger.warning(
                    "guard[%s]: dropped %s value %r (name not in question)",
                    question[:40], key, item)
        form[key] = kept if kept else None
    return form


def parse(question, *, retry=1, enable_thinking=False):
    """Parse a natural-language question into a structured filter form.

    Returns a dict with keys: priority, issue_type, open, resolution,
    created_from, created_to, text_terms.

    Fail-open semantics:
      - Any list value not in the DB's allowed enum list is dropped (warning
        logged); if a list becomes empty, the field is set to null.
      - A parse failure after the retry returns the empty form (all null,
        text_terms=[]) with "parse_error": True in the form, so callers
        (e.g. the eval) can still count it.
      - Thinking is OFF by default (enable_thinking=False). Set to None to
        let the server use its default (thinking ON), or True to force ON.

    enable_thinking: False (default, thinking disabled), True, or None
    (unchanged server behavior — thinking ON for Qwen models).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(question=question)},
    ]
    last_err = None
    for attempt in range(retry + 1):
        try:
            raw = _call_llm(messages, enable_thinking=enable_thinking)
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

            # Fail-open: drop values not in the DB's allowed enum lists.
            # If a list becomes empty, the field is null.
            for key, allowed in (("priority", PRIORITIES),
                                ("issue_type", ISSUE_TYPES),
                                ("resolution", RESOLUTIONS)):
                v = form.get(key)
                if v:
                    kept = []
                    for item in v:
                        if item in allowed:
                            kept.append(item)
                        else:
                            logger.warning(
                                "parse[%s]: dropped invalid %s value %r "
                                "(not in allowed list)",
                                question[:40], key, item)
                    if kept:
                        form[key] = kept
                    else:
                        form[key] = None

            # Code-level guard: keep a priority/issue_type value only if its
            # name appears in the question.  See _apply_question_name_guard.
            _apply_question_name_guard(question, form)

            form["parse_error"] = False
            return form
        except Exception as exc:
            last_err = exc
            if attempt == retry:
                logger.warning("parse[%s] failed after %d attempts: %s",
                               question[:40], retry + 1, last_err)
                result = dict(_EMPTY_FORM)
                result["parse_error"] = True
                return result
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

    # NOTE: text_terms are intentionally NOT used as a WHERE condition.
    # They are kept in the form for downstream retrieval/ranking (e.g.
    # reranking candidates by summary-term hits).

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
