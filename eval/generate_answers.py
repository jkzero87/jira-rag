#!/usr/bin/env python3
"""Phase 3 of the generation eval: generate answers (generation only).

LLM calls: one llama-server chat/completions request per gold question.
The contexts come from a phase-2 contexts file (build_contexts.py output),
so no retrieval or parsing happens here. The 27B server must be live.
For each question the model must answer using ONLY the given issues and
cite every claim with the issue key in square brackets, e.g. [SPARK-1234].

Usage:
  ~/jira-rag/.venv/bin/python eval/generate_answers.py --contexts <file> --out <file> \
      [--limit N] [--desc-chars 1500]

Output JSON:
  {"model", "model_source", "commit", "contexts", "settings", "created_at",
   "questions": [{"id", "question", "answer", "cited_keys", "context_keys",
                  "prompt_tokens", "completion_tokens", "seconds",
                  "desc_chars_total"}]}
An existing output file is NEVER overwritten.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))
sys.path.insert(0, str(ROOT / "eval"))

import psycopg2
import query_parse  # noqa: E402

# head_commit / resolve_model are copied from eval/run_eval.py (run_eval
# imports retrieve -> psycopg2 chain, which we avoid): the 27B server is
# live, so the DB is only needed for descriptions.


def head_commit():
    import subprocess
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def resolve_model(explicit=None):
    """Model name for the request -> (name, source).

    Server reachable: GET /v1/models ("server"); an explicit name that
    disagrees is refused. Server unreachable: only --model or
    $JIRA_RAG_MODEL ("explicit"). Neither -> refuse."""
    import os
    import urllib.request
    explicit = explicit or os.environ.get("JIRA_RAG_MODEL")
    try:
        models_url = query_parse.LLAMA_URL.rsplit("/chat/completions", 1)[0] + "/models"
        data = json.loads(urllib.request.urlopen(models_url, timeout=10).read())
        server = data["data"][0]["id"] if "data" in data else data["models"][0]["name"]
    except OSError as exc:
        if not explicit:
            sys.exit(f"error: llama-server unreachable ({exc}) and no --model / "
                     f"JIRA_RAG_MODEL given")
        return explicit, "explicit"
    if explicit and explicit != server:
        sys.exit(f"error: explicit model {explicit!r} != server model {server!r}; refusing")
    return server, "server"


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
ENV_FILE = ROOT / ".env"


def load_dsn():
    """DSN from ~/.jira-rag/.env (PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD)."""
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return {"host": env["PGHOST"], "port": env["PGPORT"], "dbname": env["PGDATABASE"],
            "user": env["PGUSER"], "password": env["PGPASSWORD"]}


def fetch_descriptions(keys):
    """One query: issue_key -> description (None kept as None)."""
    if not keys:
        return {}
    conn = psycopg2.connect(**load_dsn())
    try:
        cur = conn.cursor()
        cur.execute("SELECT issue_key, description FROM jira.issues "
                    "WHERE issue_key = ANY(%s)", (keys,))
        return {k: d for k, d in cur.fetchall()}
    finally:
        conn.close()


def build_user_prompt(question, contexts, desc_chars):
    """Numbered contexts in order; each block is one line with the key plus
    every metadata field present, followed by a separate final line
    description=<first desc_chars chars> when the description is not NULL.
    Returns (prompt, desc_chars_total)."""
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


def chat(model, user_prompt):
    """One chat/completions call; returns (content, prompt_tokens, completion_tokens)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(query_parse.LLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        result = json.loads(resp.read())
    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage") or {}
    return content, usage.get("prompt_tokens"), usage.get("completion_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True, help="phase-2 contexts JSON file")
    ap.add_argument("--out", required=True, help="output JSON file (never overwritten)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N questions (default: all)")
    ap.add_argument("--desc-chars", type=int, default=1500,
                    help="max description characters appended per issue (default 1500)")
    ap.add_argument("--model", help="model name (gguf path) when llama-server is down; "
                                    "default $JIRA_RAG_MODEL")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"error: {out} already exists; refusing to overwrite")

    contexts_file = Path(args.contexts)
    data = json.loads(contexts_file.read_text())
    questions = data["questions"]
    if args.limit is not None:
        questions = questions[:args.limit]

    model, model_source = resolve_model(args.model)

    answers = []
    print(f"generating answers for {len(questions)} questions "
          f"(model {model}, {model_source}) ...", flush=True)
    for q in questions:
        t0 = time.monotonic()
        context_keys = [c["issue_key"] for c in q["contexts"]]
        descs = fetch_descriptions(context_keys)
        for ctx in q["contexts"]:
            ctx["description"] = descs.get(ctx["issue_key"])
        user_prompt, desc_chars_total = build_user_prompt(q["question"], q["contexts"], args.desc_chars)
        content, prompt_tokens, completion_tokens = chat(model, user_prompt)
        cited = []
        for key in CITE_RE.findall(content):
            if key not in cited:
                cited.append(key)
        answers.append({
            "id": q["id"],
            "question": q["question"],
            "answer": content,
            "cited_keys": cited,
            "context_keys": context_keys,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "seconds": round(time.monotonic() - t0, 2),
            "desc_chars_total": desc_chars_total,
        })
        print(f"  {q['id']}: {time.monotonic() - t0:.2f}s, "
              f"{prompt_tokens} prompt / {completion_tokens} completion tokens, "
              f"{len(cited)} citations", flush=True)

    payload = {
        "model": model,
        "model_source": model_source,
        "commit": head_commit(),
        "contexts": str(contexts_file),
        "settings": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "stream": False,
            "enable_thinking": False,
            "desc_chars": args.desc_chars,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "questions": answers,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nanswers written: {out} ({len(answers)} questions)")


if __name__ == "__main__":
    main()
