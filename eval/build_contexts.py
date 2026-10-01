#!/usr/bin/env python3
"""Phase 2 of the generation eval: build retrieval contexts (retrieval only).

No LLM calls: the parse form comes from a parse cache (validated by the same
parser fingerprint as run_eval.py), so the 27B server may be stopped.
For each gold question the same route as search_rerank is run
(`summary_desc` strategy, k=10):
  - WHERE clause present  -> filtered path (vector order unchanged)
  - WHERE clause absent   -> rerank path (vector top-20 + m3 + RRF k=60)
and the top 10 issues become contexts with their fields (description cut
to DESC_CHARS=1500 chars).

Usage:
  python eval/build_contexts.py --parse-cache <file> --out <file> [--limit N]

Output JSON:
  {"fingerprint", "commit", "device", "created_at",
   "questions": [{"id", "question", "form", "route", "contexts": [...]}]}
An existing output file is NEVER overwritten.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))
sys.path.insert(0, str(ROOT / "eval"))

import retrieve  # noqa: E402
import query_parse  # noqa: E402
from run_eval import (check_parse_errors, head_commit, parser_fingerprint,  # noqa: E402
                      resolve_model)

STRATEGY = "summary_desc"
K = 10
DESC_CHARS = 1500
RERANK_SNAPSHOT = ROOT / "eval" / "cache" / "hf_hub" / "models--BAAI--bge-reranker-v2-m3"
RERANK_MAX_LENGTH = 512
RERANK_DESC_CHARS = 500  # doc text fed to the reranker (same as search_rerank)

CONTEXT_FIELDS = ["issue_key", "summary", "status", "resolution",
                  "priority", "issue_type", "created"]

_reranker = None


def _rerank_scores(question, cand_keys):
    """bge-reranker-v2-m3 scores for (question, doc) pairs.

    doc = summary + first RERANK_DESC_CHARS of description — exactly the
    text search_rerank feeds the CrossEncoder, so scores and order match
    retrieve.search_rerank.  sentence_transformers is not installed in the
    dl env, so load XLMRobertaForSequenceClassification directly from the
    local HF snapshot (model.safetensors includes the classifier weights).
    The score is out.logits[0] (num_labels=1)."""
    global _reranker
    if _reranker is None:
        from transformers import (
            AutoTokenizer,
            XLMRobertaConfig,
            XLMRobertaForSequenceClassification,
        )
        from safetensors.torch import load_file
        import glob as _glob
        snap_dir = _glob.glob(str(RERANK_SNAPSHOT / "snapshots" / "*"))[0]
        model_path = snap_dir + "/model.safetensors"
        print(f"loading bge-reranker-v2-m3 (fp32, {retrieve.DEVICE}) from local snapshot ...",
              flush=True)
        cfg = XLMRobertaConfig.from_pretrained(snap_dir, local_files_only=True)
        model = XLMRobertaForSequenceClassification(cfg).to(retrieve.DEVICE)
        model.load_state_dict(load_file(model_path), strict=True)
        model.eval()
        tok = AutoTokenizer.from_pretrained(snap_dir, local_files_only=True)
        _reranker = (tok, model)
    tok, model = _reranker
    docs = {}
    conn = psycopg2.connect(retrieve.DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT issue_key, summary, description FROM jira.issues "
            "WHERE issue_key = ANY(%s)", (cand_keys,))
        docs = {k: (s or "") + "\n" + (d or "")[:RERANK_DESC_CHARS]
                for k, s, d in cur.fetchall()}
    finally:
        conn.close()
    pairs = [(question, docs.get(k, "")) for k in cand_keys]
    enc = tok(pairs, padding=True, truncation=True, max_length=RERANK_MAX_LENGTH,
              return_tensors="pt").to(retrieve.DEVICE)
    with torch.no_grad():
        scores = model(**enc).logits  # shape (batch, 1)
    return {k: float(s) for k, s in zip(cand_keys, scores[:, 0].tolist())}


def rerank_route(question, form, where, params, qvec):
    """The rerank path of search_rerank, with a cached form (no parse, no
    server): vector top RERANK_N -> m3 scores -> blended RRF k=60 (1-based
    ranks, ties broken by vector rank). Returns [(issue_key, rrf), ...]."""
    where_prefix = " AND " if where else ""
    conn = psycopg2.connect(retrieve.DSN)
    try:
        cur = conn.cursor()
        vec_sql = (
            "SELECT ic.issue_key, ic.embedding <=> %s::vector AS distance\n"
            "FROM jira.issue_chunks ic\n"
            "JOIN jira.issues i ON i.issue_key = ic.issue_key\n"
            "WHERE ic.strategy = %s"
            + (where_prefix + where if where else "")
            + "\nORDER BY distance, ic.issue_key\nLIMIT "
            + str(retrieve.RERANK_N)
        )
        cur.execute(vec_sql, [str(qvec), STRATEGY] + list(params))
        vec_rows = cur.fetchall()  # [(issue_key, distance)]
    finally:
        conn.close()
    if where:
        # Filtered path: vector order unchanged (search_rerank's routing rule).
        return vec_rows[:K]
    cand_keys = [k for k, _ in vec_rows]
    scores = _rerank_scores(question, cand_keys)
    rank_vec = {k: i for i, k in enumerate(cand_keys)}
    rank_rr = {k: i for i, k in enumerate(sorted(cand_keys, key=lambda k: -scores[k]))}
    rrf = {k: 1.0 / (retrieve.RERANK_RRF_K + rank_vec[k] + 1)
           + 1.0 / (retrieve.RERANK_RRF_K + rank_rr[k] + 1) for k in cand_keys}
    order = sorted(cand_keys, key=lambda k: (-rrf[k], rank_vec[k]))
    return [(k, rrf[k]) for k in order[:K]]


def load_parse_cache(path, gold, model=None):
    """Same validity rule as run_eval.py: the cache is accepted only when its
    parser fingerprint matches the current one (sha256 of query_parse.py
    source + model name).  Never calls the parser; the model name comes from
    the server if it is up, else only from --model / $JIRA_RAG_MODEL."""
    data = json.loads(Path(path).read_text())
    server_model, model_source = resolve_model(model)
    fp = parser_fingerprint(server_model)
    if data.get("fingerprint") != fp:
        sys.exit(f"error: parse cache {path} fingerprint {data.get('fingerprint')!r} "
                 f"does not match current parser fingerprint {fp} "
                 f"(cache commit {data.get('commit')!r}); refusing")
    forms = {e["question"]: e["form"] for e in data["parses"]}
    missing = [g["id"] for g in gold if g["question"] not in forms]
    if missing:
        sys.exit(f"error: parse cache {path} lacks questions {missing}")
    check_parse_errors(((e.get("id", e["question"]), e["form"]) for e in data["parses"]),
                       f"parse cache {path}")

    def cached_parse(q):
        return forms[q]  # KeyError, never a live parse

    retrieve.parse_question = cached_parse
    print(f"parse cache: {path} (fingerprint {fp[:12]}…, model {server_model} "
          f"[{model_source}], commit {head_commit()[:7]} [info])", flush=True)
    return fp, server_model, model_source


def fetch_contexts(keys):
    """One query per issue set: fields + description cut to DESC_CHARS."""
    if not keys:
        return {}
    conn = psycopg2.connect(retrieve.DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT issue_key, summary, status, resolution, priority, "
            "issue_type, created, LEFT(description, %s) "
            "FROM jira.issues WHERE issue_key = ANY(%s)",
            (DESC_CHARS, keys))
        rows = cur.fetchall()
    finally:
        conn.close()
    out = {}
    for r in rows:
        ctx = dict(zip(CONTEXT_FIELDS, r))
        ctx["created"] = ctx["created"].isoformat() if ctx["created"] else None
        out[ctx["issue_key"]] = ctx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-cache", required=True,
                    help="parse cache file; fingerprint must match current parser")
    ap.add_argument("--out", required=True, help="output JSON file (never overwritten)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N gold questions (default: all)")
    ap.add_argument("--model", help="parser model name (gguf path) when llama-server is down; "
                                    "default $JIRA_RAG_MODEL")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"error: {out} already exists; refusing to overwrite")

    gold = [json.loads(line)
            for line in (ROOT / "eval" / "gold.jsonl").read_text().splitlines() if line.strip()]
    if args.limit is not None:
        gold = gold[:args.limit]

    fp, model, model_source = load_parse_cache(args.parse_cache, gold, args.model)
    retrieve.init()

    questions = []
    print(f"retrieving top {K} for {len(gold)} questions ({STRATEGY}) ...", flush=True)
    for g in gold:
        t0 = time.monotonic()
        form = retrieve.parse_question(g["question"])  # cached form
        where, params = query_parse.to_sql(form)
        qvec = retrieve.embed_query(g["question"])
        rows = rerank_route(g["question"], form, where, params, qvec)
        route = "filtered" if where else "rerank"
        keys = [k for k, _ in rows]
        ctx_map = fetch_contexts(keys)
        contexts = []
        for rank, key in enumerate(keys, 1):
            ctx = dict(ctx_map[key])
            ctx["rank"] = rank
            contexts.append(ctx)
        questions.append({"id": g["id"], "question": g["question"],
                          "form": form, "route": route, "contexts": contexts})
        print(f"  {g['id']}: route={route} n={len(contexts)} "
              f"({time.monotonic() - t0:.2f}s)", flush=True)

    payload = {
        "fingerprint": fp,
        "model": model,
        "model_source": model_source,
        "commit": head_commit(),
        "device": retrieve.DEVICE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": STRATEGY,
        "k": K,
        "desc_chars": DESC_CHARS,
        "parse_cache": args.parse_cache,
        "questions": questions,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\ncontexts written: {out} ({len(questions)} questions)")


if __name__ == "__main__":
    main()
