#!/usr/bin/env python3
"""Baseline retrieval evaluation over eval/gold.jsonl.

Each gold question is embedded ONCE and the vector is reused for every
strategy. Per (strategy, question) a pgvector cosine top-10 search is run
and scored:
  recall@5, recall@10 = |found ∩ expected| / |expected|
  MRR = 1/rank of the first expected key in the top 10 (0 if none).
Prints a table of means (overall and by type) and saves everything,
including per-question top-10 keys and first ranks, to
eval/results/<YYYY-MM-DD>_<short git hash>_baseline.json.
An existing results file is NEVER overwritten.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

import retrieve  # noqa: E402
import query_parse  # noqa: E402
from metrics import recall_at_k, mrr_at_k  # noqa: E402

STRATEGIES = ("summary_only", "summary_desc", "summary_desc_filtered",
              "summary_desc_hybrid", "summary_desc_rescue",
              "summary_desc_rerank")
TYPES = ("lookup", "topic", "filtered")
K = 10


def check_parse_errors(entries, source):
    """Exit if any parsed form has parse_error set.

    entries: iterable of (question id or text, form). A parse_error form is
    the all-null fallback, and scoring it would silently turn a filtered
    question into an unfiltered one."""
    bad = [qid for qid, form in entries if form.get("parse_error")]
    if bad:
        sys.exit(f"error: {source}: parse_error for {len(bad)} question(s): "
                 f"{', '.join(map(str, bad))}; refusing to score")


def question_metrics(keys, expected):
    """keys: top-K issue keys (nearest first)."""
    exp = set(expected)
    return {
        "recall_at_5": recall_at_k(keys, expected, 5),
        "recall_at_10": recall_at_k(keys, expected, K),
        "mrr": mrr_at_k(keys, expected, K),
        "first_rank": next((i + 1 for i, k in enumerate(keys[:K]) if k in exp), None),
    }


def means(entries):
    n = len(entries)
    return {
        "n": n,
        "recall_at_5": sum(e["recall_at_5"] for e in entries) / n,
        "recall_at_10": sum(e["recall_at_10"] for e in entries) / n,
        "mrr": sum(e["mrr"] for e in entries) / n,
    }


def head_commit():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def server_model_name():
    """Model name reported by the llama-server (GET /v1/models)."""
    models_url = query_parse.LLAMA_URL.rsplit("/chat/completions", 1)[0] + "/models"
    data = json.loads(urllib.request.urlopen(models_url, timeout=10).read())
    return data["data"][0]["id"] if "data" in data else data["models"][0]["name"]


def resolve_model(explicit=None):
    """Model name for the parser fingerprint -> (name, source).

    Server reachable: GET /v1/models ("server"); an explicit name that
    disagrees is refused. Server unreachable (27B stopped for a GPU run): only
    --model or $JIRA_RAG_MODEL ("explicit"), never the model recorded in the
    cache, which would validate the cache against itself. Neither -> refuse."""
    explicit = explicit or os.environ.get("JIRA_RAG_MODEL")
    try:
        server = server_model_name()
    except OSError as exc:
        if not explicit:
            sys.exit(f"error: llama-server unreachable ({exc}) and no --model / "
                     "JIRA_RAG_MODEL given; cannot compute the parser fingerprint")
        return explicit, "explicit"
    if explicit and explicit != server:
        sys.exit(f"error: explicit model {explicit!r} != server model {server!r}; refusing")
    return server, "server"


def parser_fingerprint(model=None):
    """Fingerprint of the parser: sha256 of query_parse.py source + server model name.
    The commit alone is not a valid cache key: a new commit (even a doc-only one)
    invalidates a still-correct cache, while a cache can go stale on the same
    commit (server session changed, parser outputs changed)."""
    src_hash = hashlib.sha256((ROOT / "src" / "jira_rag" / "query_parse.py").read_bytes()).hexdigest()
    if model is None:
        model = server_model_name()
    return hashlib.sha256(f"{src_hash}|{model}".encode()).hexdigest()


def load_parse_cache(path, gold, model=None):
    """Load a parse cache and install it as retrieve.parse_question.
    Validity is checked by the parser fingerprint (sha256 of query_parse.py
    source + server model name); "commit" is stored only as information.
    Refuses a cache whose fingerprint differs (a stale cache silently scored
    the wrong parser once: lists.json)."""
    data = json.loads(Path(path).read_text())
    head = head_commit()
    server_model, model_source = resolve_model(model)
    fp = parser_fingerprint(server_model)
    if data.get("fingerprint") != fp:
        sys.exit(f"error: parse cache {path} fingerprint {data.get('fingerprint')!r} "
                 f"does not match current parser fingerprint {fp} "
                 f"(cache commit {data.get('commit')!r}, HEAD {head[:7]}); refusing")
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
          f"[{model_source}], commit {head[:7]} [info])", flush=True)
    data["model_source"] = model_source
    return data


def save_parse_cache(path, parses):
    """Write a parse cache file keyed by the parser fingerprint.

    parses: list of {"id", "question", "form", "seconds"} entries.
    "commit" is stored for information only; "fingerprint" decides validity."""
    path = Path(path)
    model = server_model_name()
    payload = {
        "fingerprint": parser_fingerprint(model),
        "commit": head_commit(),
        "model": model,
        "llama_url": query_parse.LLAMA_URL,
        "enable_thinking": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parses": parses,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"parse cache written: {path} (fingerprint {payload['fingerprint'][:12]}…)",
          flush=True)


def write_parse_cache(path, gold):
    """Parse every gold question live (27B server up) and save a parse cache.

    Refuses to overwrite an existing file, and writes nothing if any
    question comes back with parse_error."""
    path = Path(path)
    if path.exists():
        sys.exit(f"error: {path} already exists; refusing to overwrite")
    parses = []
    for g in gold:
        t = time.monotonic()
        form = query_parse.parse(g["question"])
        parses.append({"id": g["id"], "question": g["question"], "form": form,
                       "seconds": round(time.monotonic() - t, 3)})
        print(f"  {g['id']}: parsed ({parses[-1]['seconds']:.2f}s)", flush=True)
    check_parse_errors(((e["id"], e["form"]) for e in parses), "live parse")
    save_parse_cache(path, parses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse-cache", help="eval/cache/parse_*.json; fingerprint must match current parser")
    ap.add_argument("--write-parse-cache", metavar="PATH",
                    help="parse all gold questions live (27B server up), write a parse "
                         "cache to PATH and exit; no retrieval is run")
    ap.add_argument("--model", help="parser model name (gguf path) when llama-server is down; "
                                    "default $JIRA_RAG_MODEL")
    args = ap.parse_args()

    gold = [json.loads(line)
            for line in (ROOT / "eval" / "gold.jsonl").read_text().splitlines() if line.strip()]
    if args.write_parse_cache:
        write_parse_cache(args.write_parse_cache, gold)
        return
    t0 = time.monotonic()
    parse_cache = load_parse_cache(args.parse_cache, gold, args.model) if args.parse_cache else None

    retrieve.init()

    print(f"embedding {len(gold)} questions (once each) ...", flush=True)
    qvecs = {g["id"]: retrieve.embed_query(g["question"]) for g in gold}

    # time embed_query separately so the rerank "vector" stage splits into embed + SQL
    embed_s = []
    _embed = retrieve.embed_query

    def timed_embed(q, *a, **kw):
        t = time.monotonic()
        try:
            return _embed(q, *a, **kw)
        finally:
            embed_s.append(time.monotonic() - t)

    retrieve.embed_query = timed_embed

    per_q = {}
    print(f"searching top {K} per strategy ({', '.join(STRATEGIES)}) ...", flush=True)
    for g in gold:
        per_q[g["id"]] = {"id": g["id"], "type": g["type"], "question": g["question"],
                          "expected": g["expected"], "strategies": {}}
        for s in STRATEGIES:
            if s == "summary_desc_filtered":
                # parse → WHERE → vector rank
                rows, form, where, filter_count = retrieve.search_filtered(
                    g["question"], "summary_desc", K)
                entry = question_metrics([r[0] for r in rows], g["expected"])
                entry["top10"] = [r[0] for r in rows]
                entry["distances"] = [round(float(r[1]), 6) for r in rows]
                entry["form"] = form
                entry["where"] = where
                entry["filter_row_count"] = filter_count
                entry["result_count"] = len(rows)
                entry["expected_count"] = min(K, filter_count)
            elif s == "summary_desc_rescue":
                # parse → WHERE → vector order + keyword tail rescue (m=1)
                rows, form, where = retrieve.search_rescue(
                    g["question"], "summary_desc", K)
                entry = question_metrics([r[0] for r in rows], g["expected"])
                entry["top10"] = [r[0] for r in rows]
                entry["scores"] = [round(r[1], 8) for r in rows]
                entry["form"] = form
                entry["where"] = where
                entry["result_count"] = len(rows)
            elif s == "summary_desc_hybrid":
                # parse → WHERE → vector list + keyword list → RRF
                rows, form, where = retrieve.search_hybrid(
                    g["question"], "summary_desc", K)
                entry = question_metrics([r[0] for r in rows], g["expected"])
                entry["top10"] = [r[0] for r in rows]
                entry["scores"] = [round(r[1], 8) for r in rows]
                entry["form"] = form
                entry["where"] = where
                entry["result_count"] = len(rows)
            elif s == "summary_desc_rerank":
                # parse → WHERE → vector top-20 → bge-reranker-v2-m3 (routed)
                # Fresh vector search each question; parse live unless --parse-cache.
                embed_s.clear()
                t_stage = time.monotonic()
                rows, form, where, stages = retrieve.search_rerank(
                    g["question"], "summary_desc", K)
                total_s = time.monotonic() - t_stage
                stages["embed"] = sum(embed_s)
                stages["sql"] = stages["vector"] - stages["embed"]
                entry = question_metrics([r[0] for r in rows], g["expected"])
                entry["top10"] = [r[0] for r in rows]
                entry["scores"] = [round(r[1], 8) for r in rows]
                entry["form"] = form
                entry["where"] = where
                entry["result_count"] = len(rows)
                entry["rerank_seconds"] = round(total_s, 3)
                entry["stage_seconds"] = {k: round(v, 4) for k, v in stages.items()}
            else:
                rows = retrieve.search_vec(qvecs[g["id"]], s, K)
                entry = question_metrics([r[0] for r in rows], g["expected"])
                entry["top10"] = [r[0] for r in rows]
                entry["distances"] = [round(float(r[1]), 6) for r in rows]
            if "form" in entry:
                check_parse_errors([(g["id"], entry["form"])], s)
            per_q[g["id"]]["strategies"][s] = entry

    summary = {}
    for s in STRATEGIES:
        all_entries = [per_q[g["id"]]["strategies"][s] for g in gold]
        summary[s] = {
            "overall": means(all_entries),
            "by_type": {t: means([per_q[g["id"]]["strategies"][s] for g in gold if g["type"] == t])
                        for t in TYPES},
        }

    # ---- table ----
    hdr = f"{'strategy':<15} {'scope':<8} {'n':>3} {'recall@5':>10} {'recall@10':>11} {'MRR':>8}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for s in STRATEGIES:
        for scope, m in [("overall", summary[s]["overall"]),
                         *((t, summary[s]["by_type"][t]) for t in TYPES)]:
            print(f"{s:<15} {scope:<8} {m['n']:>3} {m['recall_at_5']:>10.4f} "
                  f"{m['recall_at_10']:>11.4f} {m['mrr']:>8.4f}")

    # ---- rerank stage timing ----
    rerank_entries = [per_q[g["id"]]["strategies"]["summary_desc_rerank"]
                      for g in gold]
    n = len(rerank_entries)
    avg_parse = sum(e["stage_seconds"]["parse"] for e in rerank_entries) / n
    avg_vector = sum(e["stage_seconds"]["vector"] for e in rerank_entries) / n
    avg_rerank = sum(e["stage_seconds"]["rerank"] for e in rerank_entries) / n
    avg_embed = sum(e["stage_seconds"]["embed"] for e in rerank_entries) / n
    avg_sql = sum(e["stage_seconds"]["sql"] for e in rerank_entries) / n
    avg_total = sum(e["rerank_seconds"] for e in rerank_entries) / n
    print(f"\nsummary_desc_rerank end-to-end (avg over {n} questions):")
    print(f"  parse:    {avg_parse:.4f} s")
    print(f"  vector:   {avg_vector:.4f} s  (embed {avg_embed:.4f} + SQL {avg_sql:.4f})")
    print(f"  rerank:   {avg_rerank:.4f} s")
    print(f"  total:    {avg_total:.4f} s")

    # ---- rerank vs filtered: worse/better lists ----
    print("\nsummary_desc_rerank vs summary_desc_filtered (MRR delta > 1e-9):")
    worse = []
    better = []
    for g in gold:
        rid = g["id"]
        filtered_mrr = per_q[rid]["strategies"]["summary_desc_filtered"]["mrr"]
        rerank_mrr = per_q[rid]["strategies"]["summary_desc_rerank"]["mrr"]
        if rerank_mrr < filtered_mrr - 1e-9:
            worse.append((rid, filtered_mrr, rerank_mrr))
        elif rerank_mrr > filtered_mrr + 1e-9:
            better.append((rid, filtered_mrr, rerank_mrr))
    print(f"  worse ({len(worse)}):")
    for rid, fm, rm in worse:
        print(f"    {rid}  filtered MRR={fm:.4f} → rerank MRR={rm:.4f}")
    print(f"  better ({len(better)}):")
    for rid, fm, rm in better:
        print(f"    {rid}  filtered MRR={fm:.4f} → rerank MRR={rm:.4f}")

    # ---- results file (never overwrite) ----
    today = date.today().isoformat()
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    out = ROOT / "eval" / "results" / f"{today}_{git}_baseline.json"
    if out.exists():
        sys.exit(f"error: {out} already exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": today,
        "git_hash": git,
        "model": retrieve.MODEL,
        "device": retrieve.DEVICE,
        "parse_cache": ({"path": args.parse_cache,
                         "fingerprint": parse_cache["fingerprint"],
                         "commit": parse_cache["commit"],  # informational only
                         "model": parse_cache.get("model"),
                         "model_source": parse_cache["model_source"]} if parse_cache else None),
        "prefix": retrieve.QUERY_PREFIX,
        "dim": retrieve.DIM,
        "strategies": list(STRATEGIES),
        "k": K,
        "seconds": round(time.monotonic() - t0, 1),
        "summary": summary,
        "questions": per_q,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nresults written: {out.relative_to(ROOT)} "
          f"({(time.monotonic() - t0):.1f}s total)")


if __name__ == "__main__":
    main()
