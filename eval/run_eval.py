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
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

import retrieve  # noqa: E402

STRATEGIES = ("summary_only", "summary_desc")
TYPES = ("lookup", "topic", "filtered")
K = 10


def question_metrics(keys, expected):
    """keys: top-K issue keys (nearest first)."""
    exp = set(expected)
    return {
        "recall_at_5": len(set(keys[:5]) & exp) / len(exp),
        "recall_at_10": len(set(keys[:K]) & exp) / len(exp),
        "mrr": (1.0 / r) if (r := next((i + 1 for i, k in enumerate(keys[:K]) if k in exp), None)) else 0.0,
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


def main():
    gold = [json.loads(line)
            for line in (ROOT / "eval" / "gold.jsonl").read_text().splitlines() if line.strip()]
    t0 = time.monotonic()

    retrieve.init()

    print(f"embedding {len(gold)} questions (once each) ...", flush=True)
    qvecs = {g["id"]: retrieve.embed_query(g["question"]) for g in gold}

    per_q = {}
    print(f"searching top {K} per strategy ({', '.join(STRATEGIES)}) ...", flush=True)
    for g in gold:
        per_q[g["id"]] = {"id": g["id"], "type": g["type"], "question": g["question"],
                          "expected": g["expected"], "strategies": {}}
        for s in STRATEGIES:
            rows = retrieve.search_vec(qvecs[g["id"]], s, K)
            entry = question_metrics([r[0] for r in rows], g["expected"])
            entry["top10"] = [r[0] for r in rows]
            entry["distances"] = [round(float(r[1]), 6) for r in rows]
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
