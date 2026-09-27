"""Offline cross-encoder rerank of the vector top-50 (lists.json).

For each question in eval/cache/lists.json:
  * take rec["vec"][:50] (the w=0 vector list)
  * fetch (summary, description) for those keys from jira.issues (DB read-only)
  * rerank with BAAI/bge-reranker-v2-m3 on CPU:
        (question, summary + "\n" + first 1000 chars of description)
  * cache per-question scores to eval/cache/rerank_scores.json (resumable:
    already-scored questions are skipped on re-run)

Table for rerank depth N in {20, 30, 50}: shared metrics (metrics.py) on the
top-N of the reranked list — overall/lookup/topic/filtered r@10 and MRR,
#worse / #better vs the w=0 baseline (same lists.json), seconds per question.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

from metrics import recall_mrr, group_metrics  # noqa: E402

LISTS = ROOT / "eval" / "cache" / "lists.json"
OUT = ROOT / "eval" / "cache" / "rerank_scores.json"
DEPTHS = [20, 30, 50]
K = 10
DESC_CHARS = 1000
TOPN = 50
MODEL = "BAAI/bge-reranker-v2-m3"


def main():
    lists = json.loads(LISTS.read_text())["questions"]
    gold_by_id = {r["id"]: r for r in lists}

    # --- load or refresh the score cache ---
    scores = {}
    if OUT.exists():
        scores = json.loads(OUT.read_text())

    todo = [r for r in lists if r["id"] not in scores]
    if todo:
        import psycopg2
        from sentence_transformers import CrossEncoder

        conn = psycopg2.connect("")  # env DSN from .env (PGHOST etc.)
        cur = conn.cursor()
        model = CrossEncoder(MODEL, device="cpu")

        t0 = time.monotonic()
        for rec in todo:
            keys = rec["vec"][:TOPN]
            cur.execute(
                "SELECT issue_key, summary, description FROM jira.issues "
                "WHERE issue_key = ANY(%s)", (keys,))
            text = {k: (summary or "") + "\n" + (description or "")[:DESC_CHARS]
                    for k, summary, description in cur.fetchall()}
            pairs = [(rec["question"], text[k]) for k in keys if k in text]
            keys = [k for k in keys if k in text]
            t_q = time.monotonic()
            s = model.predict(pairs, convert_to_scores=True)
            dt = time.monotonic() - t_q
            scores[rec["id"]] = {
                "question": rec["question"],
                "type": rec["type"],
                "expected": rec["expected"],
                "keys": keys,                     # top-50 in vector order
                "scores": {k: float(sc) for k, sc in zip(keys, s)},
                "seconds": round(dt, 2),
            }
            print(f"[{rec['id']}] {rec['type']:10} n={len(keys):3d} "
                  f"{dt:7.2f}s", flush=True)
            OUT.write_text(json.dumps(scores))
        conn.close()
        print(f"scored {len(todo)} questions in {time.monotonic() - t0:.1f}s")
    else:
        print(f"all {len(lists)} questions already in cache: {OUT.name}")

    # --- baseline (w=0): vector top-10 from lists.json ---
    base = {}
    for r in lists:
        r10, mrr = recall_mrr(r["vec"], r["expected"], K)
        base[r["id"]] = {"r10": r10, "mrr": mrr, "type": r["type"]}

    # --- reranked top-N, table ---
    per_n = {n: {} for n in DEPTHS}
    for r in lists:
        sc = scores[r["id"]]
        order = sorted(sc["keys"], key=lambda k: -sc["scores"][k])
        for n in DEPTHS:
            r10, mrr = recall_mrr(order[:n], r["expected"], K)
            per_n[n][r["id"]] = {"r10": r10, "mrr": mrr, "type": r["type"]}

    types = ["lookup", "topic", "filtered"]
    print()
    print(f"{'N':>3}  {'overall r@10/MRR':>18}  {'lookup r@10/MRR':>17}  "
          f"{'topic r@10/MRR':>16}  {'filtered r@10/MRR':>18}  {'#worse':>6} "
          f"{'#better':>7}  {'s/q':>6}")
    for n in DEPTHS:
        d = per_n[n]
        entries = [{"r10": d[q]["r10"], "mrr": d[q]["mrr"]} for q in d]
        overall = group_metrics(entries)
        by_type = {t: group_metrics([{"r10": d[q]["r10"], "mrr": d[q]["mrr"]}
                                     for q in d if d[q]["type"] == t])
                   for t in types}
        worse = sum(1 for q in base if d[q]["mrr"] < base[q]["mrr"] - 1e-9)
        better = sum(1 for q in base if d[q]["mrr"] > base[q]["mrr"] + 1e-9)
        avg_s = sum(scores[q]["seconds"] for q in scores) / len(scores)
        print(f"{n:>3}  {overall['r10']:.4f}/{overall['mrr']:.4f}  "
              f"{by_type['lookup']['r10']:.4f}/{by_type['lookup']['mrr']:.4f}  "
              f"{by_type['topic']['r10']:.4f}/{by_type['topic']['mrr']:.4f}  "
              f"{by_type['filtered']['r10']:.4f}/{by_type['filtered']['mrr']:.4f}  "
              f"{worse:>6} {better:>7}  {avg_s:>6.2f}")
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
