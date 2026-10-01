#!/usr/bin/env python3
"""Re-score m3 (fp32, 6 threads) with summary + first 500 chars of description.

Vector top-20 for all 40 questions.
Cache → eval/cache/rerank_scores_m3_d500.json
Print: avg tokens/pair, s/q N=10 & N=20, routed rows N=20 & N=10.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))
from metrics import recall_mrr, group_metrics  # noqa: E402

LISTS = ROOT / "eval" / "cache" / "lists.json"
PARSE = ROOT / "eval" / "results" / "2026-09-27_6410fb1_parse.json"
MODEL = "BAAI/bge-reranker-v2-m3"
K = 10
RRF_K = 60
TOPN = 20
DESC_CHARS = 500
N_TIME = 5
TYPES = ["lookup", "topic", "filtered"]


def has_where(form: dict) -> bool:
    return any(form[k] is not None for k in
               ["priority", "issue_type", "open", "resolution",
                "created_from", "created_to"])


def compute_routed(scores_by_q, lists, lists_by_id, form_by_id, n):
    base = {}
    per_q = {}
    for r in lists:
        rid = r["id"]
        vec = r["vec"]
        r10b, mrrb = recall_mrr(vec[:K], r["expected"], K)
        base[rid] = {"r10": r10b, "mrr": mrrb, "type": r["type"], "vec": vec}
        cand = vec[:n]
        score = {k: scores_by_q.get(rid, {}).get(k, 0.0) for k in cand}
        if has_where(form_by_id[rid]):
            order = list(cand)
        else:
            rank_vec = {k: i for i, k in enumerate(cand)}
            rank_rr = {k: i for i, k in enumerate(
                sorted(cand, key=lambda k: -score[k]))}
            rrf = {k: 1.0 / (RRF_K + rank_vec[k] + 1)
                   + 1.0 / (RRF_K + rank_rr[k] + 1) for k in cand}
            order = sorted(cand, key=lambda k: (-rrf[k], rank_vec[k]))
        r10, mrr = recall_mrr(order[:K], r["expected"], K)
        per_q[rid] = {"r10": r10, "mrr": mrr, "type": r["type"], "order": order[:K]}
    overall = group_metrics([per_q[q] for q in per_q])
    by_type = {t: group_metrics([per_q[q] for q in per_q if per_q[q]["type"] == t])
               for t in TYPES}
    worse = sum(1 for q in base if per_q[q]["mrr"] < base[q]["mrr"] - 1e-9)
    better = sum(1 for q in base if per_q[q]["mrr"] > base[q]["mrr"] + 1e-9)
    return overall, by_type, worse, better


def print_row(config, s_per_q, overall, by_type, worse, better):
    print(f"{config:<32} {s_per_q:>6.2f}  "
          f"{overall['r10']:>13.4f} {overall['mrr']:>12.4f}  "
          f"{by_type['lookup']['r10']:>11.4f} {by_type['lookup']['mrr']:>11.4f}  "
          f"{by_type['topic']['r10']:>10.4f} {by_type['topic']['mrr']:>10.4f}  "
          f"{by_type['filtered']['r10']:>10.4f} {by_type['filtered']['mrr']:>10.4f}  "
          f"{worse:>6} {better:>7}")


def main():
    lists = json.loads(LISTS.read_text())["questions"]
    lists_by_id = {r["id"]: r for r in lists}
    parse = json.loads(PARSE.read_text())["results"]
    form_by_id = {r["id"]: r["form"] for r in parse}

    # Load doc text with 500-char description
    import psycopg2
    conn = psycopg2.connect("")
    cur = conn.cursor()
    doc_text = {}
    for r in lists:
        keys = r["vec"][:TOPN]
        cur.execute("SELECT issue_key, summary, description FROM jira.issues "
                    "WHERE issue_key = ANY(%s)", (keys,))
        for k, s, d in cur.fetchall():
            doc_text[k] = (s or "") + "\n" + (d or "")[:DESC_CHARS]
    conn.close()

    import torch
    torch.set_num_threads(6)
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = CrossEncoder(MODEL, device="cpu", max_length=512)

    # Score all 40q × top-20 pairs
    cache_path = ROOT / "eval" / "cache" / "rerank_scores_m3_d500.json"
    scores_by_q = {}
    total_tokens = 0
    total_pairs = 0

    print("=== m3 d500 (fp32, 6 threads, desc[:500], top-20) ===", flush=True)
    for r in lists:
        rid = r["id"]
        keys = r["vec"][:TOPN]
        pairs = [(r["question"], doc_text[k]) for k in keys if k in doc_text]
        scored_keys = [k for k in keys if k in doc_text]
        t0 = time.monotonic()
        sc = model.predict(pairs, convert_to_scores=True).tolist()
        dt = time.monotonic() - t0
        scores_by_q[rid] = dict(zip(scored_keys, sc))
        # count tokens
        enc = tok([p[0] for p in pairs], [p[1] for p in pairs],
                  return_tensors="pt", truncation=True, max_length=512, padding=True)
        total_tokens += enc["input_ids"].shape[0] * enc["input_ids"].shape[1]
        total_pairs += len(pairs)
        print(f"  [{rid}] {len(pairs)} pairs, {dt:.2f}s", flush=True)

    avg_tokens = total_tokens / total_pairs if total_pairs else 0
    print(f"\n  avg tokens/pair: {avg_tokens:.1f}  (total {total_tokens} tokens / {total_pairs} pairs)", flush=True)

    # Save cache
    out = {}
    for r in lists:
        rid = r["id"]
        out[rid] = {
            "question": r["question"],
            "type": r["type"],
            "expected": r["expected"],
            "keys": r["vec"][:TOPN],
            "scores": scores_by_q[rid],
            "desc_chars": DESC_CHARS,
            "n_candidates": TOPN,
        }
    cache_path.write_text(json.dumps(out, indent=1))
    print(f"  cached → {cache_path}", flush=True)

    # Timing: 5 questions at N=10 and N=20
    ids = list(lists_by_id.keys())[:N_TIME]

    def time_it(n_top):
        t0 = time.monotonic()
        for rid in ids:
            keys = lists_by_id[rid]["vec"][:n_top]
            pairs = [(lists_by_id[rid]["question"], doc_text[k])
                     for k in keys if k in doc_text]
            model.predict(pairs, convert_to_scores=True)
        return (time.monotonic() - t0) / N_TIME

    s10 = time_it(10)
    s20 = time_it(20)
    print(f"\n  s/q N=10: {s10:.2f}   N=20: {s20:.2f}", flush=True)

    # Table
    print(f"\n{'config':<32} {'s/q':>6}  "
          f"{'overall r@10':>13} {'overall MRR':>12}  "
          f"{'lookup r@10':>11} {'lookup MRR':>11}  "
          f"{'topic r@10':>10} {'topic MRR':>10}  "
          f"{'filt r@10':>10} {'filt MRR':>10}  "
          f"{'#worse':>6} {'#better':>7}")
    print("-" * 126)
    for n, s in ((20, s20), (10, s10)):
        overall, by_type, worse, better = compute_routed(
            scores_by_q, lists, lists_by_id, form_by_id, n)
        print_row(f"m3-d500 N={n}", s, overall, by_type, worse, better)


if __name__ == "__main__":
    main()
