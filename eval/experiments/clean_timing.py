#!/usr/bin/env python3
"""Clean sequential timing: m3 N=10, m3 N=20, torch-quant N=10, torch-quant N=20.

Each config: 5 questions, torch.set_num_threads(6), one at a time.
Also: Spearman vs m3, score-file provenance, and routed table rows.
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
SCORES = ROOT / "eval" / "cache" / "rerank_scores.json"
PARSE = ROOT / "eval" / "results" / "2026-09-27_6410fb1_parse.json"
MODEL = "BAAI/bge-reranker-v2-m3"
K = 10
RRF_K = 60
TOPN = 50
DESC_CHARS = 1000
N_TIME = 5
TYPES = ["lookup", "topic", "filtered"]


def has_where(form: dict) -> bool:
    return any(form[k] is not None for k in
               ["priority", "issue_type", "open", "resolution",
                "created_from", "created_to"])


def load_data():
    lists = json.loads(LISTS.read_text())["questions"]
    lists_by_id = {r["id"]: r for r in lists}
    scores = json.loads(SCORES.read_text())
    parse = json.loads(PARSE.read_text())["results"]
    form_by_id = {r["id"]: r["form"] for r in parse}

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
    return lists, lists_by_id, scores, form_by_id, doc_text


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


def spearman_vs_m3(new_scores_by_q, m3_scores, lists):
    import numpy as np
    from scipy.stats import spearmanr
    m3_vals, new_vals = [], []
    for r in lists:
        rid = r["id"]
        for k in r["vec"][:TOPN]:
            m3_vals.append(m3_scores[rid]["scores"].get(k, 0.0))
            new_vals.append(new_scores_by_q.get(rid, {}).get(k, 0.0))
    m3_arr, new_arr = np.array(m3_vals, dtype=np.float64), np.array(new_vals, dtype=np.float64)
    rho, pval = spearmanr(m3_arr, new_arr)
    max_abs = float(np.abs(m3_arr - new_arr).max())
    mean_abs = float(np.abs(m3_arr - new_arr).mean())
    # count scores that differ by more than 1e-6
    ndiff = int(np.sum(np.abs(m3_arr - new_arr) > 1e-6))
    print(f"  Spearman rho = {rho:.6f}  (n={len(m3_arr)}, p={pval:.2e})")
    print(f"  max |diff| = {max_abs:.6f}   mean |diff| = {mean_abs:.6f}")
    print(f"  scores differing >1e-6: {ndiff}/{len(m3_arr)}")
    return rho


def main():
    lists, lists_by_id, m3_scores, form_by_id, doc_text = load_data()
    m3_flat = {rid: v["scores"] for rid, v in m3_scores.items()}
    ids = list(lists_by_id.keys())[:N_TIME]

    # ── 1. m3 timing ──────────────────────────────────────────────
    import torch
    torch.set_num_threads(6)
    print("=== m3 (fp32, 6 threads) ===", flush=True)
    from sentence_transformers import CrossEncoder
    model_m3 = CrossEncoder(MODEL, device="cpu", max_length=512)

    def time_m3(n_top):
        t0 = time.monotonic()
        for rid in ids:
            keys = lists_by_id[rid]["vec"][:n_top]
            pairs = [(lists_by_id[rid]["question"], doc_text[k])
                     for k in keys if k in doc_text]
            model_m3.predict(pairs, convert_to_scores=True)
        return (time.monotonic() - t0) / N_TIME

    s_m3_10 = time_m3(10)
    s_m3_20 = time_m3(20)
    print(f"  m3 s/q N=10: {s_m3_10:.2f}   N=20: {s_m3_20:.2f}", flush=True)

    # free m3 model
    del model_m3
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 2. torch-quant timing ─────────────────────────────────────
    from transformers import XLMRobertaForSequenceClassification, AutoTokenizer
    print("\n=== torch-quant (qint8, 6 threads) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model_tq = XLMRobertaForSequenceClassification.from_pretrained(MODEL)
    torch.ao.quantization.quantize_dynamic(model_tq, {torch.nn.Linear}, dtype=torch.qint8)
    model_tq.eval()

    def time_tq(n_top):
        t0 = time.monotonic()
        for rid in ids:
            keys = lists_by_id[rid]["vec"][:n_top]
            pairs = [(lists_by_id[rid]["question"], doc_text[k])
                     for k in keys if k in doc_text]
            enc = tok([p[0] for p in pairs], [p[1] for p in pairs],
                      return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                model_tq(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"]).logits
        return (time.monotonic() - t0) / N_TIME

    s_tq_10 = time_tq(10)
    s_tq_20 = time_tq(20)
    print(f"  torch-quant s/q N=10: {s_tq_10:.2f}   N=20: {s_tq_20:.2f}", flush=True)

    # ── 3. torch-quant score provenance ───────────────────────────
    # Re-score all 40q with torch-quant to get scores for the table
    print("\n  Re-scoring 40q with torch-quant for table rows …", flush=True)
    scores_by_q = {}
    for r in lists:
        rid = r["id"]
        keys = r["vec"][:TOPN]
        pairs = [(r["question"], doc_text[k]) for k in keys if k in doc_text]
        scored_keys = [k for k in keys if k in doc_text]
        enc = tok([p[0] for p in pairs], [p[1] for p in pairs],
                  return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            out = model_tq(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"]).logits
        sc = out.squeeze(-1).tolist()
        scores_by_q[rid] = dict(zip(scored_keys, sc))
        print(f"    [{rid}] done", flush=True)

    # Save torch-quant scores
    tq_path = ROOT / "eval" / "cache" / "rerank_scores_torch_quant.json"
    tq_out = {}
    for r in lists:
        rid = r["id"]
        tq_out[rid] = {
            "question": r["question"],
            "type": r["type"],
            "expected": r["expected"],
            "keys": r["vec"][:TOPN],
            "scores": scores_by_q[rid],
            "seconds": None,
        }
    tq_path.write_text(json.dumps(tq_out, indent=1))
    print(f"\n  torch-quant scores → {tq_path}", flush=True)

    # Spearman
    print("\n  torch-quant vs m3:", flush=True)
    spearman_vs_m3(scores_by_q, m3_scores, lists)

    # ── 4. Final table ────────────────────────────────────────────
    print(f"\n{'config':<32} {'s/q':>6}  "
          f"{'overall r@10':>13} {'overall MRR':>12}  "
          f"{'lookup r@10':>11} {'lookup MRR':>11}  "
          f"{'topic r@10':>10} {'topic MRR':>10}  "
          f"{'filt r@10':>10} {'filt MRR':>10}  "
          f"{'#worse':>6} {'#better':>7}")
    print("-" * 126)

    # m3 rows (from cached scores)
    for n, s in ((20, s_m3_20), (10, s_m3_10)):
        overall, by_type, worse, better = compute_routed(
            m3_flat, lists, lists_by_id, form_by_id, n)
        print_row(f"m3 N={n}", s, overall, by_type, worse, better)

    # torch-quant rows (from re-scored scores)
    for n, s in ((20, s_tq_20), (10, s_tq_10)):
        overall, by_type, worse, better = compute_routed(
            scores_by_q, lists, lists_by_id, form_by_id, n)
        print_row(f"torch-quant N={n}", s, overall, by_type, worse, better)

    print(f"\n  torch-quant score file: {tq_path}")


if __name__ == "__main__":
    main()
