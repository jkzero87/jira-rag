#!/usr/bin/env python3
"""Final consolidated table: config | s/q | overall r@10 | overall MRR | lookup MRR | topic MRR | filtered MRR

Loads:
  - m3 scores (d1000) from eval/cache/rerank_scores.json
  - torch-quant scores from eval/cache/rerank_scores_torch_quant.json
  - m3-d500 scores from eval/cache/rerank_scores_m3_d500.json

Timings from clean_timing.py output (hardcoded from verified run).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))
from metrics import recall_mrr, group_metrics  # noqa: E402

LISTS = ROOT / "eval" / "cache" / "lists.json"
PARSE = ROOT / "eval" / "results" / "2026-09-27_6410fb1_parse.json"
K = 10
RRF_K = 60
TOPN = 50
TOPN_D500 = 20
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
    return overall, by_type


def main():
    lists = json.loads(LISTS.read_text())["questions"]
    lists_by_id = {r["id"]: r for r in lists}
    parse = json.loads(PARSE.read_text())["results"]
    form_by_id = {r["id"]: r["form"] for r in parse}

    # Load scores
    m3_raw = json.loads((ROOT / "eval" / "cache" / "rerank_scores.json").read_text())
    m3_flat = {rid: v["scores"] for rid, v in m3_raw.items()}

    tq_raw = json.loads((ROOT / "eval" / "cache" / "rerank_scores_torch_quant.json").read_text())
    tq_flat = {rid: v["scores"] for rid, v in tq_raw.items()}

    d500_raw = json.loads((ROOT / "eval" / "cache" / "rerank_scores_m3_d500.json").read_text())
    d500_flat = {rid: v["scores"] for rid, v in d500_raw.items()}

    # Timings (from clean_timing.py, 6 threads, no other CPU jobs)
    timings = {
        "m3 N=20": 19.97,
        "m3 N=10": 8.63,
        "torch-quant N=20": 18.18,
        "torch-quant N=10": 8.28,
        "m3-d500 N=20": 7.96,
        "m3-d500 N=10": 3.58,
    }

    print(f"{'config':<22} {'s/q':>6}  {'overall r@10':>13} {'overall MRR':>12}  "
          f"{'lookup MRR':>11} {'topic MRR':>10}  {'filtered MRR':>12}")
    print("-" * 108)

    configs = [
        ("m3 N=20", m3_flat, 20),
        ("m3 N=10", m3_flat, 10),
        ("torch-quant N=20", tq_flat, 20),
        ("torch-quant N=10", tq_flat, 10),
        ("m3-d500 N=20", d500_flat, 20),
        ("m3-d500 N=10", d500_flat, 10),
    ]

    for config, scores, n in configs:
        overall, by_type = compute_routed(scores, lists, lists_by_id, form_by_id, n)
        print(f"{config:<22} {timings[config]:>6.2f}  "
              f"{overall['r10']:>13.4f} {overall['mrr']:>12.4f}  "
              f"{by_type['lookup']['mrr']:>11.4f} {by_type['topic']['mrr']:>10.4f}  "
              f"{by_type['filtered']['mrr']:>12.4f}")

    # Spearman summary
    import numpy as np
    from scipy.stats import spearmanr
    m3_vals, tq_vals = [], []
    for r in lists:
        rid = r["id"]
        for k in r["vec"][:TOPN]:
            m3_vals.append(m3_flat[rid].get(k, 0.0))
            tq_vals.append(tq_flat[rid].get(k, 0.0))
    rho, _ = spearmanr(np.array(m3_vals), np.array(tq_vals))
    max_diff = np.abs(np.array(m3_vals) - np.array(tq_vals)).max()
    ndiff = int(np.sum(np.abs(np.array(m3_vals) - np.array(tq_vals)) > 1e-6))
    print(f"\n  torch-quant vs m3: Spearman rho={rho:.6f}, max|diff|={max_diff:.4f}, "
          f"scores differing>1e-6: {ndiff}/{len(m3_vals)}")
    print(f"  torch-quant score file: eval/cache/rerank_scores_torch_quant.json")
    print(f"  m3-d500 score file: eval/cache/rerank_scores_m3_d500.json")


if __name__ == "__main__":
    main()
