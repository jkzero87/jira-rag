"""Speed test for reranker options at N=20.

1. bge-reranker-v2-m3, torch.set_num_threads(6), max_length=512
2. BAAI/bge-reranker-base, torch.set_num_threads(6), max_length=512

For each: time 5 questions at N=20, score all 40 questions at N=20,
compute routed blended N=20 MRR row.

Also prints: torch.get_num_threads(), model max_length, avg tokens per pair.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

LISTS = ROOT / "eval" / "cache" / "lists.json"
PARSE = ROOT / "eval" / "results" / "2026-09-27_6410fb1_parse.json"
N = 20
K = 10
RRF_K = 60
HF_HUB_CACHE = str(ROOT / "eval" / "cache" / "hf_hub")

os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_HOME"] = HF_HUB_CACHE

import torch
from sentence_transformers import CrossEncoder
import psycopg2
from metrics import recall_mrr, group_metrics

TYPES = ["lookup", "topic", "filtered"]


def has_where(form):
    return any(form[k] is not None for k in
               ["priority", "issue_type", "open", "resolution",
                "created_from", "created_to"])


def load_data():
    lists = json.loads(LISTS.read_text())["questions"]
    lists_by_id = {r["id"]: r for r in lists}
    parse = json.loads(PARSE.read_text())["results"]
    form_by_id = {r["id"]: r["form"] for r in parse}
    vec_keys_by_id = {r["id"]: r["vec"][:N] for r in lists}

    conn = psycopg2.connect("")
    cur = conn.cursor()
    pairs_by_q = {}
    for r in lists:
        keys = vec_keys_by_id[r["id"]]
        cur.execute(
            "SELECT issue_key, summary, description FROM jira.issues "
            "WHERE issue_key = ANY(%s)", (keys,))
        text = {k: (s or "") + "\n" + (d or "")[:1000]
                for k, s, d in cur.fetchall()}
        pairs_by_q[r["id"]] = [(r["question"], text.get(k, "")) for k in keys]
    conn.close()
    return lists, lists_by_id, form_by_id, vec_keys_by_id, pairs_by_q


def score_all(pairs_by_q, vec_keys_by_id, model):
    scores = {}
    for rid, pairs in pairs_by_q.items():
        sc = model.predict(pairs, convert_to_scores=True)
        keys = vec_keys_by_id[rid]
        scores[rid] = {k: sc[i] for i, k in enumerate(keys) if pairs[i][1]}
    return scores


def compute_routed_mrr(scores, lists_by_id, form_by_id):
    per_q = {}
    for rid, r in lists_by_id.items():
        cand = r["vec"][:N]
        score = {k: scores[rid].get(k, 0.0) for k in cand}
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
        per_q[rid] = {"r10": r10, "mrr": mrr, "type": r["type"]}
    overall = group_metrics([per_q[q] for q in per_q])
    by_type = {t: group_metrics([per_q[q] for q in per_q if per_q[q]["type"] == t])
              for t in TYPES}
    return overall, by_type


def run_model(name, model_name, pairs_by_q, vec_keys_by_id,
              lists, lists_by_id, form_by_id, max_length=512, n_threads=6):
    print(f"\n=== {name} ===")
    torch.set_num_threads(n_threads)
    print(f"torch.get_num_threads() = {torch.get_num_threads()}")
    print(f"max_length = {max_length}")

    t0 = time.monotonic()
    model = CrossEncoder(model_name, device="cpu", max_length=max_length)
    print(f"load: {time.monotonic() - t0:.1f}s")

    # Token count
    tok = model.tokenizer
    all_tokens = []
    for rid, pairs in pairs_by_q.items():
        for q, d in pairs:
            if d:
                all_tokens.append(len(tok.encode(q, d, add_special_tokens=True)))
    avg_tok = sum(all_tokens) / len(all_tokens) if all_tokens else 0
    print(f"avg tokens/pair: {avg_tok:.1f}  (n={len(all_tokens)})")

    # Time 5 questions
    ids5 = list(pairs_by_q.keys())[:5]
    t0 = time.monotonic()
    for rid in ids5:
        model.predict(pairs_by_q[rid], convert_to_scores=True)
    dt5 = time.monotonic() - t0
    s_per_q = dt5 / len(ids5)
    print(f"5 questions N={N}: {dt5:.1f}s total, {s_per_q:.2f}s/q")

    # Score all 40
    t0 = time.monotonic()
    scores = score_all(pairs_by_q, vec_keys_by_id, model)
    dt_all = time.monotonic() - t0
    print(f"all 40 questions N={N}: {dt_all:.1f}s, {dt_all / 40:.2f}s/q")

    o, t = compute_routed_mrr(scores, lists_by_id, form_by_id)
    print(f"routed N={N}: overall {o['r10']:.4f}/{o['mrr']:.4f}  "
          f"lookup {t['lookup']['r10']:.4f}/{t['lookup']['mrr']:.4f}  "
          f"topic {t['topic']['r10']:.4f}/{t['topic']['mrr']:.4f}  "
          f"filtered {t['filtered']['r10']:.4f}/{t['filtered']['mrr']:.4f}")

    torch.set_num_threads(12)
    return name, s_per_q, o, t, avg_tok


def main():
    lists, lists_by_id, form_by_id, vec_keys_by_id, pairs_by_q = load_data()

    r1 = run_model("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3",
                   pairs_by_q, vec_keys_by_id, lists, lists_by_id, form_by_id)
    r2 = run_model("bge-reranker-base", "BAAI/bge-reranker-base",
                   pairs_by_q, vec_keys_by_id, lists, lists_by_id, form_by_id)

    # ONNX check
    print("\n=== Option 3: ONNX int8 ===")
    try:
        import onnxruntime
        from optimum.onnxruntime import ORTModelForSequenceClassification
        print("optimum + onnxruntime available")
    except ImportError:
        print("optimum/onnxruntime not installed — skipping")

    # Summary
    print("\n" + "=" * 90)
    print(f"{'model':>30} {'s/q@N20':>8} {'avg_tok':>8}  "
          f"{'overall MRR':>12} {'lookup MRR':>10} {'topic MRR':>10} {'filtered MRR':>12}")
    print("-" * 90)
    for name, s, o, t, tok in [r1, r2]:
        print(f"{name:>30} {s:>8.2f} {tok:>8.1f}  "
              f"{o['mrr']:>12.4f} {t['lookup']['mrr']:>10.4f} "
              f"{t['topic']['mrr']:>10.4f} {t['filtered']['mrr']:>12.4f}")


if __name__ == "__main__":
    main()
