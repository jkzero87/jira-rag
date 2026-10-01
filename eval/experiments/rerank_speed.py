#!/usr/bin/env python3
"""Reranker speed / accuracy on bge-reranker-v2-m3 (CPU, 6 torch threads).

Usage:
  python eval/experiments/rerank_speed.py m3            # cached m3: rows N=20, N=10, worse list
  python eval/experiments/rerank_speed.py torch-quant   # PyTorch dynamic qint8, full vocab
  python eval/experiments/rerank_speed.py onnx          # ONNX optimum-cli + ORTQuantizer int8

Fixed columns (print_row):
  config | s/q | overall r@10 | overall MRR |
  lookup r@10 | lookup MRR | topic r@10 | topic MRR |
  filtered r@10 | filtered MRR | #worse | #better
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


# ── shared helpers ──────────────────────────────────────────────

def has_where(form: dict) -> bool:
    return any(form[k] is not None for k in
               ["priority", "issue_type", "open", "resolution",
                "created_from", "created_to"])


def print_header():
    print(f"{'config':<32} {'s/q':>6}  "
          f"{'overall r@10':>13} {'overall MRR':>12}  "
          f"{'lookup r@10':>11} {'lookup MRR':>11}  "
          f"{'topic r@10':>10} {'topic MRR':>10}  "
          f"{'filt r@10':>10} {'filt MRR':>10}  "
          f"{'#worse':>6} {'#better':>7}")
    print("-" * 126)


def print_row(config, s_per_q, overall, by_type, worse, better):
    print(f"{config:<32} {s_per_q:>6.2f}  "
          f"{overall['r10']:>13.4f} {overall['mrr']:>12.4f}  "
          f"{by_type['lookup']['r10']:>11.4f} {by_type['lookup']['mrr']:>11.4f}  "
          f"{by_type['topic']['r10']:>10.4f} {by_type['topic']['mrr']:>10.4f}  "
          f"{by_type['filtered']['r10']:>10.4f} {by_type['filtered']['mrr']:>10.4f}  "
          f"{worse:>6} {better:>7}")


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
    """Routed blended RRF.  Returns (overall, by_type, per_q, base, worse, better, worse_list).

    worse_list = [(qid, gold_key, rank_before, rank_after)]
    """
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

    worse_list = []
    for r in lists:
        rid = r["id"]
        vec = r["vec"]
        for g in r["expected"]:
            r_before = vec.index(g) + 1 if g in vec else -1
            if r_before > K:
                continue
            r_after = per_q[rid]["order"].index(g) + 1 if g in per_q[rid]["order"] else -1
            if r_after > r_before:
                worse_list.append((rid, g, r_before, r_after))
    return overall, by_type, per_q, base, worse, better, worse_list


def spearman_vs_m3(new_scores_by_q, m3_scores, lists):
    """Spearman rho + max/mean abs diff + exact top-10 match count vs m3 cache."""
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

    exact = 0
    for r in lists:
        rid = r["id"]
        m3_top10 = sorted(r["vec"][:TOPN],
                          key=lambda k: -m3_scores[rid]["scores"].get(k, 0.0))[:K]
        new_top10 = sorted(r["vec"][:TOPN],
                           key=lambda k: -new_scores_by_q.get(rid, {}).get(k, 0.0))[:K]
        if m3_top10 == new_top10:
            exact += 1

    print(f"  Spearman rho = {rho:.6f}  (n={len(m3_arr)}, p={pval:.2e})")
    print(f"  max |diff| = {max_abs:.6f}   mean |diff| = {mean_abs:.6f}")
    print(f"  exact top-10 match: {exact}/40")


# ── m3 (cached) ──────────────────────────────────────────────────

def run_m3():
    lists, lists_by_id, m3_scores, form_by_id, doc_text = load_data()
    # Flatten: {qid: {key: score}} for compute_routed
    m3_flat = {rid: v["scores"] for rid, v in m3_scores.items()}
    avg_s_cached = sum(v["seconds"] for v in m3_scores.values()) / len(m3_scores)

    # rows from cache
    rows = {}
    for n in (20, 10):
        overall, by_type, per_q, base, worse, better, worse_list = compute_routed(
            m3_flat, lists, lists_by_id, form_by_id, n)
        rows[n] = (overall, by_type, worse, better, worse_list)

    # time 5 questions at N=10
    print("\n  Timing m3 N=10 (5 questions, 6 threads):", flush=True)
    import torch
    torch.set_num_threads(6)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL, device="cpu", max_length=512)

    pairs_with_keys = {}
    for r in lists:
        keys = r["vec"][:20]
        pairs = [(r["question"], doc_text[k]) for k in keys if k in doc_text]
        pairs_with_keys[r["id"]] = (pairs, keys)

    ids = list(pairs_with_keys.keys())[:N_TIME]
    t0 = time.monotonic()
    for rid in ids:
        pairs = pairs_with_keys[rid][0][:10]
        model.predict(pairs, convert_to_scores=True)
    dt = time.monotonic() - t0
    s10 = dt / N_TIME
    print(f"  {dt:.1f}s / {N_TIME} q → {s10:.2f} s/q")

    # print table
    print_header()
    print_row(f"m3 N=20 (cached, scored N=50)", avg_s_cached,
              rows[20][0], rows[20][1], rows[20][2], rows[20][3])
    print_row(f"m3 N=10 (timed, N=10 scored)", s10,
              rows[10][0], rows[10][1], rows[10][2], rows[10][3])

    # worse list N=10
    print(f"\n  Worse list (m3 N=10, golds that moved down/out):")
    for qid, g, rb, ra in rows[10][4]:
        print(f"    {qid} {g:<15} {rb:>3} → {ra:>3}")


# ── PyTorch dynamic quantization ─────────────────────────────────

def run_torch_quant():
    lists, lists_by_id, m3_scores, form_by_id, doc_text = load_data()
    print("\n=== PyTorch dynamic quantization (qint8) ===", flush=True)

    import torch
    from transformers import XLMRobertaForSequenceClassification

    torch.set_num_threads(6)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL)
    torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()

    # score all 40 questions × top-50 pairs
    print("  Scoring 40 q × 50 pairs …", flush=True)
    scores_by_q = {}
    for r in lists:
        rid = r["id"]
        keys = r["vec"][:TOPN]
        pairs = [(r["question"], doc_text[k]) for k in keys if k in doc_text]
        scored_keys = [k for k in keys if k in doc_text]
        enc = tok([p[0] for p in pairs], [p[1] for p in pairs],
                  return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"]).logits
        sc = out.squeeze(-1).tolist()
        scores_by_q[rid] = dict(zip(scored_keys, sc))
        print(f"    [{rid}] done", flush=True)

    spearman_vs_m3(scores_by_q, m3_scores, lists)

    # timing: 5 questions at N=10 and N=20
    def time_tq(n_top):
        ids = list(lists_by_id.keys())[:N_TIME]
        t0 = time.monotonic()
        for rid in ids:
            keys = lists_by_id[rid]["vec"][:n_top]
            pairs = [(lists_by_id[rid]["question"], doc_text[k])
                     for k in keys if k in doc_text]
            enc = tok([p[0] for p in pairs], [p[1] for p in pairs],
                      return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                model(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"]).logits
        return (time.monotonic() - t0) / N_TIME

    print("\n  Timing torch-quant …", flush=True)
    s10 = time_tq(10)
    s20 = time_tq(20)
    print(f"  s/q N=10: {s10:.2f}   N=20: {s20:.2f}")

    print_header()
    for n, s in ((20, s20), (10, s10)):
        overall, by_type, per_q, base, worse, better, _ = compute_routed(
            scores_by_q, lists, lists_by_id, form_by_id, n)
        print_row(f"torch-quant N={n}", s, overall, by_type, worse, better)


# ── ONNX optimum-cli + ORTQuantizer ──────────────────────────────

def run_onnx():
    lists, lists_by_id, m3_scores, form_by_id, doc_text = load_data()
    print("\n=== ONNX (optimum-cli export + ORTQuantizer int8) ===", flush=True)

    from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
    import torch

    export_dir = str(ROOT / "eval" / "cache" / "reranker_v2m3_onnx")
    print("  Exporting to ONNX …", flush=True)

    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL)

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(export_path))

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "num_labels"},
    }
    dummy = tok("q", "d", return_tensors="pt", truncation=True, max_length=512)
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(export_path / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  Export OK → {export_dir}", flush=True)

    # Step 2: ONNX int8 quantization failed
    # onnxruntime.quantization.preprocess + quantize_dynamic both fail with:
    #   ShapeInferenceError: Inferred shape and existing shape differ in dimension 0: (1024) vs (1)
    # Dropping D2 (ONNX int8) — not viable with this export.
    print("  ONNX int8 quantization FAILED (shape inference error) — dropping D2", flush=True)
    return

    def time_ort(n_top):
        ids = list(lists_by_id.keys())[:N_TIME]
        t0 = time.monotonic()
        for rid in ids:
            keys = lists_by_id[rid]["vec"][:n_top]
            pairs = [(lists_by_id[rid]["question"], doc_text[k])
                     for k in keys if k in doc_text]
            enc = tok([p[0] for p in pairs], [p[1] for p in pairs],
                      return_tensors="np", truncation=True, max_length=512, padding=True)
            sess.run(None, {"input_ids": enc["input_ids"],
                            "attention_mask": enc["attention_mask"]})
        return (time.monotonic() - t0) / N_TIME

    print("\n  Timing onnx-int8 …", flush=True)
    s10 = time_ort(10)
    s20 = time_ort(20)
    print(f"  s/q N=10: {s10:.2f}   N=20: {s20:.2f}")

    print_header()
    for n, s in ((20, s20), (10, s10)):
        overall, by_type, per_q, base, worse, better, _ = compute_routed(
            scores_by_q, lists, lists_by_id, form_by_id, n)
        print_row(f"onnx-int8 N={n}", s, overall, by_type, worse, better)


# ── main ─────────────────────────────────────────────────────────

def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "m3"
    if method == "m3":
        run_m3()
    elif method == "torch-quant":
        run_torch_quant()
    elif method == "onnx":
        run_onnx()
    else:
        print(f"Unknown method: {method}")
        print("Usage: python eval/experiments/rerank_speed.py [m3|torch-quant|onnx]")
        sys.exit(1)


if __name__ == "__main__":
    main()
