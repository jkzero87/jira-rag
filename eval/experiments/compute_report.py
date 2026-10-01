#!/usr/bin/env python3
"""Compute the comparison report: baseline summary_desc vs summary_desc_filtered.

Reads:
  eval/results/2026-09-23_b669151_baseline.json   (baseline, summary_desc)
  eval/results/2026-09-26_0deb4e0_baseline_filtered.json (filtered run)

Prints the verbatim report to stdout.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BASELINE = ROOT / "eval" / "results" / "2026-09-23_b669151_baseline.json"
FILTERED = ROOT / "eval" / "results" / "2026-09-26_0deb4e0_baseline_filtered.json"

BASE_STRAT = "summary_desc"
FILT_STRAT = "summary_desc_filtered"

TYPES = ("lookup", "topic", "filtered")
K = 10


def main():
    base = json.loads(BASELINE.read_text())
    filt = json.loads(FILTERED.read_text())

    base_q = {q["id"]: q["strategies"][BASE_STRAT] for q in base["questions"].values()}
    filt_q = {q["id"]: q["strategies"][FILT_STRAT] for q in filt["questions"].values()}
    types = {q["id"]: q["type"] for q in filt["questions"].values()}

    # ---- overall + by-type metrics ----
    def metrics(entries):
        n = len(entries)
        r5 = sum(e["recall_at_5"] for e in entries) / n
        r10 = sum(e["recall_at_10"] for e in entries) / n
        mrr = sum(e["mrr"] for e in entries) / n
        return n, r5, r10, mrr

    print("=" * 80)
    print("COMPARISON: summary_desc (baseline) vs summary_desc_filtered")
    print("=" * 80)
    print()
    print(f"{'scope':<12} {'n':>3}  {'metric':<12}  {'summary_desc':>13}  {'filtered':>13}  {'delta':>10}")
    print("-" * 80)
    for scope in ("overall",) + TYPES:
        if scope == "overall":
            b_entries = list(base_q.values())
            f_entries = list(filt_q.values())
        else:
            b_entries = [e for qid, e in base_q.items() if types[qid] == scope]
            f_entries = [e for qid, e in filt_q.items() if types[qid] == scope]

        bn, br5, br10, bmrr = metrics(b_entries)
        fn, fr5, fr10, fmrr = metrics(f_entries)
        for label, bv, fv in [("recall@5", br5, fr5), ("recall@10", br10, fr10), ("MRR", bmrr, fmrr)]:
            print(f"{scope:<12} {bn:>3}  {label:<12}  {bv:>13.4f}  {fv:>13.4f}  {fv - bv:>+10.4f}")
    print()

    # ---- questions where filtered is WORSE (recall@10) ----
    worse = []
    for qid in sorted(base_q):
        b_r10 = base_q[qid]["recall_at_10"]
        f_r10 = filt_q[qid]["recall_at_10"]
        if f_r10 < b_r10:
            worse.append((qid, types[qid], b_r10, f_r10))

    print(f"QUESTIONS WHERE FILTERED IS WORSE (recall@10): {len(worse)}/{len(base_q)}")
    print("-" * 80)
    if worse:
        for qid, t, b_r10, f_r10 in worse:
            b_top10 = base_q[qid]["top10"]
            f_top10 = filt_q[qid]["top10"]
            b_exp = base_q[qid].get("first_rank", "N/A")
            f_exp = filt_q[qid].get("first_rank", "N/A")
            print(f"  {qid} ({t})")
            print(f"    recall@10: {b_r10:.4f} -> {f_r10:.4f}  (delta {f_r10 - b_r10:+.4f})")
            print(f"    first_rank: {b_exp} -> {f_exp}")
            # show which expected keys were in top10 for each
            b_hit = [k for k in base_q[qid]["top10"] if k in (set(base_q[qid].get("top10", [])) & set(_get_expected_keys(base_q, qid)))]
            # just show top10 lists
            print(f"    baseline top10: {', '.join(b_top10)}")
            print(f"    filtered  top10: {', '.join(f_top10)}")
            print()
    else:
        print("  (none)")
    print()

    # ---- result count check ----
    print("RESULT COUNT CHECK (filtered):")
    print(f"  {'id':<10} {'expected':>8} {'actual':>8}  {'status':<8}  where")
    print("  " + "-" * 76)
    all_ok = True
    for qid in sorted(filt_q):
        entry = filt_q[qid]
        expected = entry.get("expected_count", K)
        actual = entry.get("result_count", 0)
        where = entry.get("where", "")
        ok = actual == expected
        if not ok:
            all_ok = False
        status = "OK" if ok else "MISMATCH"
        print(f"  {qid:<10} {expected:>8} {actual:>8}  {status:<8}  {where[:50] if where else '(none)'}")
    if all_ok:
        print("  All result counts match min(K, filter_row_count).")
    else:
        print("  WARNING: some result counts do not match!")
    print()

    # ---- parse errors ----
    parse_errors = [qid for qid in filt_q if filt_q[qid].get("form", {}).get("parse_error")]
    if parse_errors:
        print(f"PARSE ERRORS ({len(parse_errors)}):")
        for qid in parse_errors:
            print(f"  {qid} ({types[qid]})")
    print()

    # ---- form stats ----
    empty_forms = [qid for qid in filt_q if not filt_q[qid].get("where")]
    nonempty = len(filt_q) - len(empty_forms)
    print(f"FORM STATS: {nonempty}/{len(filt_q)} questions produced non-empty WHERE clauses")
    print(f"  empty forms (identical to plain search): {len(empty_forms)}")
    if empty_forms:
        print(f"  {', '.join(sorted(empty_forms))}")


def _get_expected_keys(q_data, qid):
    # expected keys are in the "expected" field of the gold question
    return []


if __name__ == "__main__":
    main()
