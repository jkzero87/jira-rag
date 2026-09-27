"""Routed rerank, offline from cached scores.

Routing rule: rerank (blended RRF k=60) ONLY when the parser produced NO WHERE
clause (i.e. no priority, issue_type, open, resolution, created_from, created_to).
Otherwise keep the pure vector order.

Uses eval/results/2026-09-27_6410fb1_parse.json for the parser form.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

from metrics import recall_mrr, group_metrics  # noqa: E402

LISTS = ROOT / "eval" / "cache" / "lists.json"
SCORES = ROOT / "eval" / "cache" / "rerank_scores.json"
PARSE = ROOT / "eval" / "results" / "2026-09-27_6410fb1_parse.json"
K = 10
RRF_K = 60
DEPTHS = [20, 30]
TYPES = ["lookup", "topic", "filtered"]


def has_where(form: dict) -> bool:
    """True if any WHERE-producing field is non-null."""
    return any(form[k] is not None for k in
               ["priority", "issue_type", "open", "resolution",
                "created_from", "created_to"])


def main():
    lists = json.loads(LISTS.read_text())["questions"]
    lists_by_id = {r["id"]: r for r in lists}
    scores = json.loads(SCORES.read_text())
    parse = json.loads(PARSE.read_text())["results"]
    form_by_id = {r["id"]: r["form"] for r in parse}

    # baseline: w=0 vector top-10
    base = {}
    for r in lists:
        r10, mrr = recall_mrr(r["vec"][:K], r["expected"], K)
        base[r["id"]] = {"r10": r10, "mrr": mrr, "type": r["type"],
                         "vec": r["vec"]}

    for n in DEPTHS:
        per_q = {}
        for r in lists:
            rid = r["id"]
            cand = r["vec"][:n]
            sc = scores[rid]
            score = {k: sc["scores"].get(k, 0.0) for k in cand}

            if has_where(form_by_id[rid]):
                # routed: keep vector order
                order = list(cand)
            else:
                # blended: RRF of vector rank and reranker rank
                rank_vec = {k: i for i, k in enumerate(cand)}
                rank_rr = {k: i for i, k in enumerate(
                    sorted(cand, key=lambda k: -score[k]))}
                rrf = {k: 1.0 / (RRF_K + rank_vec[k] + 1)
                       + 1.0 / (RRF_K + rank_rr[k] + 1) for k in cand}
                order = sorted(cand, key=lambda k: (-rrf[k], rank_vec[k]))

            r10, mrr = recall_mrr(order[:K], r["expected"], K)
            per_q[rid] = {"r10": r10, "mrr": mrr, "type": r["type"],
                          "order": order[:K]}

        overall = group_metrics([per_q[q] for q in per_q])
        by_type = {t: group_metrics([per_q[q] for q in per_q if per_q[q]["type"] == t])
                  for t in TYPES}
        worse = sum(1 for q in base if per_q[q]["mrr"] < base[q]["mrr"] - 1e-9)
        better = sum(1 for q in base if per_q[q]["mrr"] > base[q]["mrr"] + 1e-9)

        print(f"\n=== Routed blended N={n} ===")
        print(f"{'scope':>10}  {'r@10':>7} {'MRR':>7}  {'#worse':>6} {'#better':>7}")
        print(f"{'overall':>10}  {overall['r10']:.4f} {overall['mrr']:.4f}  {worse:>6} {better:>7}")
        for t in TYPES:
            d = by_type[t]
            print(f"{t:>10}  {d['r10']:.4f} {d['mrr']:.4f}")

        # list every worse question with gold key rank before → after
        print(f"\n  Worse questions (MRR dropped):")
        for q in sorted(base):
            if per_q[q]["mrr"] < base[q]["mrr"] - 1e-9:
                vec = base[q]["vec"]
                order = per_q[q]["order"]
                gold = lists_by_id[q]["expected"]
                for g in gold:
                    r_before = vec.index(g) + 1 if g in vec else -1
                    r_after = order.index(g) + 1 if g in order else -1
                    tag = " " if r_after > r_before else "↓"
                    print(f"  {q} {g:<15} rank {r_before:>3} → {r_after:>3}  {tag}")

    lists_by_id = {r["id"]: r for r in lists}


if __name__ == "__main__":
    main()
