"""Summarize a parse-results JSON file (printed, computed from the data).

Usage: summarize_parse.py <results.json>
"""

import json
import sys
from pathlib import Path


def parse_gold_kept(s):
    """'x/y' -> (x, y)"""
    try:
        x, y = s.split("/")
        return int(x), int(y)
    except (ValueError, AttributeError):
        return None, None


def fmt_missing(keys):
    if not keys:
        return "-"
    return ", ".join(keys)


def main(path):
    data = json.loads(Path(path).read_text())
    results = data["results"]
    print(f"File: {path}")
    print(f"n_questions={data['n_questions']}  total_elapsed_s={data['total_elapsed_s']}  avg_per_question_s={data['avg_per_question_s']}")

    with_cand = [r for r in results if r.get("candidate_sql")]
    without_cand = [r for r in results if not r.get("candidate_sql")]

    # ---- Table 1: questions with candidate_sql ----
    print("\nTable 1: questions with candidate_sql")
    hdr = f"{'id':<6} {'rows':>5} {'kept':>6} {'exact':>6}  missing"
    print(hdr)
    print("-" * len(hdr))
    exact_count = 0
    for r in with_cand:
        exact = "yes" if r.get("exact_set_match") else "no"
        if r.get("exact_set_match"):
            exact_count += 1
        kept = r.get("gold_kept", "-")
        print(f"{r['id']:<6} {str(r.get('rows_returned', '-')):>5} {kept:>6} {exact:>6}  {fmt_missing(r.get('missing_gold'))}")

    # ---- Table 2: all other questions ----
    print("\nTable 2: all other questions")
    hdr = f"{'id':<6} {'empty':>6} {'rows':>5} {'kept':>6}  missing"
    print(hdr)
    print("-" * len(hdr))
    for r in without_cand:
        form = r.get("form")
        if form is None:
            empty = "err"
        else:
            # empty iff no field would produce a SQL clause.
            # NOTE: text_terms are no longer a WHERE condition (kept in the form
            # for retrieval/ranking), so they do not make the form non-empty here.
            nontrivial = (
                (form.get("priority") or [])
                or (form.get("issue_type") or [])
                or (form.get("resolution") or [])
                or form.get("open") is not None
                or form.get("created_from")
                or form.get("created_to")
            )
            empty = "no" if nontrivial else "yes"
        kept = r.get("gold_kept", "-")
        rows = r.get("rows_returned")
        rows_s = "-" if rows is None else str(rows)
        print(f"{r['id']:<6} {empty:>6} {rows_s:>5} {kept:>6}  {fmt_missing(r.get('missing_gold'))}")

    # ---- Totals ----
    exact_total = sum(1 for r in with_cand if r.get("exact_set_match"))
    print(f"\nTotals")
    print(f"  exact matches: {exact_total}/{len(with_cand)}")
    dropped = [r["id"] for r in results if r.get("missing_gold")]
    print(f"  questions that dropped ANY gold: {len(dropped)}  {dropped}")
    errs = [r["id"] for r in results if r.get("parse_error")]
    print(f"  parse errors: {len(errs)}  {errs}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: summarize_parse.py <results.json>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
