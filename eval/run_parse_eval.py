"""Parse-eval: run the query parser on all 40 gold questions, run the resulting
SQL filters against the DB, and compare results to the gold issue sets.

Output: eval/results/<date>_<commit>_parse.json  (never overwritten)
"""

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import psycopg2

# Make src/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jira_rag.query_parse import parse, to_sql

# ---------------------------------------------------------------------------
# Load gold
# ---------------------------------------------------------------------------
GOLD_PATH = ROOT / "eval" / "gold.jsonl"


def load_gold():
    items = []
    with open(GOLD_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------------
# Run candidate_sql for exact-set comparison (only for questions that have one)
# ---------------------------------------------------------------------------
def run_candidate_sql(cur, candidate_sql):
    """Run the stored candidate_sql as a WHERE clause and return the set of keys."""
    cur.execute(
        f"SELECT issue_key FROM jira.issues WHERE {candidate_sql} ORDER BY issue_id"
    )
    return [r[0] for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    gold = load_gold()
    conn = psycopg2.connect("")
    cur = conn.cursor()

    results = []
    total_start = time.time()

    for item in gold:
        qid = item["id"]
        qtype = item["type"]
        question = item["question"]
        expected = item["expected"]
        candidate_sql = item.get("candidate_sql")

        t0 = time.time()

        # 1. Parse
        # Thinking-off is now the default in parse().  The PARSE_DISABLE_THINKING
        # env var is kept for backward compatibility (redundant when set to 1).
        enable_thinking = False  # thinking off by default
        form = parse(question, enable_thinking=enable_thinking)
        if form.get("parse_error"):
            sys.exit(f"error: [{qid}] parse_error (invalid model output); "
                     "refusing to score an all-null fallback form")
        elapsed = time.time() - t0

        # 2. to_sql + run
        where, params = to_sql(form)
        sql = "SELECT issue_key FROM jira.issues"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY issue_id"

        try:
            cur.execute(sql, params)
            rows = [r[0] for r in cur.fetchall()]
        except Exception as exc:
            rows = []
            sql_error = str(exc)
        else:
            sql_error = None

        # 3. Gold kept
        gold_set = set(expected)
        row_set = set(rows)
        kept = len(gold_set & row_set)
        missing_gold = sorted(gold_set - row_set)

        # 4. Exact-set match with candidate_sql (only if it exists)
        exact_match = None
        cand_missing = None
        cand_extra = None
        if candidate_sql:
            cand_rows = run_candidate_sql(cur, candidate_sql)
            cand_set = set(cand_rows)
            exact_match = (row_set == cand_set)
            cand_missing = sorted(cand_set - row_set)   # in candidate, not in ours
            cand_extra = sorted(row_set - cand_set)      # in ours, not in candidate

        # Extra keys = rows not in gold (at most 20 stored, total count stored)
        extra = [k for k in rows if k not in gold_set]
        result = {
            "id": qid,
            "type": qtype,
            "question": question,
            "expected": expected,
            "form": form,
            "parse_error": form["parse_error"],
            "sql_error": sql_error,
            "rows_returned": len(rows),
            "gold_kept": f"{kept}/{len(expected)}",
            "missing_gold": missing_gold,
            "extra_keys": extra[:20],
            "extra_count": len(extra),
            "elapsed_s": round(elapsed, 2),
        }
        if candidate_sql:
            result["candidate_sql"] = candidate_sql
            result["candidate_rows"] = len(cand_rows)
            result["exact_set_match"] = exact_match
            result["cand_missing"] = cand_missing  # keys in candidate, not in ours
            result["cand_extra"] = cand_extra  # keys in ours, not in candidate

        results.append(result)
        status = "OK" if not missing_gold else f"MISS {missing_gold}"
        print(
            f"[{qid}] {qtype:<10} rows={len(rows):>4}  kept={kept}/{len(expected)}  "
            f"{elapsed:>5.1f}s  {status}",
            flush=True,
        )

    total_elapsed = time.time() - total_start
    avg = total_elapsed / len(gold) if gold else 0

    # -----------------------------------------------------------------------
    # Save results (never overwrite)
    # -----------------------------------------------------------------------
    commit = os.popen("git rev-parse --short HEAD").read().strip()
    today = date.today().isoformat()
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{today}_{commit}_parse.json"

    if out_path.exists():
        # Add a numeric infix to avoid overwrite
        n = 2
        while True:
            candidate = out_dir / f"{out_path.stem}_{n}.json"
            if not candidate.exists():
                out_path = candidate
                break
            n += 1

    payload = {
        "model": os.environ.get("LLAMA_MODEL", "default"),
        "llama_url": os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8092"),
        "enable_thinking": False,
        "snapshot_date": "2026-09-18",
        "n_questions": len(gold),
        "total_elapsed_s": round(total_elapsed, 2),
        "avg_per_question_s": round(avg, 2),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Filtered questions
    filtered = [r for r in results if r["type"] == "filtered"]
    exact_count = sum(1 for r in filtered if r.get("exact_set_match"))
    print(f"\nFiltered questions: exact set match {exact_count}/{len(filtered)}")
    for r in filtered:
        if r.get("candidate_sql"):
            tag = "EXACT" if r["exact_set_match"] else "DIFFER"
            print(f"  {r['id']:<5} {tag:6} rows={r['rows_returned']:>4}  "
                  f"kept={r['gold_kept']}  missing={r['missing_gold']}")

    # G24 (judged, not in the 7)
    g24 = [r for r in results if r["id"] == "G24"]
    if g24:
        r = g24[0]
        tag = "EXACT" if r.get("exact_set_match") else "DIFFER"
        print(f"  G24   {tag:6} rows={r['rows_returned']:>4}  "
              f"kept={r['gold_kept']}  missing={r['missing_gold']}")

    # Non-filtered questions: empty form (no SQL clause) vs non-empty
    non_filtered = [r for r in results if r["type"] != "filtered"]
    empty = [r for r in non_filtered
             if r["form"] and not to_sql(r["form"])[0]]
    nonempty = [r for r in non_filtered if r not in empty]
    print(f"\nNon-filtered questions: {len(non_filtered)} total")
    print(f"  Empty form (no filters):  {len(empty)}")
    print(f"  Non-empty form (filters): {len(nonempty)}")
    for r in nonempty:
        print(f"    {r['id']:<5} form={json.dumps(r['form'])}")

    # Questions where the form DROPPED a gold issue (gold not in rows)
    dropped = [r for r in results if r.get("missing_gold")]
    print(f"\nQuestions where form dropped gold: {len(dropped)}")
    for r in dropped:
        print(f"  {r['id']:<5} {r['type']:<10} missing={r['missing_gold']}")
        print(f"         form={json.dumps(r['form'])}")

    conn.close()


if __name__ == "__main__":
    main()
