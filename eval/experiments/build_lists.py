"""Part C — one retrieval per gold question, NO fusion.

For each question:
  * call retrieve.search_filtered(q, "summary_desc", TOPN)  — the SAME function
    run_eval uses for summary_desc_filtered
  * from its return value (results, form, where, filter_row_count):
      - results: top-100 vector list, WHERE-filtered  →  vec keys
  * call experimental._rare_kw_tsquery + experimental.KEYWORD_SQL for the keyword list
      (same code path as search_hybrid, but no fusion)

Write eval/cache/lists.json with, per question: id, type, question, expected,
where (the WHERE string from search_filtered), vec, kw.

No RRF fusion happens here.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

import retrieve
import experimental
import query_parse

GOLD_PATH = ROOT / "eval" / "gold.jsonl"
OUT_PATH = ROOT / "eval" / "cache" / "lists.json"
TOPN = 100
STRATEGY = "summary_desc"


def kw_list(q, where, params):
    """Top-100 keyword list (rare-lexeme OR), WHERE-filtered. Returns keys."""
    where_prefix = " AND " if where else ""
    kw_tsquery, kept = experimental._rare_kw_tsquery(q)
    if not kw_tsquery:
        return []
    sql = experimental.KEYWORD_SQL.format(where_prefix=where_prefix, where=where)
    conn = retrieve.psycopg2.connect(retrieve.DSN)
    try:
        cur = conn.cursor()
        kw_params = [kw_tsquery, STRATEGY] + list(params) + [kw_tsquery]
        cur.execute(sql, kw_params)
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def main():
    gold = [json.loads(l) for l in GOLD_PATH.read_text().splitlines() if l.strip()]
    retrieve.init()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []
    t0 = time.monotonic()
    for g in gold:
        q = g["question"]

        # Call the EXACT function run_eval uses for summary_desc_filtered.
        # Returns (results, form, where, filter_row_count) where results is
        # top-100 vector list (WHERE-filtered).
        results, form, where, filter_row_count = retrieve.search_filtered(
            q, STRATEGY, TOPN)
        vec_keys = [r[0] for r in results]

        # Keyword list: same WHERE, same strategy, same KEYWORD_SQL
        params = query_parse.to_sql(form)[1]
        kw_keys = kw_list(q, where, params)

        rec = {
            "id": g["id"],
            "type": g["type"],
            "question": q,
            "expected": g["expected"],
            "where": where,
            "filter_row_count": filter_row_count,
            "vec": vec_keys,
            "kw": kw_keys,
        }
        records.append(rec)
        print(f"[{g['id']}] {g['type']:10} vec={len(vec_keys):3d} "
              f"kw={len(kw_keys):3d}  filter_rows={filter_row_count:6d}  "
              f"where={where or '(none)'}", flush=True)

    payload = {
        "strategy": STRATEGY,
        "topn": TOPN,
        "n_questions": len(gold),
        "total_elapsed_s": round(time.monotonic() - t0, 2),
        "questions": records,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {OUT_PATH}  ({len(records)} questions, "
          f"{time.monotonic() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
