#!/usr/bin/env python3
"""Build router.text_dataset: the text-classifier dataset for the router.

One row per in-project issue (no flag_out_of_project) that has at least one
component (silver.issue_component). The router no longer depends on the
error signals (only ~12% of issues carry one); it is a text classifier
over all issues, so every qualifying issue goes in, with the four signals
kept as extra columns, reported separately.

Columns:
  issue_key, created_at (silver's corrected date), issue_type (silver.issue)
  text      = summary + "\\n" + description, both from bronze.jira_issue_raw
              (description is the empty string when null)
  components text[] (silver.issue_component, sorted)
  has_java_trace, has_python_trace, has_error_class, has_exception_name
              the four detectors already defined in scripts/router_dataset.py
              (imported, not re-implemented), each a boolean
  split     'train' when created_at < 2024-01-01 (UTC), 'test' when >=
  parent_components_now  text[], null for non-sub-tasks: the sub-task's
                         parent's CURRENT components (silver.issue_component)
  parent_components_asof text[], null for non-sub-tasks: the parent's
                         components AT THE SUB-TASK'S created_at, reconstructed
                         by undoing the parent's Component changelog (bronze)
                         newest-first (scripts/router_parent_context.py).
                         Never uses information from after created_at.

Reads bronze and silver only; writes only to the router schema.
Re-runnable: drops and rebuilds router.text_dataset and prints an identical
report on every run.

Run:
    .venv/bin/python scripts/router_text_dataset.py
"""
import datetime
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from profile_jira import load_env  # noqa: E402
from router_dataset import (  # noqa: E402
    exception_name, error_class, java_trace, python_trace)
from router_parent_context import parent_context  # noqa: E402

CUTOFF = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)

DDL = """
CREATE SCHEMA IF NOT EXISTS router;
DROP TABLE IF EXISTS router.text_dataset;
CREATE TABLE router.text_dataset (
    issue_key          text PRIMARY KEY,
    created_at         timestamptz NOT NULL,
    issue_type         text NOT NULL,
    text               text NOT NULL,
    components         text[] NOT NULL,
    has_java_trace     boolean NOT NULL,
    has_python_trace   boolean NOT NULL,
    has_error_class    boolean NOT NULL,
    has_exception_name boolean NOT NULL,
    split              text NOT NULL CHECK (split IN ('train', 'test')),
    parent_components_now  text[],
    parent_components_asof text[]
);
"""

ROWS_SQL = """
SELECT i.issue_key,
       i.created_at,
       i.issue_type,
       COALESCE(b.payload->'fields'->>'summary', ''),
       b.payload->'fields'->>'description',
       i.parent_key
FROM silver.issue i
LEFT JOIN bronze.jira_issue_raw b USING (issue_key)
WHERE NOT i.flag_out_of_project
"""

COMPS_SQL = """
SELECT issue_key, array_agg(component ORDER BY component)
FROM silver.issue_component
GROUP BY issue_key
"""


def main() -> None:
    env = load_env(REPO / ".env")
    import psycopg2

    conn = psycopg2.connect(
        host=env["PGHOST"], port=env["PGPORT"], dbname=env["PGDATABASE"],
        user=env["PGUSER"], password=env["PGPASSWORD"])
    cur = conn.cursor()

    cur.execute(DDL)
    cur.execute(ROWS_SQL)
    issues = cur.fetchall()
    cur.execute(COMPS_SQL)
    comp_map = {k: list(c) for k, c in cur.fetchall()}
    issues.sort(key=lambda r: r[0])

    eligible = []
    for key, created_at, issue_type, summary, desc, parent_key in issues:
        comps = comp_map.get(key)
        if not comps:
            continue  # population needs at least one component
        eligible.append((key, created_at, issue_type, summary, desc,
                         comps, parent_key))

    # Parent context: for sub-tasks, the parent's current components and its
    # components at the sub-task's created_at (no future information).
    subtasks = {e[0]: (e[6], e[1]) for e in eligible if e[6]}
    ctx, missing_parent = parent_context(cur, subtasks)

    train_rows, test_rows = [], []
    for key, created_at, issue_type, summary, desc, comps, parent_key in eligible:
        text = summary + "\n" + (desc if desc is not None else "")
        signals = (
            java_trace(desc) is not None,
            bool(python_trace(desc)),
            bool(error_class(desc)),
            bool(exception_name(desc)),
        )
        pnow, pasof = ctx.get(key, (None, None))
        split = "train" if created_at < CUTOFF else "test"
        row = (key, created_at, issue_type, text, comps, *signals,
               split, pnow, pasof)
        (train_rows if split == "train" else test_rows).append(row)

    cur.executemany(
        "INSERT INTO router.text_dataset"
        " (issue_key, created_at, issue_type, text, components,"
        "  has_java_trace, has_python_trace, has_error_class,"
        "  has_exception_name, split, parent_components_now,"
        "  parent_components_asof)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        train_rows + test_rows)
    conn.commit()
    conn.close()

    # --- report --------------------------------------------------------------
    print("=== router.text_dataset (in-project issues with >=1 component) ===")
    print(f"train (created_at <  2024-01-01): {len(train_rows)}"
          f"   multi-component: {sum(1 for r in train_rows if len(r[4]) > 1)}")
    print(f"test  (created_at >= 2024-01-01): {len(test_rows)}"
          f"   multi-component: {sum(1 for r in test_rows if len(r[4]) > 1)}")
    print(f"total:                          {len(train_rows) + len(test_rows)}"
          f"   multi-component:"
          f" {sum(1 for r in train_rows + test_rows if len(r[4]) > 1)}")

    train_components = {c for r in train_rows for c in r[4]}
    absent_issues = [r for r in test_rows if not set(r[4]) & train_components]
    absent_comps = Counter()
    for r in absent_issues:
        absent_comps.update(r[4])
    print()
    print(f"test issues with ALL components absent from train: "
          f"{len(absent_issues)} (of {len(test_rows)})")
    print("| component | issues |")
    print("|-----------|-------:|")
    for c, n in sorted(absent_comps.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"| {c} | {n} |")

    # --- parent context report (sub-tasks) ------------------------------------
    st_train = [r for r in train_rows if r[10] is not None]
    st_test = [r for r in test_rows if r[10] is not None]

    def changed_stats(rs):
        ch = sum(1 for r in rs if r[10] != r[11])
        pct = f"{100.0 * ch / len(rs):.1f}%" if rs else "0.0%"
        return ch, len(rs), pct

    ch_tr, n_tr, pct_tr = changed_stats(st_train)
    ch_te, n_te, pct_te = changed_stats(st_test)
    ch_all, n_all, pct_all = changed_stats(st_train + st_test)
    print()
    print("=== parent context (sub-tasks) ===")
    print(f"sub-tasks in dataset: train {n_tr}, test {n_te}"
          f"   parent missing from bronze: {missing_parent}")
    print(f"parent components changed after the sub-task was created"
          f" (parent_components_now != parent_components_asof):")
    print(f"  train: {ch_tr} of {n_tr} ({pct_tr})   "
          f"test: {ch_te} of {n_te} ({pct_te})   "
          f"total: {ch_all} of {n_all} ({pct_all})")
    examples = [r for r in train_rows + test_rows if r[10] != r[11]][:3]
    for key, created_at, _it, _text, _comps, *_sig, _sp, pnow, pasof in examples:
        print(f"  {key} created_at={created_at.isoformat()}  "
              f"now={pnow}  asof={pasof}")


if __name__ == "__main__":
    main()
