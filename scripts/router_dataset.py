#!/usr/bin/env python3
"""Build router.dataset and measure the error signals in issue descriptions.

Part 1 — router.dataset (unchanged):
one row per issue whose description (bronze.jira_issue_raw,
fields.description) has a main Java trace package, per
profile_jira.main_package (section 6 of docs/data_profile.md), with at
least one component (silver.issue_component), without flag_out_of_project
(silver.issue), split on silver's corrected created_at:
'train' when created_at < 2024-01-01, 'test' when >= 2024-01-01.

Part 2 — signal measurement (read-only over bronze and silver):
Population: issues with at least one component and without
flag_out_of_project (an issue with a null description simply carries no
signal). One detector function per signal, over the description text:
  java_trace     a line "at org.apache.spark..." (profile_jira.main_package)
  python_trace   a Python traceback frame  File ".../pyspark/<package>/..."
  error_class    a bracketed Spark error code, uppercase words joined by
                 underscores, optionally dotted: [DIVIDE_BY_ZERO],
                 [UNRESOLVED_COLUMN.WITH_SUGGESTION]; a dotted code is
                 counted whole and also by the part before the dot
  exception_name a CamelCase name ending in "Exception" (AnalysisException,
                 SparkException, ...)

Report, in deterministic order:
  1) three real examples per signal (issue_key + matched line, first issues
     by issue_key in the population)
  2) per year (year of the corrected created_at, UTC): total issues, issues
     with each signal, issues with at least one signal, with % computed in
     Python
  3) top 15 error codes / pyspark packages, each with its most frequent
     component and the % of the population carrying it

Writes only to the router schema. Re-runnable: drops and rebuilds
router.dataset and prints an identical report on every run.

Run (re-runnable):
    .venv/bin/python scripts/router_dataset.py
"""
import datetime
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from profile_jira import TRACE_LINE_RE, load_env, main_package  # noqa: E402

CUTOFF = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)

# --- signal detectors (one function per signal, over the description) -------

# Python traceback frame:  File "<path>/pyspark/<package>/..."
# (the path may start at the pyspark directory or after a directory slash)
PYSPARK_FILE_RE = re.compile(r'(?m)^\s*File\s+"(?:[^"]*/)?pyspark/([A-Za-z0-9_]+)/')
# Bracketed uppercase token, optionally dotted: [DIVIDE_BY_ZERO],
# [UNRESOLVED_COLUMN.WITH_SUGGESTION]. A '['CODE'](' markdown link is not an
# error code, so an immediately following '(' is excluded.
ERROR_CODE_RE = re.compile(r"\[([A-Z][A-Z0-9_]*(?:\.[A-Z][A-Z0-9_]+)*)\](?!\()")
# CamelCase exception name, e.g. AnalysisException, SparkException
EXCEPTION_NAME_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*Exception)\b")


def java_trace(desc):
    """Main Java trace package of the description, or None (profile logic)."""
    if desc is None:
        return None
    return main_package(desc)


def python_trace(desc):
    """Set of pyspark packages named in Python `File` frames, or empty."""
    if desc is None:
        return set()
    return {m.group(1) for m in PYSPARK_FILE_RE.finditer(desc)}


def error_class(desc):
    """Set of Spark error codes: each dotted code plus the part before the dot.

    A candidate is kept only when it carries at least one underscore: that
    is what the signal spec says distinguishes a Spark error code from a
    plain [BRACKET] word.
    """
    if desc is None:
        return set()
    codes = set()
    for m in ERROR_CODE_RE.finditer(desc):
        code = m.group(1)
        if "_" not in code:
            continue
        codes.add(code)
        if "." in code:
            codes.add(code.split(".")[0])
    return codes


def exception_name(desc):
    """Set of CamelCase exception names (ending in 'Exception'), or empty."""
    if desc is None:
        return set()
    return set(EXCEPTION_NAME_RE.findall(desc))


# --- helpers for the examples -----------------------------------------------

def _matched_line(desc, pos, width=160):
    """The line containing position pos, stripped and truncated."""
    start = desc.rfind("\n", 0, pos) + 1
    end = desc.find("\n", pos)
    if end == -1:
        end = len(desc)
    line = desc[start:end].strip()
    if len(line) > width:
        line = line[:width] + "…"
    return line


def first_match_line(desc, regex):
    """The line holding the first regex match of desc, or None."""
    if desc is None:
        return None
    m = regex.search(desc)
    return _matched_line(desc, m.start()) if m else None


def first_error_code_line(desc):
    """The line holding the first valid error code, or None."""
    if desc is None:
        return None
    for m in ERROR_CODE_RE.finditer(desc):
        if "_" in m.group(1):
            return _matched_line(desc, m.start())
    return None


# label, presence test, first matched line
SIGNALS = (
    ("java_trace",
     lambda desc: java_trace(desc) is not None,
     lambda desc: first_match_line(desc, TRACE_LINE_RE)),
    ("python_trace",
     lambda desc: bool(python_trace(desc)),
     lambda desc: first_match_line(desc, PYSPARK_FILE_RE)),
    ("error_class",
     lambda desc: bool(error_class(desc)),
     first_error_code_line),
    ("exception_name",
     lambda desc: bool(exception_name(desc)),
     lambda desc: first_match_line(desc, EXCEPTION_NAME_RE)),
)

DDL = """
CREATE SCHEMA IF NOT EXISTS router;
DROP TABLE IF EXISTS router.dataset;
CREATE TABLE router.dataset (
    issue_key    text PRIMARY KEY,
    created_at   timestamptz NOT NULL,
    package      text NOT NULL,
    components   text[] NOT NULL,
    n_components integer NOT NULL,
    split        text NOT NULL CHECK (split IN ('train', 'test'))
);
"""

ROWS_SQL = """
SELECT i.issue_key,
       i.created_at,
       i.flag_out_of_project,
       b.payload->'fields'->>'description'
FROM silver.issue i
LEFT JOIN bronze.jira_issue_raw b USING (issue_key);
"""

COMPS_SQL = """
SELECT issue_key, array_agg(component ORDER BY component)
FROM silver.issue_component
GROUP BY issue_key;
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

    # --- Part 1: router.dataset (Java-trace issues only) ---------------------
    with_pkg = excl_no_comp = excl_oop = excl_both = 0
    train_rows = []
    test_rows = []
    for key, created_at, oop, desc in issues:
        pkg = main_package(desc) if desc is not None else None
        if pkg is None:
            continue
        with_pkg += 1
        comps = comp_map.get(key, [])
        no_comp = not comps
        if no_comp:
            excl_no_comp += 1
        if oop:
            excl_oop += 1
        if no_comp and oop:
            excl_both += 1
        if no_comp or oop:
            continue
        row = (key, created_at, pkg, comps, len(comps))
        if created_at < CUTOFF:
            train_rows.append(row + ("train",))
        else:
            test_rows.append(row + ("test",))

    cur.executemany(
        "INSERT INTO router.dataset (issue_key, created_at, package,"
        " components, n_components, split) VALUES (%s, %s, %s, %s, %s, %s)",
        train_rows + test_rows)
    conn.commit()

    multi_train = sum(1 for r in train_rows if r[4] > 1)
    multi_test = sum(1 for r in test_rows if r[4] > 1)
    print("=== router.dataset (Java-trace issues, components, in-project) ===")
    print(f"issues with a main trace package:            {with_pkg}")
    print(f"excluded, no component:                       {excl_no_comp}")
    print(f"excluded, flag_out_of_project:                {excl_oop}")
    print(f"excluded, both (no component AND out_of_proj): {excl_both}")
    print(f"kept:                                         "
          f"{len(train_rows) + len(test_rows)}")
    print(f"  train (created_at < 2024-01-01):            {len(train_rows)}"
          f"   multi-component: {multi_train}")
    print(f"  test  (created_at >= 2024-01-01):           {len(test_rows)}"
          f"   multi-component: {multi_test}")

    # --- Part 2: signal measurement over the population ----------------------
    # Population: >=1 component, no flag_out_of_project.
    pop = [(k, ts, d) for k, ts, oop, d in issues
           if not oop and comp_map.get(k)]
    pop_total = len(pop)
    print()
    print(f"=== signal measurement (population: {pop_total} issues with"
          f" >=1 component and no flag_out_of_project) ===")

    print("--- 3 examples per signal (first issues by issue_key) ---")
    for label, present, line_for in SIGNALS:
        print(label + ":")
        shown = 0
        for key, _ts, desc in pop:
            if desc is None or not present(desc):
                continue
            print(f"  {key}: {line_for(desc)}")
            shown += 1
            if shown == 3:
                break

    year_stats = {}
    code_issues = Counter()
    code_comps = {}
    pypk_issues = Counter()
    pypk_comps = {}
    for key, created_at, desc in pop:
        year = created_at.astimezone(datetime.timezone.utc).year
        vals = {
            "java_trace": java_trace(desc),
            "python_trace": python_trace(desc),
            "error_class": error_class(desc),
            "exception_name": exception_name(desc),
        }
        ys = year_stats.setdefault(year, Counter())
        ys["total"] += 1
        for label in vals:
            if vals[label]:
                ys[label] += 1
        if any(vals.values()):
            ys["any"] += 1
        comps = comp_map[key]
        for c in vals["error_class"]:
            code_issues[c] += 1
            code_comps.setdefault(c, Counter()).update(comps)
        for p in vals["python_trace"]:
            pypk_issues[p] += 1
            pypk_comps.setdefault(p, Counter()).update(comps)

    def cell(n, total):
        return f"{n} ({n * 100 / total:.1f}%)"

    print()
    print("--- per year (year of corrected created_at, UTC) ---")
    print("| year | issues | java_trace | python_trace | error_class"
          " | exception_name | >=1 signal |")
    print("|-----:|-------:|-----------:|-------------:|------------:"
          " |---------------:|-----------:|")
    tot = Counter()
    for year in sorted(year_stats):
        ys = year_stats[year]
        tot.update(ys)
        print(f"| {year} | {ys['total']} | {cell(ys['java_trace'], ys['total'])}"
              f" | {cell(ys['python_trace'], ys['total'])}"
              f" | {cell(ys['error_class'], ys['total'])}"
              f" | {cell(ys['exception_name'], ys['total'])}"
              f" | {cell(ys['any'], ys['total'])} |")
    print(f"| TOTAL | {tot['total']} | {cell(tot['java_trace'], tot['total'])}"
          f" | {cell(tot['python_trace'], tot['total'])}"
          f" | {cell(tot['error_class'], tot['total'])}"
          f" | {cell(tot['exception_name'], tot['total'])}"
          f" | {cell(tot['any'], tot['total'])} |")

    def top15(title, issues_counter, comps_by_name):
        print()
        print(f"--- {title} (population: {pop_total} issues) ---")
        print("| name | issues | top component | % of issues |")
        print("|------|-------:|---------------|------------:|")
        for name, n in sorted(issues_counter.items(),
                              key=lambda kv: (-kv[1], kv[0]))[:15]:
            top_comp = sorted(comps_by_name[name].items(),
                              key=lambda kv: (-kv[1], kv[0]))[0][0]
            print(f"| {name} | {n} | {top_comp} | {n * 100 / pop_total:.1f}% |")

    top15("top 15 error codes (error_class)", code_issues, code_comps)
    top15("top 15 pyspark packages (python_trace)", pypk_issues, pypk_comps)

    conn.close()


if __name__ == "__main__":
    main()
