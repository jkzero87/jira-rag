#!/usr/bin/env python3
"""Profile the jira.issues table and regenerate docs/data_profile.md.

Read-only: the script opens a normal connection, runs only SELECTs and
never writes to any table. It connects to the compose database
(127.0.0.1:5433) using the credentials from the repo .env file
(PGDATABASE / PGUSER / PGPASSWORD).

Run (re-runnable, overwrites the document):
    .venv/bin/python scripts/profile_jira.py
"""
import argparse
import datetime
import json
import re
from collections import Counter
from pathlib import Path

import psycopg2

REPO = Path(__file__).resolve().parent.parent


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def esc(s, limit: int = 60) -> str:
    s = re.sub(r"\s+", " ", str(s)).replace("|", "\\|")
    return s if len(s) <= limit else s[: limit - 1] + "..."


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def pct(part: int, total: int) -> str:
    return f"{100.0 * part / total:.1f}%" if total else "0.0%"


def section_columns(cur, total: int) -> str:
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'jira' AND table_name = 'issues'
        ORDER BY ordinal_position""")
    rows = []
    for col, dtype in cur.fetchall():
        is_text = dtype in ("text", "character varying")
        empty = f"{col} IS NULL" + (f" OR btrim({col}) = ''" if is_text else "")
        cur.execute(f"SELECT count(*) FILTER (WHERE {empty}) FROM jira.issues")
        n_empty = cur.fetchone()[0]
        if is_text or dtype == "jsonb":
            cur.execute(f"SELECT count(DISTINCT md5({col}::text)) FROM jira.issues")
        else:
            cur.execute(f"SELECT count(DISTINCT {col}) FROM jira.issues")
        n_distinct = cur.fetchone()[0]
        where = f"{col} IS NOT NULL" + (f" AND btrim({col}) <> ''" if is_text else "")
        cur.execute(f"SELECT {col}::text FROM jira.issues WHERE {where} LIMIT 3")
        examples = [esc(r[0]) for r in cur.fetchall()] or ["-"]
        rows.append((f"`{col}`", f"`{dtype}`", pct(n_empty, total),
                     f"{n_distinct:,}", " / ".join(examples)))
    return (
        "## 1. Column profile of `jira.issues`\n\n"
        f"Total issues: **{total:,}** (project SPARK).\n\n"
        + md_table(["Column", "Type", f"Null/empty, % of all issues ({total:,})", "Distinct values", "Examples"], rows)
        + "\n"
    )


def section_raw_keys(cur, total: int) -> str:
    cur.execute("""
        SELECT DISTINCT k
        FROM jira.issues, LATERAL jsonb_object_keys(raw) k
        WHERE raw IS NOT NULL""")
    keys = [r[0] for r in cur.fetchall()]
    parts, params = [], []
    for k in keys:
        parts.append(
            "count(*) FILTER (WHERE raw ? %s"
            " AND jsonb_typeof(raw->%s) <> 'null'"
            " AND raw->%s <> '{}'::jsonb"
            " AND NOT (jsonb_typeof(raw->%s) = 'array'"
            "   AND jsonb_array_length(raw->%s) = 0)"
            " AND NOT (jsonb_typeof(raw->%s) = 'string' AND btrim(raw->>%s) = ''))"
        )
        params += [k] * 7
    cur.execute("SELECT " + ", ".join(parts) + " FROM jira.issues", params)
    counts = dict(zip(keys, cur.fetchone()))
    named = sorted((k for k in keys if not k.startswith("customfield_")),
                   key=lambda k: -counts[k])
    custom = sorted((k for k in keys if k.startswith("customfield_")),
                    key=lambda k: -counts[k])
    out = (
        "## 2. Top-level keys of `fields` in the `raw` column\n\n"
        "The `raw jsonb` column stores the full `fields` object of each issue\n"
        "(see `src/jira_rag/ingest.py`). A key counts as *non-empty* when it is\n"
        "present, not `null`, and not an empty string/array/object.\n\n"
        f"{len(keys)} distinct keys appear across all issues\n"
        f"({len(named)} named fields, {len(custom)} `customfield_*` fields).\n\n"
        "### 2.1 Named fields\n\n"
        + md_table(["Key", f"Non-empty, % of all issues ({total:,})"],
                   [(f"`{k}`", pct(counts[k], total)) for k in named])
        + "\n\n### 2.2 Custom fields (`customfield_*`, no name)\n\n"
        + md_table(["Key", f"Non-empty, % of all issues ({total:,})"],
                   [(f"`{k}`", pct(counts[k], total)) for k in custom])
        + "\n"
    )
    return out


def section_status(cur) -> str:
    cur.execute("""
        SELECT COALESCE(status, '(no value)'),
               COALESCE(resolution, '(no value)'),
               COALESCE(issue_type, '(no value)'),
               count(*)
        FROM jira.issues
        GROUP BY 1, 2, 3""")
    total = 0
    status, resolution, itype = Counter(), Counter(), Counter()
    for s, r, t, c in cur.fetchall():
        total += c
        status[s] += c
        resolution[r] += c
        itype[t] += c

    def table(title: str, counter: Counter) -> str:
        rows = [(f"`{k}`", f"{v:,}", pct(v, total))
                for k, v in counter.most_common()]
        return f"### {title}\n\n" + md_table(["Value", "Count", f"% of all issues ({total:,})"], rows) + "\n"

    return ("## 3. Status, resolution and issue type\n\n"
            + table("Status", status)
            + table("Resolution", resolution)
            + table("Issue type", itype))


def section_components(cur, total: int) -> str:
    cur.execute("""
        SELECT count(*) FILTER (WHERE jsonb_typeof(raw->'components') = 'array'
               AND jsonb_array_length(raw->'components') > 0)
        FROM jira.issues""")
    with_c = cur.fetchone()[0]
    comps_expr = (
        "CASE WHEN jsonb_typeof(raw->'components') = 'array'"
        " THEN raw->'components' ELSE '[]'::jsonb END"
    )
    cur.execute(f"""
        SELECT count(DISTINCT c->>'name')
        FROM jira.issues, LATERAL jsonb_array_elements({comps_expr}) c""")
    distinct_c = cur.fetchone()[0]
    cur.execute(f"""
        SELECT c->>'name', count(*)
        FROM jira.issues, LATERAL jsonb_array_elements({comps_expr}) c
        GROUP BY 1 ORDER BY 2 DESC LIMIT 15""")
    top = cur.fetchall()
    out = (
        "## 4. Components\n\n"
        f"- Issues with at least one component: **{with_c:,}** ({pct(with_c, total)} of all {total:,} issues)\n"
        f"- Distinct component names: **{distinct_c:,}**\n\n"
        "### 4.1 Top 15 components\n\n"
        + md_table(["Component", "Issues", f"% of all issues ({total:,})"],
                   [(f"`{name}`", f"{n:,}", pct(n, total)) for name, n in top])
        + "\n"
    )
    return out


# Main trace package: taken from the FIRST "at org.apache.spark..." line of a
# description. Group 1 is the first segment after "org.apache.spark."; a
# segment that starts with an uppercase letter is a class (SparkContext,
# SparkFunSuite), so the package is plain org.apache.spark.
TRACE_LINE_RE = re.compile(r"(?m)^\s*at\s+org\.apache\.spark(?:\.([A-Za-z0-9_]+))?")


def main_package(desc: str):
    """Main trace package of a description, or None without an `at` line."""
    m = TRACE_LINE_RE.search(desc)
    if not m:
        return None
    seg = m.group(1)
    if not seg or seg[0].isupper():
        return "org.apache.spark"
    return f"org.apache.spark.{seg}"


def sections_traces(cur, total: int) -> tuple:
    """Returns (section 5 markdown, section 6 markdown)."""
    cur.execute("""
        SELECT issue_key, description, raw->'components'
        FROM jira.issues
        WHERE description IS NOT NULL AND btrim(description) <> ''""")

    info = {}  # issue_key -> (is_trace, main package or None, [component names])
    for key, desc, comps_json in cur.fetchall():
        comps_raw = json.loads(comps_json) if comps_json else []
        comps = [c["name"] for c in comps_raw
                 if isinstance(c, dict) and c.get("name")]
        pkg = main_package(desc)
        is_trace = pkg is not None or ("Exception" in desc)
        info[key] = (is_trace, pkg, comps)

    n_desc = len(info)
    trace_n = sum(1 for t, _, _ in info.values() if t)
    with_pkg = sum(1 for t, p, _ in info.values() if t and p)
    exception_only = trace_n - with_pkg
    pkg_counter = Counter(p for t, p, _ in info.values() if t and p)
    trace_with_comp = sum(1 for t, _, comps in info.values() if t and comps)

    sec_e = (
        "## 5. Error traces in descriptions\n\n"
        "An issue counts as *having a trace* when its description contains a line\n"
        "starting with `at org.apache.spark` (indentation allowed) or contains the\n"
        "word `Exception`.\n\n"
        f"Denominators: all issues = **{total:,}**; issues with a non-empty "
        f"description = **{n_desc:,}**; traced issues = **{trace_n:,}**.\n\n"
        f"- Issues with a trace: **{trace_n:,}** "
        f"({pct(trace_n, n_desc)} of issues with a description; "
        f"{pct(trace_n, total)} of all issues)\n"
        f"- Traced issues with an `at org.apache.spark` line (have a main package): "
        f"**{with_pkg:,}** ({pct(with_pkg, trace_n)} of traced issues)\n"
        f"- Traced only by the word `Exception` (no `at org.apache.spark` line, "
        f"so no main package; excluded from 5.1 and 6): **{exception_only:,}** "
        f"({pct(exception_only, trace_n)} of traced issues)\n"
        f"- Traced issues that also have at least one component: "
        f"**{trace_with_comp:,}** ({pct(trace_with_comp, trace_n)} of traced issues)\n\n"
        "### 5.1 Top 15 main packages\n\n"
        "The *main package* of an issue comes from the FIRST `at org.apache.spark...`\n"
        "line of its description, cut at the first path segment: `org.apache.spark.sql.\n"
        "execution...` counts as `org.apache.spark.sql`. A segment that starts with an\n"
        "uppercase letter is a class, not a package, so `org.apache.spark.SparkContext...`\n"
        "and `org.apache.spark.SparkFunSuite...` count as plain `org.apache.spark`.\n"
        "Each issue counts exactly once, in its main package.\n\n"
        + md_table(["Main package", "Issues",
                    f"% of issues with a main package ({with_pkg:,})"],
                   [(f"`{p}`", f"{c:,}", pct(c, with_pkg))
                    for p, c in pkg_counter.most_common(15)])
        + "\n"
    )

    rows = []
    for pkg, _c in pkg_counter.most_common(15):
        comp_counter = Counter()
        subset = 0
        for t, p, comps in info.values():
            if t and p == pkg and comps:
                subset += 1
                comp_counter.update(set(comps))
        if not subset:
            rows.append((f"`{pkg}`", "0", "-", "-"))
            continue
        top_comp, top_n = comp_counter.most_common(1)[0]
        rows.append((f"`{pkg}`", f"{subset:,}", f"`{top_comp}`",
                     pct(top_n, subset)))
    sec_f = (
        "## 6. Chapter-1 signal: main trace package -> component\n\n"
        "Only issues with a main package (section 5.1) AND at least one component.\n"
        "Each issue counts once, in its main package. For each of the top 15 main\n"
        "packages: the number of such issues, their most frequent component, and the\n"
        "share of those issues (denominator: the issues in that row) that have it.\n\n"
        + md_table(["Main package", "Issues (main package + component)",
                    "Most frequent component",
                    "% of the row's issues with that component"], rows)
        + "\n"
    )
    return sec_e, sec_f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default=str(REPO / ".env"),
                    help="path to .env with PGDATABASE/PGUSER/PGPASSWORD")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5433,
                    help="compose host port for the db service")
    ap.add_argument("--out", default=str(REPO / "docs" / "data_profile.md"))
    args = ap.parse_args()

    env = load_env(Path(args.env))
    conn = psycopg2.connect(host=args.host, port=args.port,
                            dbname=env["PGDATABASE"],
                            user=env["PGUSER"], password=env["PGPASSWORD"])
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM jira.issues")
    total = cur.fetchone()[0]
    if total == 0:
        raise SystemExit("jira.issues is empty; nothing to profile")

    gen = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = (
        "# JIRA Data Profile\n\n"
        f"Generated by `scripts/profile_jira.py` on {gen}. "
        f"Source: `jira.issues` ({total:,} issues, project SPARK), "
        "read-only connection to the compose database (127.0.0.1:5433).\n\n"
    )
    doc += section_columns(cur, total) + "\n"
    doc += section_raw_keys(cur, total) + "\n"
    doc += section_status(cur) + "\n"
    doc += section_components(cur, total) + "\n"
    sec_e, sec_f = sections_traces(cur, total)
    doc += sec_e + "\n"
    doc += sec_f

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    conn.close()
    print(f"wrote {out} ({len(doc):,} chars, {total:,} issues)")


if __name__ == "__main__":
    main()
