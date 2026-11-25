#!/usr/bin/env python3
"""Parent-component context for sub-tasks, as of the sub-task's creation date.

A sub-task (silver.issue.parent_key) has a parent. The parent's component
list AT THE MOMENT the sub-task was created is what a router could have
known then; the parent's CURRENT component list may include component
changes made afterwards (information from the future).

Source of the history: bronze.jira_issue_raw, the Jira changelog
(payload->'changelog'->'histories'). Each history carries one timestamp
(created) and a list of changed items; the component changes are the items
with field = 'Component'. Observed format (verified against real data):
each change carries exactly one component, in fromString (removed) or
toString (added), never both, e.g.
    SPARK-10000  2015-08-14 20:20:16 UTC  fromString NULL   toString 'SQL'
    SPARK-10000  2015-08-14 20:27:50 UTC  fromString 'SQL'  toString NULL

components_asof(current, changes, t): start from the issue's current
component list and undo, newest first, every change with changed_at > t:
an added component (toString) is removed, a removed one (fromString) is
put back. The result is the component list the issue had at time t.

Reads bronze and silver only; writes nothing to the database.

Run standalone (prints 3 real changelog examples for format review):
    .venv/bin/python scripts/router_parent_context.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

COMPONENT_CHANGES_SQL = """
SELECT b.issue_key,
       (h->>'created')::timestamptz,
       it->>'fromString',
       it->>'toString'
FROM bronze.jira_issue_raw b,
     jsonb_array_elements(b.payload->'changelog'->'histories') h,
     jsonb_array_elements(h->'items') it
WHERE it->>'field' = 'Component'
"""


def load_component_changes(cur):
    """{issue_key: [(changed_at, fromString, toString), ...]} in changelog
    document (chronological) order; ties keep document order (stable sort)."""
    changes = {}
    cur.execute(COMPONENT_CHANGES_SQL)
    for key, ts, frm, to in cur.fetchall():
        changes.setdefault(key, []).append((ts, frm, to))
    for key in changes:
        changes[key].sort(key=lambda e: e[0])  # stable: document order kept
    return changes


def components_asof(current, changes, t):
    """Component list of an issue at time t: current components with every
    change after t undone, newest first."""
    comps = set(current)
    for ts, frm, to in reversed(changes):
        if ts <= t:
            break
        if to is not None:
            comps.discard(to)
        if frm is not None:
            comps.add(frm)
    return sorted(comps)


def parent_context(cur, subtasks):
    """{issue_key: (parent_components_now, parent_components_asof)} for the
    given {issue_key: (parent_key, created_at)} mapping. parent_components_now
    is the parent's current component list (silver.issue_component);
    parent_components_asof is the parent's list at the sub-task's created_at.
    A parent missing from bronze/silver gives ([], []) and is counted."""
    changes = load_component_changes(cur)
    cur.execute("SELECT issue_key, array_agg(component ORDER BY component)"
                " FROM silver.issue_component GROUP BY issue_key")
    comp_map = {k: list(c) for k, c in cur.fetchall()}
    cur.execute("SELECT issue_key FROM bronze.jira_issue_raw")
    in_bronze = {k for k, in cur.fetchall()}

    out = {}
    missing_parent = 0
    for key, (parent_key, created_at) in subtasks.items():
        if parent_key not in in_bronze:
            out[key] = (None, None)
            missing_parent += 1
            continue
        now = sorted(comp_map.get(parent_key, []))
        asof = components_asof(now, changes.get(parent_key, []), created_at)
        out[key] = (now, asof)
    return out, missing_parent


def main() -> None:
    env_file = REPO / ".env"
    from profile_jira import load_env

    env = load_env(env_file)
    import psycopg2

    conn = psycopg2.connect(
        host=env["PGHOST"], port=env["PGPORT"], dbname=env["PGDATABASE"],
        user=env["PGUSER"], password=env["PGPASSWORD"])
    cur = conn.cursor()
    cur.execute(COMPONENT_CHANGES_SQL + " LIMIT 3")
    print("=== 3 real Component changelog entries (format check) ===")
    for key, ts, frm, to in cur.fetchall():
        print(f"issue_key={key}  changed_at={ts.isoformat()}  "
              f"fromString={frm!r}  toString={to!r}")
    cur.execute("""
SELECT b.issue_key, (h->>'created')::timestamptz, it->>'fromString', it->>'toString'
FROM bronze.jira_issue_raw b,
     jsonb_array_elements(b.payload->'changelog'->'histories') h,
     jsonb_array_elements(h->'items') it
WHERE it->>'field' = 'Component' AND it->>'fromString' IS NOT NULL
LIMIT 1""")
    key, ts, frm, to = cur.fetchone()
    print(f"(removal example: issue_key={key}  changed_at={ts.isoformat()}"
          f"  fromString={frm!r}  toString={to!r})")
    conn.close()


if __name__ == "__main__":
    main()
