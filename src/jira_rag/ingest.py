#!/usr/bin/env python3
"""Ingest Apache JIRA issues into jira.issues. Idempotent: re-runs upsert."""
import argparse, sys, time
import requests, psycopg2
from psycopg2.extras import execute_batch

API = "https://issues.apache.org/jira/rest/api/2/search"
FIELDS = "summary,description,issuetype,priority,status,resolution,created,updated,resolutiondate"
# Empty DSN: psycopg2/libpq read PGHOST, PGPORT, PGDATABASE, PGUSER and
# PGPASSWORD from the environment (see .env.example).
DSN = ""

UPSERT = """
INSERT INTO jira.issues (issue_key, issue_id, project, summary, description,
    issue_type, priority, status, resolution, created, updated, resolutiondate, raw)
VALUES (%(key)s, %(id)s, %(project)s, %(summary)s, %(description)s,
    %(issue_type)s, %(priority)s, %(status)s, %(resolution)s,
    %(created)s, %(updated)s, %(resolutiondate)s, %(raw)s)
ON CONFLICT (issue_key) DO UPDATE SET
    summary=EXCLUDED.summary, description=EXCLUDED.description,
    status=EXCLUDED.status, resolution=EXCLUDED.resolution,
    priority=EXCLUDED.priority, updated=EXCLUDED.updated,
    resolutiondate=EXCLUDED.resolutiondate, raw=EXCLUDED.raw,
    ingested_at=now();
"""

def name_of(v):
    return v.get("name") if isinstance(v, dict) else None

def flatten(issue, project):
    import json
    f = issue["fields"]
    return {
        "key": issue["key"], "id": int(issue["id"]), "project": project,
        "summary": f.get("summary"), "description": f.get("description"),
        "issue_type": name_of(f.get("issuetype")), "priority": name_of(f.get("priority")),
        "status": name_of(f.get("status")), "resolution": name_of(f.get("resolution")),
        "created": f.get("created"), "updated": f.get("updated"),
        "resolutiondate": f.get("resolutiondate"), "raw": json.dumps(f),
    }

def upsert_from_bronze(conn, keys_file, project):
    """Upsert the issues listed in keys_file that are absent from jira.issues.

    Each bronze.jira_issue_raw payload has the API search-result layout (key, id,
    fields); its fields are cut to FIELDS so raw matches an API-ingested row, then
    it goes through flatten() and UPSERT. Keys already in jira.issues are skipped,
    so existing rows are never updated. Returns the number of rows inserted.
    """
    keys = [k.strip() for k in open(keys_file) if k.strip()]
    cur = conn.cursor()
    cur.execute("""SELECT b.payload FROM bronze.jira_issue_raw b
        WHERE b.issue_key = ANY(%s)
          AND NOT EXISTS (SELECT 1 FROM jira.issues i WHERE i.issue_key = b.issue_key)
        ORDER BY b.issue_id""", (keys,))
    wanted = FIELDS.split(",")
    issues = [{"key": p["key"], "id": p["id"],
               "fields": {f: p["fields"].get(f) for f in wanted}} for (p,) in cur.fetchall()]
    execute_batch(cur, UPSERT, [flatten(i, project) for i in issues], page_size=100)
    conn.commit()
    return len(issues)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="SPARK")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--max-issues", type=int, default=0, help="0 = all")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--bronze-keys-file", default=None,
        help="optional file with one issue_key per line: insert those issues from "
             "bronze.jira_issue_raw instead of the API, skipping keys already in "
             "jira.issues (default: page through the API)")
    args = ap.parse_args()

    conn = psycopg2.connect(DSN); conn.autocommit = False
    cur = conn.cursor()
    if args.bronze_keys_file:
        n = upsert_from_bronze(conn, args.bronze_keys_file, args.project)
        cur.execute("SELECT count(*) FROM jira.issues WHERE project=%s", (args.project,))
        print(f"inserted from bronze: {n}; rows in db for {args.project}: {cur.fetchone()[0]}")
        conn.close()
        return
    jql = f"project={args.project} ORDER BY created ASC"
    start, total, done = 0, None, 0

    while True:
        r = requests.get(API, params={"jql": jql, "startAt": start,
            "maxResults": args.page_size, "fields": FIELDS}, timeout=60)
        if r.status_code != 200:
            print(f"HTTP {r.status_code} at startAt={start}: {r.text[:200]}", file=sys.stderr)
            conn.commit(); sys.exit(1)
        data = r.json()
        total = data["total"] if total is None else total
        issues = data.get("issues", [])
        if not issues:
            break
        execute_batch(cur, UPSERT, [flatten(i, args.project) for i in issues], page_size=100)
        conn.commit()
        done += len(issues); start += len(issues)
        print(f"{done}/{total} upserted (startAt={start})", flush=True)
        if args.max_issues and done >= args.max_issues:
            break
        time.sleep(args.sleep)

    cur.execute("SELECT count(*) FROM jira.issues WHERE project=%s", (args.project,))
    print(f"rows in db for {args.project}: {cur.fetchone()[0]}")
    conn.close()

if __name__ == "__main__":
    main()
