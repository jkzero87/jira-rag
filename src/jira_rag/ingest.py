#!/usr/bin/env python3
"""Ingest Apache JIRA issues into jira.issues. Idempotent: re-runs upsert."""
import argparse, sys, time
import requests, psycopg2
from psycopg2.extras import execute_batch

API = "https://issues.apache.org/jira/rest/api/2/search"
FIELDS = "summary,description,issuetype,priority,status,resolution,created,updated,resolutiondate"
DSN = "host=localhost port=5432 dbname=dedb user=deuser password=depassword"

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="SPARK")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--max-issues", type=int, default=0, help="0 = all")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    conn = psycopg2.connect(DSN); conn.autocommit = False
    cur = conn.cursor()
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
