#!/usr/bin/env python3
"""Ingest full Apache JIRA issue payloads (all fields + changelog) into bronze.

Resumable: progress lives in bronze.ingest_run. Each page is upserted and the
run's pages_done is advanced in the same transaction, so the process can be
killed at any moment (including power loss) and a re-run continues from the
last complete page. Upserts by issue_key, so overlap never duplicates rows.

Usage:
    python -m jira_rag.ingest_bronze                 # resume or start
    python -m jira_rag.ingest_bronze --max-pages 2   # sample run, resumable later
"""
import argparse
import datetime as dt
import json
import os
import signal
import sys
import time

import psycopg2
import requests
from psycopg2.extras import Json, execute_values

BASE = "https://issues.apache.org/jira/rest/api/2"
USER_AGENT = "jira-rag/0.x (+https://github.com/jkzero87/jira-rag)"
MAX_RETRIES = 5
RESUMABLE = ("running", "interrupted", "failed")


def load_env(path):
    """Read KEY=VALUE lines into os.environ without overriding; tolerates CRLF."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n").strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Stop(Exception):
    """Raised from the signal handler so the current page is not half-committed."""


class Client:
    def __init__(self, sleep_base):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self.sleep_base = sleep_base
        self.retries = 0

    def get(self, path, params=None):
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = self.s.get(f"{BASE}{path}", params=params, timeout=120)
            except (requests.ConnectionError, requests.Timeout) as e:
                err, wait = f"network: {e}", None
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code != 429 and r.status_code < 500:
                    raise RuntimeError(f"HTTP {r.status_code} {path}: {r.text[:300]}")
                err = f"HTTP {r.status_code}"
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.strip().isdigit() else None
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"giving up after {MAX_RETRIES} retries on {path}: {err}")
            wait = max(wait or 0, self.sleep_base * 2 ** attempt)
            self.retries += 1
            print(f"  retry {attempt + 1}/{MAX_RETRIES} in {wait:.0f}s ({err})", flush=True)
            time.sleep(wait)


def ingest_fields(cli, cur):
    fields = cli.get("/field")
    execute_values(cur, """
        INSERT INTO bronze.jira_field_catalog (field_id, name, custom, schema_type, fetched_at)
        VALUES %s
        ON CONFLICT (field_id) DO UPDATE SET name=EXCLUDED.name, custom=EXCLUDED.custom,
            schema_type=EXCLUDED.schema_type, fetched_at=EXCLUDED.fetched_at""",
        [(f["id"], f["name"], bool(f.get("custom")),
          (f.get("schema") or {}).get("type"), dt.datetime.now(dt.timezone.utc))
         for f in fields])
    return len(fields)


def complete_changelog(cli, issue):
    """Replace a truncated search changelog with the issue endpoint's. Returns True if refetched."""
    cl = issue.get("changelog") or {}
    hist = cl.get("histories") or []
    if cl.get("total", len(hist)) <= len(hist):
        return False
    full = cli.get(f"/issue/{issue['key']}", params={"expand": "changelog", "fields": "none"})
    fcl = full.get("changelog") or {}
    if len(fcl.get("histories") or []) < fcl.get("total", 0):
        print(f"  WARN {issue['key']}: changelog still truncated "
              f"({len(fcl['histories'])}/{fcl['total']})", flush=True)
    issue["changelog"] = fcl
    return True


def write_status(path, run, note, resume_cmd):
    if not path:
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "# Bronze ingest status\n\n"
            f"- Updated: {dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}\n"
            f"- Run: {run['run_id']} status={run['status']}\n"
            f"- Pages done: {run['pages_done']}  issues done: {run['issues_done']}"
            f" / total {run.get('total', '?')}\n"
            f"- Retries: {run['retries']}  changelogs refetched: {run['changelogs_refetched']}\n"
            f"- Note: {note}\n\n"
            "Resume (continues from the last complete page):\n\n"
            f"```bash\n{resume_cmd}\n```\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jql", default="project = SPARK ORDER BY key ASC")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=1.0, help="pause between pages (s)")
    ap.add_argument("--max-pages", type=int, default=0, help="stop after N pages this invocation; 0 = all")
    ap.add_argument("--new-run", action="store_true", help="start from page 0 even if a run is resumable")
    ap.add_argument("--status-file", default="", help="markdown status file refreshed every page")
    ap.add_argument("--env", default=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    args = ap.parse_args()
    load_env(args.env)

    resume_cmd = (f"cd {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))} && "
                  f"PYTHONPATH=src .venv/bin/python -m jira_rag.ingest_bronze"
                  + (f" --status-file {args.status_file}" if args.status_file else ""))

    def on_signal(signum, _frame):
        raise Stop(signal.Signals(signum).name)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    conn = psycopg2.connect("")
    cur = conn.cursor()
    cli = Client(sleep_base=max(args.sleep, 1.0) * 2)

    n_fields = ingest_fields(cli, cur)
    conn.commit()
    print(f"field catalog: {n_fields} fields", flush=True)

    cur.execute("""SELECT run_id, pages_done, issues_done, retries, changelogs_refetched
                   FROM bronze.ingest_run WHERE jql=%s AND status = ANY(%s)
                   ORDER BY run_id DESC LIMIT 1""", (args.jql, list(RESUMABLE)))
    row = None if args.new_run else cur.fetchone()
    if row:
        run = dict(zip(("run_id", "pages_done", "issues_done", "retries", "changelogs_refetched"), row))
        cur.execute("UPDATE bronze.ingest_run SET status='running', last_error=NULL WHERE run_id=%s",
                    (run["run_id"],))
        print(f"resuming run {run['run_id']} at page {run['pages_done']}", flush=True)
    else:
        cur.execute("INSERT INTO bronze.ingest_run (jql) VALUES (%s) RETURNING run_id", (args.jql,))
        run = {"run_id": cur.fetchone()[0], "pages_done": 0, "issues_done": 0,
               "retries": 0, "changelogs_refetched": 0}
        print(f"new run {run['run_id']}", flush=True)
    conn.commit()
    run["status"] = "running"

    def finish(status, note, err=None):
        conn.rollback()
        run["status"] = status
        cur.execute("""UPDATE bronze.ingest_run SET status=%s, last_error=%s,
                       finished_at = CASE WHEN %s='done' THEN now() ELSE finished_at END
                       WHERE run_id=%s""", (status, err, status, run["run_id"]))
        conn.commit()
        write_status(args.status_file, run, note, resume_cmd)
        print(f"run {run['run_id']} -> {status}: {note}", flush=True)

    pages_this_call = 0
    try:
        while True:
            t0 = time.monotonic()
            start_at = run["pages_done"] * args.page_size
            r0 = cli.retries
            data = cli.get("/search", params={
                "jql": args.jql, "startAt": start_at, "maxResults": args.page_size,
                "fields": "*all", "expand": "changelog"})
            issues = data.get("issues", [])
            run["total"] = data.get("total")
            if not issues:
                finish("done", f"all pages ingested ({run['issues_done']} issues)")
                break
            refetched = sum(complete_changelog(cli, i) for i in issues)
            execute_values(cur, """
                INSERT INTO bronze.jira_issue_raw (issue_key, issue_id, payload, fetched_at)
                VALUES %s
                ON CONFLICT (issue_key) DO UPDATE SET issue_id=EXCLUDED.issue_id,
                    payload=EXCLUDED.payload, fetched_at=EXCLUDED.fetched_at""",
                [(i["key"], int(i["id"]), Json(i), dt.datetime.now(dt.timezone.utc)) for i in issues])
            run["pages_done"] += 1
            run["issues_done"] = start_at + len(issues)
            run["retries"] += cli.retries - r0
            run["changelogs_refetched"] += refetched
            cur.execute("""UPDATE bronze.ingest_run SET pages_done=%s, issues_done=%s, retries=%s,
                           changelogs_refetched=%s WHERE run_id=%s""",
                        (run["pages_done"], run["issues_done"], run["retries"],
                         run["changelogs_refetched"], run["run_id"]))
            conn.commit()
            pages_this_call += 1
            print(f"page {run['pages_done']} ({run['issues_done']}/{run['total']}) "
                  f"{time.monotonic() - t0:.1f}s refetched={refetched}", flush=True)
            write_status(args.status_file, run, "ingest running", resume_cmd)
            if len(issues) < args.page_size and run["issues_done"] >= (run["total"] or 0):
                finish("done", f"all pages ingested ({run['issues_done']} issues)")
                break
            if args.max_pages and pages_this_call >= args.max_pages:
                finish("interrupted", f"stopped after --max-pages {args.max_pages}")
                break
            time.sleep(args.sleep)
    except Stop as e:
        finish("interrupted", f"stopped by {e}; last complete page kept")
        sys.exit(130)
    except Exception as e:
        finish("failed", f"error: {e}", str(e)[:1000])
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
