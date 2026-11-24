#!/usr/bin/env python3
"""Refresh every medallion layer with one command.

1. bronze: incremental ingest (src/jira_rag/ingest_bronze.py --incremental);
   with --full, a full run from page 0 instead.
2. silver: rebuild from bronze (sql/030_silver.sql).
3. gold: rebuild from silver and regenerate docs/metrics.md (scripts/report_metrics.py).
4. with --profile, also regenerate docs/data_profile.md (scripts/profile_jira.py).

Each step stops the refresh on failure. The bronze step is resumable: if it is
interrupted, running this command again continues from the last complete page.
Never touches jira.issues.

Run:
    .venv/bin/python scripts/refresh.py
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psycopg2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from jira_rag.ingest_bronze import load_env  # noqa: E402


def step(name, fn):
    t0 = time.monotonic()
    print(f"== {name}", flush=True)
    fn()
    print(f"== {name}: ok in {time.monotonic() - t0:.0f}s", flush=True)


def run(*argv):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    subprocess.run([sys.executable, *argv], cwd=REPO, env=env, check=True)


def apply_sql(path):
    conn = psycopg2.connect("")
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(path.read_text(encoding="utf-8"))
            cur.execute("SELECT count(*) FROM silver.issue")
            print(f"silver.issue: {cur.fetchone()[0]:,}")
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="full bronze run (resumes an interrupted full run) instead of incremental")
    ap.add_argument("--profile", action="store_true", help="also regenerate docs/data_profile.md")
    ap.add_argument("--status-file", default="", help="passed to the bronze ingest")
    args = ap.parse_args()
    load_env(REPO / ".env")

    bronze = ["-m", "jira_rag.ingest_bronze"] + ([] if args.full else ["--incremental"])
    if args.status_file:
        bronze += ["--status-file", args.status_file]
    try:
        step("bronze", lambda: run(*bronze))
        step("silver", lambda: apply_sql(REPO / "sql" / "030_silver.sql"))
        step("gold", lambda: run(str(REPO / "scripts" / "report_metrics.py")))
        if args.profile:
            step("profile", lambda: run(str(REPO / "scripts" / "profile_jira.py")))
    except subprocess.CalledProcessError as e:
        sys.exit(f"refresh stopped: {e.cmd[1:]} exited {e.returncode}; run again to resume")


if __name__ == "__main__":
    main()
