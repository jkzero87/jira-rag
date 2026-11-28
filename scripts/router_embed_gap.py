#!/usr/bin/env python3
"""Fill the router embedding gap: router issues absent from jira.issues / jira.issue_chunks.

1. Record the before state: jira.issues count, jira.issue_chunks count per strategy,
   and an md5 checksum of (issue_key, summary, description) over the existing rows.
2. Insert the missing router issues into jira.issues with ingest.py --bronze-keys-file,
   then assert the existing rows' count and checksum are unchanged.
3. Run scripts/embed_consistency_check.py (stops on min cosine < 0.999).
4. Run embed.py --keys-file for both strategies.
5. Assert every load_rows() row (scripts/router_text_baseline.py) has an embedding for
   both strategies, and write docs/router_embedding_gap.json (read by
   scripts/router_hippocampus.py).

Resumable: if docs/router_embedding_gap.json exists, only the final assertion runs;
steps 2 and 4 skip rows that are already present.

Run:
    .venv/bin/python scripts/router_embed_gap.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from router_text_baseline import load_env, load_rows  # noqa: E402

OUT = REPO / "docs" / "router_embedding_gap.json"
KEYS = REPO / "scratch" / "router_gap_keys.txt"
STRATEGIES = ("summary_only", "summary_desc")
# NULL is encoded by an is-null flag (Postgres text cannot hold chr(0)).
CHECKSUM = ("SELECT count(*), md5(string_agg(issue_key || chr(31) || (summary IS NULL)::text"
            " || coalesce(summary, '') || chr(31) || (description IS NULL)::text"
            " || coalesce(description, ''), chr(30) ORDER BY issue_key))"
            " FROM jira.issues WHERE NOT (issue_key = ANY(%s))")


def connect(env):
    import psycopg2
    return psycopg2.connect(host=env["PGHOST"], port=env["PGPORT"], dbname=env["PGDATABASE"],
                            user=env["PGUSER"], password=env["PGPASSWORD"])


def counts(cur):
    cur.execute("SELECT count(*) FROM jira.issues")
    out = {"jira.issues": cur.fetchone()[0]}
    for s in STRATEGIES:
        cur.execute("SELECT count(*) FROM jira.issue_chunks WHERE strategy = %s", (s,))
        out[f"jira.issue_chunks[{s}]"] = cur.fetchone()[0]
    return out


def run(env, *argv, capture=False):
    """Run a repo Python script; returns (seconds, stdout or None)."""
    child = dict(os.environ, PYTHONPATH=str(REPO / "src"), HF_HOME=str(REPO / ".hf"),
                 **{k: v for k, v in env.items() if k.startswith("PG")})
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, *argv], cwd=REPO, env=child, check=True,
                       stdout=subprocess.PIPE if capture else None, text=True)
    return time.monotonic() - t0, p.stdout


def assert_coverage(cur):
    rows = load_rows()[0]
    keys = [r[0] for r in rows]
    for s in STRATEGIES:
        cur.execute("SELECT count(DISTINCT issue_key) FROM jira.issue_chunks"
                    " WHERE strategy = %s AND issue_key = ANY(%s)", (s, keys))
        n = cur.fetchone()[0]
        assert n == len(keys), (s, n, len(keys))
    print(f"coverage ok: all {len(keys)} load_rows() rows embedded for both strategies")
    return len(keys)


def main():
    env = load_env(REPO / ".env")
    conn = connect(env)
    conn.autocommit = True
    cur = conn.cursor()
    if OUT.exists():
        assert_coverage(cur)
        print(f"{OUT.relative_to(REPO)} exists: gap already filled")
        return

    cur.execute("SELECT t.issue_key, t.created_at FROM router.text_dataset t"
                " WHERE t.split IN ('train', 'test') AND NOT EXISTS (SELECT 1 FROM"
                " jira.issue_chunks c WHERE c.issue_key = t.issue_key"
                " AND c.strategy = 'summary_desc') ORDER BY t.issue_key")
    gap = cur.fetchall()
    gap_keys = [k for k, _ in gap]
    if gap_keys:
        KEYS.parent.mkdir(exist_ok=True)
        KEYS.write_text("".join(k + "\n" for k in gap_keys))
    elif KEYS.exists():  # resumed after embedding finished but before OUT was written
        gap_keys = KEYS.read_text().split()
    before = counts(cur)
    cur.execute(CHECKSUM, (gap_keys,))
    existing_before = cur.fetchone()
    print("before:", before, "gap:", len(gap_keys), flush=True)

    t_ingest, _ = run(env, "src/jira_rag/ingest.py", "--bronze-keys-file", str(KEYS))
    cur.execute(CHECKSUM, (gap_keys,))
    existing_after = cur.fetchone()
    assert existing_after == existing_before, (existing_before, existing_after)
    print(f"existing rows unchanged: n={existing_after[0]} md5={existing_after[1]}")

    t_check, out = run(env, "scripts/embed_consistency_check.py", capture=True)
    check = json.loads(out.strip().splitlines()[-1])
    print("consistency check:", check, flush=True)
    t_embed = {s: run(env, "src/jira_rag/embed.py", "--strategy", s, "--keys-file", str(KEYS))[0]
               for s in STRATEGIES}
    after = counts(cur)
    n_rows = assert_coverage(cur)
    created = sorted(c for _, c in gap)
    report = {
        "gap_issues": len(gap_keys),
        "gap_created_min": created[0].isoformat() if created else None,
        "gap_created_max": created[-1].isoformat() if created else None,
        "before": before, "after": after,
        "existing_rows": existing_before[0], "existing_md5": existing_before[1],
        "load_rows": n_rows,
        "consistency_check": check,
        "seconds": {"ingest": round(t_ingest, 1), "consistency_check": round(t_check, 1),
                    **{f"embed[{s}]": round(t, 1) for s, t in t_embed.items()}},
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
