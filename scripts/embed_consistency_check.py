#!/usr/bin/env python3
"""Re-embed a fixed sample of already-embedded issues and compare with the stored vectors.

Sample: the 20 smallest issue keys (by numeric suffix) among jira.issues created in
2026-09 that have an embedding for both strategies. Each sample issue is re-embedded
with the embed.py code path (build_content, prepare_batches, load_model,
make_embedder) and compared by cosine with its stored jira.issue_chunks vector.
Read-only: nothing is written to the database.

Prints one JSON line with min and mean cosine per strategy; exits 1 if any
cosine is below 0.999.

Run:
    HF_HOME=.hf .venv/bin/python scripts/embed_consistency_check.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from router_text_baseline import load_env  # noqa: E402

N_SAMPLE = 20
MIN_COSINE = 0.999


def main():
    os.environ.update({k: v for k, v in load_env(REPO / ".env").items() if k.startswith("PG")})
    import psycopg2
    from jira_rag import embed as E

    conn = psycopg2.connect(E.DSN)
    cur = conn.cursor()
    cur.execute(
        "SELECT i.issue_key, i.summary, i.description FROM jira.issues i"
        " WHERE i.created >= '2026-09-01' AND i.created < '2026-10-01'"
        "   AND (SELECT count(DISTINCT c.strategy) FROM jira.issue_chunks c"
        "        WHERE c.issue_key = i.issue_key) = 2"
        " ORDER BY split_part(i.issue_key, '-', 2)::int LIMIT %s", (N_SAMPLE,))
    worklist = cur.fetchall()
    assert len(worklist) == N_SAMPLE, len(worklist)
    keys = [r[0] for r in worklist]

    tokenizer, model = E.load_model()
    embed = E.make_embedder(tokenizer, model, 1024)
    report = {"keys": keys}
    ok = True
    for strategy in E.STRATEGIES:
        batches, _, _ = E.prepare_batches(worklist, strategy, tokenizer, 16384)
        new = {}
        for batch in batches:
            for it, e in zip(batch, embed([it[3]["input_ids"] for it in batch])):
                new[it[0]] = np.asarray(e, dtype=np.float64)
        cur.execute(
            "SELECT issue_key, embedding::text, content FROM jira.issue_chunks"
            " WHERE strategy = %s AND issue_key = ANY(%s)", (strategy, keys))
        cos, content_eq = [], 0
        for key, emb, content in cur.fetchall():
            old = np.asarray(json.loads(emb), dtype=np.float64)
            cos.append(float(new[key] @ old / np.linalg.norm(new[key]) / np.linalg.norm(old)))
            row = next(r for r in worklist if r[0] == key)
            content_eq += content == E.build_content(strategy, row[1], row[2])[:E.MAX_CHARS]
        assert len(cos) == N_SAMPLE
        report[strategy] = {"min_cosine": min(cos), "mean_cosine": float(np.mean(cos)),
                            "stored_content_equal": content_eq}
        ok &= min(cos) >= MIN_COSINE
    conn.close()
    print(json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
