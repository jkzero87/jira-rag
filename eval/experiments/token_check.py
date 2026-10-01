#!/usr/bin/env python3
"""Token count for the same 710 pairs at desc[:500] vs desc[:1000].

Uses ONE function: individual tok.encode(q, d) per pair (no padding).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

LISTS = ROOT / "eval" / "cache" / "lists.json"
MODEL = "BAAI/bge-reranker-v2-m3"
TOPN = 20


def main():
    import psycopg2
    from transformers import AutoTokenizer

    lists = json.loads(LISTS.read_text())["questions"]
    tok = AutoTokenizer.from_pretrained(MODEL)

    conn = psycopg2.connect("")
    cur = conn.cursor()

    # Build doc_text for both 500 and 1000 chars
    doc_500 = {}
    doc_1000 = {}
    for r in lists:
        keys = r["vec"][:TOPN]
        cur.execute("SELECT issue_key, summary, description FROM jira.issues "
                    "WHERE issue_key = ANY(%s)", (keys,))
        for k, s, d in cur.fetchall():
            doc_500[k] = (s or "") + "\n" + (d or "")[:500]
            doc_1000[k] = (s or "") + "\n" + (d or "")[:1000]
    conn.close()

    # Count tokens per pair, no padding, same 710 pairs
    tokens_500 = []
    tokens_1000 = []
    for r in lists:
        rid = r["id"]
        keys = r["vec"][:TOPN]
        for k in keys:
            if k in doc_500:
                q = r["question"]
                t500 = len(tok.encode(q, doc_500[k], add_special_tokens=True))
                t1000 = len(tok.encode(q, doc_1000[k], add_special_tokens=True))
                tokens_500.append(t500)
                tokens_1000.append(t1000)

    n = len(tokens_500)
    avg_500 = sum(tokens_500) / n
    avg_1000 = sum(tokens_1000) / n
    print(f"pairs: {n}")
    print(f"avg tokens/pair (desc[:500]):  {avg_500:.1f}")
    print(f"avg tokens/pair (desc[:1000]): {avg_1000:.1f}")


if __name__ == "__main__":
    main()
