#!/usr/bin/env python3
"""Runner for token_check with local snapshot paths (no network)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

LISTS = ROOT / "eval" / "cache" / "lists.json"
SNAP = ROOT / ("eval/cache/hf_hub/models--BAAI--bge-reranker-v2-m3/"
               "snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
TOPN = 20


def main():
    import psycopg2
    from transformers import AutoTokenizer

    lists = json.loads(LISTS.read_text())["questions"]
    tok = AutoTokenizer.from_pretrained(str(SNAP))

    conn = psycopg2.connect("host=127.0.0.1 port=5432")
    cur = conn.cursor()

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

    n500 = n1000 = 0
    cnt500 = cnt1000 = 0
    max500 = max1000 = 0
    tokens_500 = []
    tokens_1000 = []
    for r in lists:
        q = r["question"]
        for k in r["vec"][:TOPN]:
            if k in doc_500:
                t500 = len(tok.encode(q, doc_500[k], add_special_tokens=True))
                t1000 = len(tok.encode(q, doc_1000[k], add_special_tokens=True))
                tokens_500.append(t500)
                tokens_1000.append(t1000)
    n = len(tokens_500)
    print(f"pairs: {n}")
    print(f"avg tokens/pair (desc[:500]):  {sum(tokens_500)/n:.1f}")
    print(f"avg tokens/pair (desc[:1000]): {sum(tokens_1000)/n:.1f}")


if __name__ == "__main__":
    main()
