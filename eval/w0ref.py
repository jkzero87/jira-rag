"""Compute the w=0 reference: live summary_desc_filtered top-10 for all 40
questions, saved to eval/cache/w0ref.json.

Call retrieve.search_filtered(q, "summary_desc", 10) for each question —
the exact same function run_eval uses — and record the top-10 keys.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

import retrieve

GOLD_PATH = ROOT / "eval" / "gold.jsonl"
OUT_PATH = ROOT / "eval" / "cache" / "w0ref.json"


def main():
    gold = [json.loads(l) for l in GOLD_PATH.read_text().splitlines() if l.strip()]
    retrieve.init()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result = {}
    for g in gold:
        qid = g["id"]
        results, form, where, fcount = retrieve.search_filtered(
            g["question"], "summary_desc", 10)
        result[qid] = [r[0] for r in results]
        print(f"[{qid}] where={where or '(none)'} top10={result[qid][:3]}...")

    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {OUT_PATH}  ({len(result)} questions)")


if __name__ == "__main__":
    main()
