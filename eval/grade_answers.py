#!/usr/bin/env python3
"""Phase 3 of the generation eval: grade generated answers (no LLM calls).

Reads a generate_answers.py output file and eval/gold.jsonl. For every
question it checks whether the answer cited at least one gold key (hit)
and whether any gold key was actually given in its contexts
(gold_in_context = the ceiling). Every cited key is classified:
  in_context: it is one of the context_keys;
  from_text:  it is not a context key but appears in the summary or the
              description[:desc_chars] of some context issue (text
              fetched from jira.issues; desc_chars from the answers
              header's settings);
  invented:   neither.
no_citation: cited_keys is empty.

Usage:
  ~/jira-rag/.venv/bin/python eval/grade_answers.py --answers <file> --out <file>

Output JSON:
  {"answers", "gold", "settings", "created_at",
   "summary": {...global and per type...},
   "from_text_citations": [{"id", "key", "found_in_issue", "field", "excerpt"}],
   "questions": [{"id", "type", "expected", "hit", "gold_in_context",
                  "cited": {key: class}, "cite_ratio", "no_citation",
                  "seconds", "completion_tokens"}]}
An existing output file is NEVER overwritten.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

import psycopg2  # noqa: E402

ENV_FILE = ROOT / ".env"


def load_dsn():
    """DSN kwargs from ~/jira-rag/.env (PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD)."""
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return {"host": env["PGHOST"], "port": env["PGPORT"], "dbname": env["PGDATABASE"],
            "user": env["PGUSER"], "password": env["PGPASSWORD"]}


def fetch_texts(keys):
    """One query: issue_key -> (summary, description[:desc_chars])."""
    if not keys:
        return {}
    conn = psycopg2.connect(**load_dsn())
    try:
        cur = conn.cursor()
        cur.execute("SELECT issue_key, summary, description FROM jira.issues "
                    "WHERE issue_key = ANY(%s)", (keys,))
        rows = cur.fetchall()
    finally:
        conn.close()
    out = {}
    for k, s, d in rows:
        out[k] = ((s or ""), (d or "")[:desc_chars])
    return out


def find_key_in_texts(key, text_map):
    """Return (issue_key, field) where `key` appears in some context issue's
    summary or description[:desc_chars], else (None, None)."""
    for issue_key, (summary, desc) in text_map.items():
        if key in summary:
            return issue_key, "summary"
        if key in desc:
            return issue_key, "description"
    return None, None


desc_chars = None


def main():
    global desc_chars
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="generate_answers.py output file")
    ap.add_argument("--out", required=True, help="output JSON file (never overwritten)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"error: {out} already exists; refusing to overwrite")

    answers_file = Path(args.answers)
    data = json.loads(answers_file.read_text())
    desc_chars = data["settings"]["desc_chars"]
    gold = {g["id"]: g for g in
            (json.loads(line) for line in
             (ROOT / "eval" / "gold.jsonl").read_text().splitlines() if line.strip())}

    questions = []
    for q in data["questions"]:
        if q["id"] not in gold:
            sys.exit(f"error: no gold for {q['id']}")
        g = gold[q["id"]]
        expected = g["expected"]
        cited = q["cited_keys"]
        ctx = set(q["context_keys"])
        text_map = fetch_texts(list(ctx))
        cited_map = {}
        for key in cited:
            if key in ctx:
                cited_map[key] = "in_context"
            elif find_key_in_texts(key, text_map)[0] is not None:
                cited_map[key] = "from_text"
            else:
                cited_map[key] = "invented"
        hit = any(k in cited for k in expected)
        gold_in_context = any(k in ctx for k in expected)
        ctx_n = len(ctx)
        cite_ratio = (round(len(set(cited) & ctx) / ctx_n, 3) if ctx_n else 0.0)
        questions.append({
            "id": q["id"],
            "type": g["type"],
            "expected": expected,
            "hit": hit,
            "gold_in_context": gold_in_context,
            "cited": cited_map,
            "cite_ratio": cite_ratio,
            "no_citation": not cited,
            "seconds": q["seconds"],
            "completion_tokens": q["completion_tokens"],
        })

    def summarize(sub):
        n = len(sub)
        hits = [q for q in sub if q["hit"]]
        ceiling = [q for q in sub if q["gold_in_context"]]
        hit_on_ceiling = [q for q in ceiling if q["hit"]]
        invented = [(q["id"], [k for k, c in q["cited"].items() if c == "invented"])
                    for q in sub if any(c == "invented" for c in q["cited"].values())]
        from_text = [(q["id"], [k for k, c in q["cited"].items() if c == "from_text"])
                     for q in sub if any(c == "from_text" for c in q["cited"].values())]
        return {
            "n": n,
            "hit_rate": round(len(hits) / n, 3) if n else None,
            "hit": len(hits),
            "ceiling_gold_in_context": len(ceiling),
            "hit_on_ceiling": len(hit_on_ceiling),
            "hit_on_ceiling_rate": round(len(hit_on_ceiling) / len(ceiling), 3) if ceiling else None,
            "invented_total": sum(len(v) for _, v in invented),
            "invented": invented,
            "from_text_total": sum(len(v) for _, v in from_text),
            "from_text": from_text,
            "no_citation": [q["id"] for q in sub if q["no_citation"]],
            "avg_cite_ratio": round(sum(q["cite_ratio"] for q in sub) / n, 3) if n else None,
            "cited_all_contexts": [q["id"] for q in sub
                                   if q["cite_ratio"] == 1.0],
            "avg_seconds": round(sum(q["seconds"] for q in sub) / n, 2) if n else None,
            "avg_completion_tokens": round(sum(q["completion_tokens"] for q in sub) / n, 1) if n else None,
        }

    summary = {"overall": summarize(questions)}
    for t in sorted({q["type"] for q in questions}):
        summary[t] = summarize([q for q in questions if q["type"] == t])

    missed = [{"id": q["id"], "expected": q["expected"],
               "cited": list(q["cited"].keys())}
              for q in questions if q["gold_in_context"] and not q["hit"]]

    # Every citation whose FINAL classification is from_text (not a context
    # key but appears in the text), and where it appears. Keys that are
    # context keys never enter this list, even if also written in another issue.
    from_text_citations = []
    for aq, qq in zip(data["questions"], questions):
        ctx = set(aq["context_keys"])
        tmap = fetch_texts(list(ctx))
        for key, cls in qq["cited"].items():
            if cls != "from_text":
                continue
            issue_key, field = find_key_in_texts(key, tmap)
            s, d = tmap[issue_key]
            text = s if field == "summary" else d
            pos = text.find(key)
            from_text_citations.append({
                "id": aq["id"],
                "key": key,
                "found_in_issue": issue_key,
                "field": field,
                "excerpt": text[max(0, pos - 100):pos + 150],
            })

    assert len(from_text_citations) == summary["overall"]["from_text_total"], \
        f"from_text_citations len {len(from_text_citations)} != from_text_total {summary['overall']['from_text_total']}"

    payload = {
        "answers": str(answers_file),
        "gold": str(ROOT / "eval" / "gold.jsonl"),
        "settings": data["settings"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "from_text_citations": from_text_citations,
        "questions": questions,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    # -----------------------------------------------------------------
    # Printed report
    # -----------------------------------------------------------------
    def row(name, s):
        return (f"{name:<10} n={s['n']:<3} hit={s['hit']}/{s['n']} "
                f"({s['hit_rate']})  ceiling={s['ceiling_gold_in_context']} "
                f"hit_on_ceiling={s['hit_on_ceiling']}/{s['ceiling_gold_in_context']} "
                f"({s['hit_on_ceiling_rate']})  invented={s['invented_total']} "
                f"from_text={s['from_text_total']}  no_citation={len(s['no_citation'])} "
                f"avg_cite_ratio={s['avg_cite_ratio']}  "
                f"cited_all={len(s['cited_all_contexts'])}/{s['n']} "
                f"avg_s={s['avg_seconds']}  avg_ct={s['avg_completion_tokens']}")

    print("=" * 110)
    print(row("overall", summary["overall"]))
    for t in sorted(k for k in summary if k != "overall"):
        print(row(t, summary[t]))
    print("=" * 110)
    print("\ngold_in_context=True, hit=False (ceiling missed):")
    if missed:
        for m in missed:
            print(f"  {m['id']}: gold={m['expected']} cited={m['cited']}")
    else:
        print("  ninguna")
    print("\nfrom_text citations (donde aparecen):")
    if from_text_citations:
        for ft in from_text_citations:
            print(f"  {ft['id']}: {ft['key']} -> {ft['found_in_issue']} ({ft['field']})")
            print(f"      ...{ft['excerpt']}...")
    else:
        print("  ninguna")
    print(f"\ngrade written: {out}")


if __name__ == "__main__":
    main()
