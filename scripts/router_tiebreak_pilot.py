#!/usr/bin/env python3
"""Tie-break pilot: can the local 27B pick the right component among the
router's top-5 when the router is unsure?

Router. The configuration selected in experiment 2
(docs/router_hippocampus.md): combined = 0.75 x LR + 0.25 x kNN(k=10), the
month loop with as-of labels of scripts/router_hippocampus.py. Scores are
read from its per-month cache scratch/hippocampus/<YYYY-MM>.npz (computed
with its run_months when a month is missing); no training code is
duplicated here.

Population. VALIDATION issues (created in 2023) whose margin (top1 - top2
of the combined score) is below t90, the 90%-precision threshold of the
selected configuration. t90 is read from docs/router_hippocampus.md and
asserted equal to choose_threshold on the validation margins. 200 issues
are sampled with numpy default_rng(42) and saved to
scratch/tiebreak/sample.json (reused, and checked, on later runs).

Prompt. Issue summary and description (truncated to 1500 characters), then
the 5 router candidates in router order without scores, each with its
JIRA description (data/spark_components.json, "(no description)" when
empty) and the titles of its 3 most recent training issues created before
the first day of the issue's month that carry it in their as-of labels.
The model must answer exactly {"component": "<name>"} with one candidate;
anything else is invalid.

Modes. off: chat_template_kwargs enable_thinking false, max_tokens 64.
on: enable_thinking true, reasoning_effort "medium", max_tokens 4096. Both
temperature 0, on the same 200 issues.

Resumable: each response is cached to scratch/tiebreak/<mode>/<key>.json
and skipped if present.

Run:
    .venv/bin/python scripts/router_tiebreak_pilot.py run off
    .venv/bin/python scripts/router_tiebreak_pilot.py run on
    .venv/bin/python scripts/router_tiebreak_pilot.py report
"""
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from profile_jira import load_env  # noqa: E402
from router_confidence import choose_threshold  # noqa: E402
from router_novelty import load_data, train_pool  # noqa: E402
from router_hippocampus import (  # noqa: E402
    DOC_PATH, HIGHLIGHT, align, cache_ok, load_month, month_eval, run_months)

SRC, W = "knn10", 0.25
OUT = REPO / "scratch" / "tiebreak"
SAMPLE_PATH = OUT / "sample.json"
COMPONENTS_PATH = REPO / "data" / "spark_components.json"
N_SAMPLE = 200
SEED = 42
TOPK = 5
N_RECENT = 3
DESC_CHARS = 1500
MODES = {
    "off": {"max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}},
    "on": {"max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": True},
           "reasoning_effort": "medium"},
}


def doc_t90():
    """t90 of the selected configuration from docs/router_hippocampus.md."""
    txt = DOC_PATH.read_text(encoding="utf-8")
    assert "**kNN (k=10), w = 0.25**" in txt, "selected configuration changed"
    sec = txt.split("Target precision 90%")[1].split("Target precision 80%")[0]
    m = re.search(r"^\| selected \| (\d+\.\d+) \|", sec, re.M)
    assert m, "selected 90% row not found in docs/router_hippocampus.md"
    return float(m.group(1))


def router_split(data, months, year_pred):
    """Per issue of the chosen months: key, month, final, top-5 names, margin."""
    out = []
    for ym in months:
        if not year_pred(ym[0]):
            continue
        d = data[ym]
        names = d["names"]
        union = sorted(set(names))
        comb = (1.0 - W) * align(d["lr"], names, union) + W * align(d[SRC], names, union)
        top1, margin = month_eval(d, SRC, W)
        for i in range(comb.shape[0]):
            order = np.lexsort((np.arange(len(union)), -comb[i]))[:TOPK]
            top5 = [union[j] for j in order]
            assert top5[0] == top1[i]
            out.append({"key": d["keys"][i], "month": ym, "final": d["final"][i],
                        "top5": top5, "margin": float(margin[i])})
    return out


def load_router():
    summary, parent_key_map, changes, by_time, times, months, _end = load_data()
    t0 = time.perf_counter()
    missing = [ym for ym in months if not cache_ok(ym)]
    if missing:
        print(f"[cache] {len(missing)} months missing; computing with router_hippocampus.run_months")
        run_months(months, times, by_time, changes, parent_key_map, summary, t0)
    data = {ym: load_month(ym) for ym in months}
    val = router_split(data, months, lambda y: y == 2023)
    test = router_split(data, months, lambda y: y >= 2024)
    assert len(val) == 4733 and len(test) == 15754
    t90 = doc_t90()
    vm = np.array([r["margin"] for r in val])
    vc = [r["top5"][0] in r["final"] for r in val]
    assert sum(vc) == 3512, f"selected val top-1 {sum(vc)} != 3512 (docs/router_hippocampus.md)"
    t_re = choose_threshold(vm, vc, 0.90)
    assert t_re == t90, f"recomputed t90 {t_re} != doc t90 {t90}"
    print(f"[assert] selected val top-1 3512/4733; t90 = {t90} (doc and recomputed)")
    return summary, changes, by_time, times, val, test, t90


def get_sample(val, t90):
    pop = [r for r in val if r["margin"] < t90]
    rng = np.random.default_rng(SEED)
    idx = sorted(rng.choice(len(pop), N_SAMPLE, replace=False).tolist())
    keys = [pop[i]["key"] for i in idx]
    if SAMPLE_PATH.exists():
        saved = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
        assert saved["keys"] == keys, "saved sample differs from the recomputed one"
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        SAMPLE_PATH.write_text(json.dumps(
            {"seed": SEED, "t90": t90, "population": len(pop), "keys": keys},
            indent=1) + "\n", encoding="utf-8")
    return pop, [pop[i] for i in idx]


def load_descriptions(keys):
    env = load_env(REPO / ".env")
    import psycopg2
    conn = psycopg2.connect(host=env["PGHOST"], port=env["PGPORT"], dbname=env["PGDATABASE"],
                            user=env["PGUSER"], password=env["PGPASSWORD"])
    cur = conn.cursor()
    cur.execute("SELECT issue_key, payload->'fields'->>'description' FROM bronze.jira_issue_raw"
                " WHERE issue_key = ANY(%s)", (list(keys),))
    out = {k: (d or "") for k, d in cur.fetchall()}
    conn.close()
    assert set(out) == set(keys)
    return out


def recent_titles(sample, summary, changes, by_time, times):
    """{(month, component): [3 most recent training titles]} with as-of labels."""
    out = {}
    for ym in sorted({r["month"] for r in sample}):
        _s, _m, tr_rows, tr_labels = train_pool(ym, times, by_time, changes)
        need = {c for r in sample if r["month"] == ym for c in r["top5"]}
        for c in need:
            idx = [i for i, la in enumerate(tr_labels) if c in la]
            idx = sorted(idx, key=lambda i: (tr_rows[i][1], tr_rows[i][0]))[-N_RECENT:][::-1]
            out[(ym, c)] = [summary[tr_rows[i][0]] for i in idx]
    return out


def build_prompt(r, summary, desc, comp_desc, titles):
    lines = [
        "You route Apache Spark JIRA issues to a component.",
        "",
        f"Issue summary: {summary[r['key']]}",
        "Issue description:",
        (desc[r["key"]] or "")[:DESC_CHARS] or "(empty)",
        "",
        "Candidate components:",
    ]
    for n, c in enumerate(r["top5"], 1):
        lines.append(f"{n}. {c}")
        lines.append(f"   Description: {comp_desc.get(c) or '(no description)'}")
        lines.append("   Recent issues:")
        for t in titles[(r["month"], c)]:
            lines.append(f"   - {t}")
    lines += [
        "",
        "Choose the one candidate component that best fits the issue.",
        'Answer with only this JSON and nothing else: {"component": "<name>"}',
        "where <name> is exactly one of the candidate names above.",
    ]
    return "\n".join(lines)


def parse(content, cands):
    try:
        obj = json.loads(content.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    if isinstance(obj, dict) and list(obj) == ["component"] and obj["component"] in cands:
        return obj["component"]
    return None


def llm_call(prompt, mode):
    env = load_env(REPO / ".env")
    key = env.get("LLAMA_API_KEY", "").strip()
    url = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8092/v1/chat/completions")
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, **MODES[mode]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=1800) as resp:
        d = json.load(resp)
    wall = time.perf_counter() - t0
    msg = d["choices"][0]["message"]
    return {"wall_s": wall, "timings": d.get("timings", {}),
            "prompt_n": d["timings"]["prompt_n"], "predicted_n": d["timings"]["predicted_n"],
            "finish_reason": d["choices"][0].get("finish_reason"),
            "content": msg.get("content"), "reasoning_content": msg.get("reasoning_content")}


def prepare():
    summary, changes, by_time, times, val, test, t90 = load_router()
    pop, sample = get_sample(val, t90)
    desc = load_descriptions([r["key"] for r in sample])
    comp_desc = {c["name"]: c["description"] for c in
                 json.loads(COMPONENTS_PATH.read_text(encoding="utf-8"))["components"]}
    titles = recent_titles(sample, summary, changes, by_time, times)
    prompts = {r["key"]: build_prompt(r, summary, desc, comp_desc, titles) for r in sample}
    return val, test, t90, pop, sample, prompts


def run(mode):
    _v, _t, _t90, _p, sample, prompts = prepare()
    d = OUT / mode
    d.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    done = 0
    for n, r in enumerate(sample, 1):
        path = d / f"{r['key']}.json"
        if not path.exists():
            res = llm_call(prompts[r["key"]], mode)
            res["key"] = r["key"]
            res["parsed"] = parse(res["content"], r["top5"])
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(path)
            done += 1
        if n % 10 == 0 or n == len(sample):
            print(f"[{mode}] {n}/{len(sample)} (new {done}) "
                  f"elapsed={(time.perf_counter() - t_start) / 60:.1f}min", flush=True)


def q(a, p):
    return float(np.percentile(np.array(a, dtype=float), p))


def report():
    val, test, t90, pop, sample, _prompts = prepare()
    res = {}
    for mode in MODES:
        res[mode] = {}
        for r in sample:
            path = OUT / mode / f"{r['key']}.json"
            assert path.exists(), f"missing {path}; run mode {mode} first"
            res[mode][r["key"]] = json.loads(path.read_text(encoding="utf-8"))
    n_val_low = len(pop)
    n_test_low = sum(1 for r in test if r["margin"] < t90)
    a = []
    ap = a.append
    ap("# Router tie-break pilot (27B among router top-5)")
    ap("")
    ap(f"Router: combined 0.75 x LR + 0.25 x kNN(k=10). t90 = {t90:.2f}. Validation issues below t90: {n_val_low}/{len(val)}; sample n = {len(sample)} (default_rng({SEED})).")
    ap("")
    inside = [r for r in sample if set(r["top5"]) & set(r["final"])]
    ap("## a) Ceiling")
    ap("")
    ap(f"A true final component is among the 5 candidates in {len(inside)}/{len(sample)} issues ({100 * len(inside) / len(sample):.1f}%).")
    ap("")

    def hits(rows, pick):
        return sum(1 for r in rows if pick(r) in r["final"])

    pickers = {"router": lambda r: r["top5"][0],
               "27B off": lambda r: res["off"][r["key"]]["parsed"],
               "27B on": lambda r: res["on"][r["key"]]["parsed"]}
    ap("## b) Top-1")
    ap("")
    ap("Invalid outputs count as wrong.")
    ap("")
    ap(f"| system | all {len(sample)} | truth in top-5 ({len(inside)}) |")
    ap("|---|---|---|")
    for name, p in pickers.items():
        h1, h2 = hits(sample, p), hits(inside, p)
        ap(f"| {name} | {h1} ({100 * h1 / len(sample):.1f}%) | {h2} ({100 * h2 / len(inside):.1f}%) |")
    ap("")
    ap("## c) Invalid outputs")
    ap("")
    ap("| mode | invalid | finish_reason=length |")
    ap("|---|---|---|")
    for mode in MODES:
        inv = sum(1 for v in res[mode].values() if v["parsed"] is None)
        ln = sum(1 for v in res[mode].values() if v["finish_reason"] == "length")
        ap(f"| {mode} | {inv}/{len(sample)} | {ln} |")
    ap("")
    ap("## d) Time and tokens per issue")
    ap("")
    ap("| mode | wall median (s) | wall p90 (s) | prompt_n median | prompt_n p90 | predicted_n median | predicted_n p90 |")
    ap("|---|---|---|---|---|---|---|")
    med = {}
    for mode in MODES:
        vs = list(res[mode].values())
        w = [v["wall_s"] for v in vs]
        pn = [v["prompt_n"] for v in vs]
        dn = [v["predicted_n"] for v in vs]
        med[mode] = q(w, 50)
        ap(f"| {mode} | {q(w, 50):.2f} | {q(w, 90):.2f} | {q(pn, 50):.0f} | {q(pn, 90):.0f} | {q(dn, 50):.0f} | {q(dn, 90):.0f} |")
    ap("")
    ap(f"Projection with median wall time (sequential, --parallel 1): validation below t90 = {n_val_low} issues, test below t90 = {n_test_low} issues.")
    ap("")
    ap("| mode | validation (h) | test (h) |")
    ap("|---|---|---|")
    for mode in MODES:
        ap(f"| {mode} | {med[mode] * n_val_low / 3600:.1f} | {med[mode] * n_test_low / 3600:.1f} |")
    ap("")
    ap("## e) Rare components in the sample")
    ap("")
    ap("| component | n | router | 27B off | 27B on |")
    ap("|---|---|---|---|---|")
    for c in HIGHLIGHT:
        rows = [r for r in sample if c in r["final"]]
        if not rows:
            ap(f"| {c} | 0 | — | — | — |")
            continue
        ap(f"| {c} | {len(rows)} | " + " | ".join(str(hits(rows, p)) for p in pickers.values()) + " |")
    ap("")
    ap("## f) Examples where the 27B changed the router's answer")
    ap("")
    for mode in MODES:
        ch = [r for r in sample if res[mode][r["key"]]["parsed"] not in (None, r["top5"][0])]
        fixed = sum(1 for r in ch if res[mode][r["key"]]["parsed"] in r["final"])
        broke = sum(1 for r in ch if r["top5"][0] in r["final"])
        ap(f"Mode {mode}: {len(ch)} changed answers; 27B right in {fixed}, router was right in {broke}. First 10 in sample order:")
        ap("")
        ap("| key | truth | router | 27B | 27B right |")
        ap("|---|---|---|---|---|")
        for r in ch[:10]:
            p = res[mode][r["key"]]["parsed"]
            ap(f"| {r['key']} | {', '.join(r['final'])} | {r['top5'][0]} | {p} | {'yes' if p in r['final'] else 'no'} |")
        ap("")
    path = OUT / "pilot_report.md"
    path.write_text("\n".join(a), encoding="utf-8")
    print("\n".join(a))
    print(f"wrote {path}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "run" and sys.argv[2] in MODES:
        run(sys.argv[2])
    elif sys.argv[1:] == ["report"]:
        report()
    elif sys.argv[1:] == ["prompt"]:
        _v, _t, _t90, pop, sample, prompts = prepare()
        print(f"population {len(pop)}; sample {len(sample)}")
        print(prompts[sample[0]["key"]])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
