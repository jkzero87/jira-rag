#!/usr/bin/env python3
"""Experiment 5: the reassignment ("tossing") graph built from Component
changes made by people, and whether the router makes the same mistakes
people made. Writes docs/router_tossing.md (English), regenerated
byte-identical on every run; never edit the doc by hand. Also writes
scratch/tossing_summary.json (machine-readable key numbers, used by the
journal-update script).

Population: router.text_dataset (scripts/router_text_dataset.py) — in-project
issues with at least one component. For each issue:
  initial components = the components it had at creation, reconstructed
    exactly like router.text_dataset.parent_components_asof: start from the
    issue's CURRENT component list (the label the router uses, = final) and
    undo, newest first, every Component changelog entry (bronze.jira_issue_raw,
    payload->'changelog'->'histories', field='Component') with a timestamp
    after created_at (scripts/router_parent_context.components_asof).
  final components = the issue's current components (row `components`).

Periods on silver's corrected created_at (UTC):
  train < 2023-01-01, validation [2023-01-01, 2024-01-01), test >= 2024-01-01.

Changed = initial != final, split into:
  late assignment = initial empty -> final non-empty (a component was set
    after creation; NOT a toss, the router's label is the same list people
    would have assigned from the start);
  real toss = initial non-empty and initial != final.
Real toss pairs (graph, part b): for a train real-toss issue, every
removed component x (x in initial, not in final) paired with every added
component y (y in final, not in initial): edge x -> y.

Model for parts (c) and (e): REUSE scripts/router_text_baseline.py's training
exactly (vtoken_cv_selection + build_vtoken_cv_model + build_vtoken_model);
nothing is re-trained a second way. The 2023 validation slice is scored by
the CV model (trained on created_at < 2023-01-01 only); the test slice by
the final model (trained on the whole train). Test is reference only.

An error row "predicts the initial component" when its top-1 prediction is a
component the person had set at creation and later removed (top-1 in
initial \\ final; for a single-component initial that is exactly the
initial component).

Reads bronze, silver and router.text_dataset only; writes no database.

Run:
    .venv/bin/python scripts/router_tossing.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from profile_jira import load_env  # noqa: E402
from router_parent_context import components_asof, load_component_changes  # noqa: E402
from router_text_baseline import (  # noqa: E402
    CV_CUT, CUTOFF, build_vtoken_cv_model, build_vtoken_model, load_rows,
    vtoken_cv_selection)

DOC_PATH = REPO / "docs" / "router_tossing.md"
SUMMARY_PATH = REPO / "scratch" / "tossing_summary.json"

# train < 2023-01-01 (the graph is built ONLY from this period, part b)
TRAIN_CUT = CV_CUT

# The 15 components listed in the "Top 15 components by test frequency"
# table of docs/router_results.md (same list router_confidence.py asserts).
RESULTS_TOP15 = [
    "SQL", "Connect", "PySpark", "Spark Core", "Kubernetes", "Tests",
    "Build", "Structured Streaming", "Project Infra", "Documentation",
    "ML", "Pandas API on Spark", "Declarative Pipelines", "Web UI",
    "Optimizer",
]

SPARK = "Spark Core"
MARGIN_TEST = 0.95  # experiment 1 (docs/router_confidence.md): margin t90


def pct(hit, n):
    return f"{100.0 * hit / n:.1f}%" if n else "n/a"


def period_of(created_at):
    if created_at < TRAIN_CUT:
        return "train"
    if created_at < CUTOFF:
        return "validation"
    return "test"


def main():
    rows, _desc, summary, parent_key_map, _cur_comps = load_rows()
    env = load_env(REPO / ".env")
    import psycopg2

    conn = psycopg2.connect(
        host=env["PGHOST"], port=env["PGPORT"], dbname=env["PGDATABASE"],
        user=env["PGUSER"], password=env["PGPASSWORD"])
    cur = conn.cursor()
    changes = load_component_changes(cur)
    conn.close()

    # ---- initial / final per issue (same _asof logic) --------------------
    init_map = {}
    n_no_change = 0
    for r in rows:
        # sort final to match components_asof's sorted output (the dataset
        # order is the DB collation order of array_agg, which is not the
        # Python sort order); membership checks on r[4] are unaffected
        key, created_at, final = r[0], r[1], sorted(r[4])
        ch = changes.get(key, [])
        init = components_asof(final, ch, created_at)
        init_map[key] = (init, final)
        if not ch:
            n_no_change += 1
            assert init == final, \
                f"{key}: no Component change recorded but " \
                f"initial {init} != final {final}"
    # asserted: issues with no Component change at all have initial == final

    def stats(period):
        n = late = real = unchanged = 0
        for r in rows:
            if period_of(r[1]) != period:
                continue
            n += 1
            init, final = init_map[r[0]]
            if not init:
                late += 1          # late assignment (final non-empty: dataset)
            elif init != final:
                real += 1
            else:
                unchanged += 1
        assert n == late + real + unchanged
        return n, late, real

    tr_n, tr_late, tr_real = stats("train")
    va_n, va_late, va_real = stats("validation")
    te_n, te_late, te_real = stats("test")

    # ---- (b) people graph, ONLY train < 2023-01-01 ----------------------
    train_rows = [r for r in rows if r[1] < TRAIN_CUT]
    assert all(r[1] < TRAIN_CUT for r in train_rows), \
        "graph rows must all have created_at < 2023-01-01"
    pairs = Counter()
    started_in = Counter()
    for r in train_rows:
        init, final = init_map[r[0]]
        if not init:
            continue
        for x in init:
            started_in[x] += 1
        if init == final:
            continue
        for x in init:
            if x in final:
                continue
            for y in final:
                if y not in init:
                    pairs[(x, y)] += 1
    top20 = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0], kv[1]))[:20]
    top20_set = {p for p, _c in top20}
    spark_in_top20 = sum(1 for p, _c in top20 if SPARK in p)
    spark_out = sorted(((p, c) for p, c in pairs.items() if p[0] == SPARK),
                       key=lambda kv: (-kv[1], kv[0]))
    spark_in = sorted(((p, c) for p, c in pairs.items() if p[1] == SPARK),
                      key=lambda kv: (-kv[1], kv[0]))

    # ---- 3 real toss examples (train, 1 -> 1, distinct pairs, hand-checkable)
    examples, seen_pairs = [], set()
    for r in train_rows:  # rows ordered by issue_key
        init, final = init_map[r[0]]
        if init != final and len(init) == 1 and len(final) == 1:
            p = (init[0], final[0])
            if p not in seen_pairs:
                examples.append(r)
                seen_pairs.add(p)
            if len(examples) == 3:
                break
    if len(examples) < 3:  # fall back to any real toss
        for r in train_rows:
            init, final = init_map[r[0]]
            if r not in examples and init and init != final:
                examples.append(r)
            if len(examples) == 3:
                break
    assert len(examples) == 3, f"need 3 toss examples, got {len(examples)}"

    # ---- model: reuse router_text_baseline's training -------------------
    tr_all = [r for r in rows if r[9] == "train"]  # split 'train' < 2024
    val = [r for r in tr_all if CV_CUT <= r[1] < CUTOFF]
    test = [r for r in rows if r[9] == "test"]

    _cv_res, best_C = vtoken_cv_selection(tr_all, parent_key_map, summary)
    cv_names, cv_predict = build_vtoken_cv_model(tr_all, best_C,
                                                 parent_key_map, summary)
    names, predict_proba = build_vtoken_model(tr_all, best_C,
                                              parent_key_map, summary)

    def signals(rows_slice, names_, predict_):
        proba = predict_(rows_slice)
        order = np.argsort(-proba, axis=1, kind="stable")
        top1 = np.array([proba[i, order[i, 0]] for i in range(len(rows_slice))])
        top2 = np.array([proba[i, order[i, 1]] for i in range(len(rows_slice))])
        return {
            "top1_comp": [names_[order[i, 0]] for i in range(len(rows_slice))],
            "margin": top1 - top2,
            "correct1": [names_[order[i, 0]] in set(rows_slice[i][4])
                         for i in range(len(rows_slice))],
        }

    val_sig = signals(val, cv_names, cv_predict)
    test_sig = signals(test, names, predict_proba)

    # sanity: the CV model must reproduce the documented validation top-1
    val_acc = pct(sum(val_sig["correct1"]), len(val_sig["correct1"]))
    assert val_acc == "68.7%", \
        f"CV model validation top-1 is {val_acc}, expected 68.7% " \
        f"(docs/router_results.md)"

    # ---- (c) errors vs initial component, validation ---------------------
    K = len(cv_names)
    name_set = set(cv_names)
    n_err = 0
    n_init_hit = 0
    base_num = 0  # sum over error rows of |initial\final intersect names|
    comp_n = {c: sum(1 for r in val if c in r[4]) for c in RESULTS_TOP15}
    per_comp = {c: [0, 0] for c in RESULTS_TOP15}  # [errors, initial hits]
    err_idx = [i for i in range(len(val)) if not val_sig["correct1"][i]]
    for i in err_idx:
        r = val[i]
        init, final = init_map[r[0]]
        target = [c for c in init if c not in final and c in name_set]
        n_err += 1
        base_num += len(target)
        hit = val_sig["top1_comp"][i] in target
        if hit:
            n_init_hit += 1
        for c in final:
            if c in per_comp:
                per_comp[c][0] += 1
                if hit:
                    per_comp[c][1] += 1
    base_rate = base_num / (K * n_err) if n_err else 0.0
    init_hit_pct = 100.0 * n_init_hit / n_err if n_err else 0.0

    # ---- (d) model confusions vs people confusions -----------------------
    conf = Counter()
    for i in err_idx:
        r = val[i]
        for c in r[4]:  # each true component of the issue
            conf[(c, val_sig["top1_comp"][i])] += 1
    top15_conf = sorted(conf.items(), key=lambda kv: (-kv[1], kv[0], kv[1]))[:15]
    n_conf_in_people = sum(1 for p, _c in top15_conf if p in top20_set)

    # ---- (e) Spark Core: test, experiment-1 margin threshold -------------
    sc_idx = [i for i in range(len(test)) if SPARK in test[i][4]]
    sc_conf_wrong = [i for i in sc_idx
                     if test_sig["margin"][i] >= MARGIN_TEST
                     and not test_sig["correct1"][i]]
    sc_pairs = Counter(test_sig["top1_comp"][i] for i in sc_conf_wrong)
    full_y = {y for (x, y) in pairs if x == SPARK}
    top20_y = {y for (x, y), _c in top20 if x == SPARK}
    n_sc_match_full = sum(c for y, c in sc_pairs.items() if y in full_y)
    n_sc_match_top20 = sum(c for y, c in sc_pairs.items() if y in top20_y)

    # ============================== markdown ==============================
    md = []
    md.append("# Router tossing: the reassignment graph (experiment 5)")
    md.append("")
    md.append("Generated by `scripts/router_tossing.py`; regenerated "
              "byte-identical on every run (do not edit by hand).")
    md.append("")
    md.append(f"Population: the `router.text_dataset` issues (in-project, at "
              f"least one component; n = {len(rows)}). **Initial** components "
              "are the components the issue had at creation, reconstructed "
              "exactly like `parent_components_asof`: the current list with "
              "every Component changelog entry after `created_at` undone, "
              "newest first (source: `bronze.jira_issue_raw`, "
              "`payload->'changelog'->'histories'`, `field='Component'`). "
              "**Final** components are the current ones — the label the "
              "router uses. Changed = initial != final, split into "
              "**late assignment** (initial empty -> has a component; not a "
              "toss) and **real toss** (initial non-empty and initial != "
              "final). Pairs: for a real-toss issue, every removed component "
              "x is paired with every added component y (edge x -> y). "
              "Periods: train < 2023-01-01, validation 2023, test >= 2024-01-01.")
    md.append("")

    # (a)
    md.append("## (a) Tossing rate per period")
    md.append("")
    md.append("| period | n | changed (initial != final) | late assignment (initial empty) | real toss (initial X -> final Y) |")
    md.append("|---|---:|---:|---:|---:|")
    for name, n, late, real in (("train", tr_n, tr_late, tr_real),
                                ("validation", va_n, va_late, va_real),
                                ("test", te_n, te_late, te_real)):
        md.append(f"| {name} | {n} | {late + real} ({pct(late + real, n)}) "
                  f"| {late} ({pct(late, n)}) | {real} ({pct(real, n)}) |")
    md.append("")

    # (b)
    md.append(f"## (b) Reassignment graph (built ONLY from train, created_at < 2023-01-01)")
    md.append("")
    md.append("Top 20 real toss pairs X -> Y; % of the issues that started in X "
              f"(started-in-X over {tr_n} train issues).")
    md.append("")
    md.append("| # | X -> Y | count | % of issues started in X |")
    md.append("|---:|---|---:|---:|")
    for rank, ((x, y), c) in enumerate(top20, 1):
        md.append(f"| {rank} | {x} -> {y} | {c} | {pct(c, started_in.get(x, 0))} |")
    md.append("")
    md.append("Some top pairs are component **renames** recorded as Component "
              "changes (e.g. R -> SparkR: same-timestamp entries from 'R' to "
              "None and from None to 'SparkR', as in the examples below); they "
              "count as tosses because the label changed.")
    md.append("")
    md.append(f"{spark_in_top20} of the top-20 pairs involve {SPARK} "
              "(X or Y is " + SPARK + ").")
    md.append("")

    # (c)
    md.append(f"## (c) Does the router err like people? (2023 validation, V-token CV model, C = {best_C})")
    md.append("")
    md.append(f"The CV model (trained on created_at < 2023-01-01 only) makes "
              f"{n_err} top-1 errors on the {len(val)} validation issues. An "
              "error row counts as predicting the *initial* component when its "
              "top-1 is a component the person had set at creation and later "
              "removed (top-1 in initial \\ final).")
    md.append("")
    md.append(f"- initial-component hits among errors: **{n_init_hit} of {n_err} "
              f"({init_hit_pct:.1f}%)**")
    md.append(f"- base rate (errors random among the model's {K} components): "
              f"**{100.0 * base_rate:.1f}%** (sum over error rows of "
              f"|initial \\ final intersect model components| = {base_num})")
    md.append("")
    md.append("| component | validation issues | errors | errors predicting the initial component | % of errors |")
    md.append("|---|---:|---:|---:|---:|")
    for c in RESULTS_TOP15:
        e = per_comp[c]
        md.append(f"| {c} | {comp_n[c]} | {e[0]} | {e[1]} | {pct(e[1], e[0])} |")
    md.append("")

    # (d)
    md.append("## (d) Model confusions vs people confusions (2023 validation)")
    md.append("")
    md.append("Top 15 (true -> predicted) error pairs of the model; each true "
              "component of an error row forms a pair with the prediction. "
              "\"people top 20\" = the pair also appears in the people graph "
              "top 20 from (b).")
    md.append("")
    md.append("| # | true -> predicted | count | in people top 20 |")
    md.append("|---:|---|---:|---|")
    for rank, ((c, p), cnt) in enumerate(top15_conf, 1):
        mark = "yes" if (c, p) in top20_set else ""
        md.append(f"| {rank} | {c} -> {p} | {cnt} | {mark} |")
    md.append("")
    md.append(f"{n_conf_in_people} of the top-15 model confusions are also in "
              "the people top-20 toss pairs.")
    md.append("")

    # (e)
    md.append(f"## (e) {SPARK} (graph: train; confident-wrong errors: test, margin >= {MARGIN_TEST})")
    md.append("")
    md.append("Where the people's tosses of " + SPARK + " go and come from (full graph, train):")
    md.append("")
    md.append("| from -> to | count |")
    md.append("|---|---:|")
    for (x, y), c in spark_out[:10]:
        md.append(f"| {x} -> {y} | {c} |")
    for (x, y), c in spark_in[:10]:
        md.append(f"| {x} -> {y} | {c} |")
    md.append("")
    md.append(f"Experiment 1 (margin >= {MARGIN_TEST} on test): {SPARK} has "
              f"{len(sc_conf_wrong)} confidently wrong auto-routes. Pairs "
              f"{SPARK} -> predicted:")
    md.append("")
    md.append("| predicted | count | in people graph | in people top 20 |")
    md.append("|---|---:|---|---|")
    for y, c in sorted(sc_pairs.items(), key=lambda kv: (-kv[1], kv[0])):
        md.append(f"| {y} | {c} | {'yes' if y in full_y else ''} "
                  f"| {'yes' if y in top20_y else ''} |")
    md.append("")
    md.append(f"- match full people graph: {n_sc_match_full} of "
              f"{len(sc_conf_wrong)} ({pct(n_sc_match_full, len(sc_conf_wrong))})")
    md.append(f"- match people top 20: {n_sc_match_top20} of "
              f"{len(sc_conf_wrong)} ({pct(n_sc_match_top20, len(sc_conf_wrong))})")
    md.append("")

    # verification
    md.append("## Verification")
    md.append("")
    md.append("- Asserted: the graph in (b) uses only issues with created_at < 2023-01-01.")
    md.append(f"- Asserted: for the {n_no_change} issues with no Component "
              "change at all, initial == final.")
    md.append(f"- Asserted: the CV model reproduces the documented "
              f"validation top-1 (68.7% on {len(val)} issues).")
    md.append("- 3 real toss examples (train, with their Component changelog "
              "dates, for hand-checking):")
    md.append("")
    for r in examples:
        init, final = init_map[r[0]]
        md.append(f"- {r[0]} (created {r[1].isoformat()}): "
                  f"{' -> '.join(init) if init else '(empty)'} -> "
                  f"{' -> '.join(final) if final else '(empty)'}")
        for ts, frm, to in changes.get(r[0], []):
            md.append(f"  - {ts.isoformat()}: from {frm!r} to {to!r}")
    md.append("")

    DOC_PATH.write_text("\n".join(md), encoding="utf-8")

    # ============================ summary json ============================
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_json = {
        "n_issues": len(rows),
        "periods": {
            "train": [tr_n, tr_late, tr_real],
            "validation": [va_n, va_late, va_real],
            "test": [te_n, te_late, te_real],
        },
        "people_top20": [[p[0], p[1], c] for p, c in top20],
        "spark_in_top20": spark_in_top20,
        "val_n": len(val),
        "val_errors": n_err,
        "val_init_hits": n_init_hit,
        "val_init_hit_pct": round(init_hit_pct, 1),
        "base_rate_pct": round(100.0 * base_rate, 1),
        "k_components": K,
        "model_confusions_top15": [[p[0], p[1], c, p in top20_set]
                                   for p, c in top15_conf],
        "n_conf_in_people": n_conf_in_people,
        "spark_conf_wrong_test": len(sc_conf_wrong),
        "spark_match_full": [n_sc_match_full, len(sc_conf_wrong)],
        "spark_match_top20": [n_sc_match_top20, len(sc_conf_wrong)],
        "spark_out": spark_out[:5],
        "spark_in": spark_in[:5],
        "cv_C": best_C,
    }
    SUMMARY_PATH.write_text(json.dumps(summary_json, indent=2, sort_keys=True),
                            encoding="utf-8")

    print(f"wrote {DOC_PATH}")
    print(f"periods: train {tr_real}/{tr_n} real toss, val {va_real}/{va_n}, "
          f"test {te_real}/{te_n}")
    print(f"top5 people pairs: {[(f'{x}->{y}', c) for (x, y), c in top20[:5]]}")
    print(f"val: {n_init_hit}/{n_err} errors predict initial "
          f"({init_hit_pct:.1f}%) vs base {100.0 * base_rate:.1f}%")
    print(f"spark: {n_sc_match_full}/{len(sc_conf_wrong)} confidently wrong "
          f"match full graph; {n_sc_match_top20} match top 20")


if __name__ == "__main__":
    main()
