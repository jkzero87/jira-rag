#!/usr/bin/env python3
"""Experiment 3 (journal 4.7): does the router age, and does recency help?

Question. The router was selected once, before 2024. As the data ages, does a
statically trained model keep up — and do recency-weighted sampling or
monthly retraining help?

Four variants, all V-token+balanced (tf-idf + one-vs-rest logistic
regression with class_weight="balanced", the pipeline of
scripts/router_label_hygiene.py) with C = 8 FIXED (no C grid in this
experiment; C = 8 is the value experiment 7 selected for V-token+balanced,
see docs/router_label_hygiene.md):

  static          trained once on created_at < CV_CUT (final labels) and
                  applied to every month 2023 .. last data month: the
                  aging baseline.
  recency         same training set and labels, sample_weight =
                  0.5 ** (age_years / h) with age = CV_CUT - created_at;
                  h in {1, 2, 4} chosen on validation top-1 (tie ->
                  smaller h).
  rolling         for each month M: trained on the issues created before
                  the first day of M, predicting the issues created in M.
  rolling+recency same as rolling, plus the sample weights of `recency`,
                  age measured against the first day of M, h fixed at the
                  h chosen by `recency`.

Label rule (step 1). Step 1 found an existing as-of builder —
components_asof in scripts/router_parent_context.py:58, the builder the
earlier experiments used for "labels as-of". So the training labels of
every month M are the components as known at the first day of M (the final
component list with every Component changelog entry on/after that day
undone); issues whose as-of list is empty are left out of that month's
training set. The ground truth for the issues created in month M is their
FINAL component list — the same label definition every other experiment
uses (a top-1 hit = predicted component is one of the issue's final
components). The fallback rule (train only on issues RESOLVED before the
first day of M, with their final labels) was not needed.

Confidence. Per variant: margin = top1 - top2 probability; the 90% and 80%
margin thresholds are chosen ONLY on that variant's validation predictions
(router_confidence.choose_threshold, grid 0.05..0.95, lowest t reaching the
target); the already-chosen thresholds are then applied to the test
predictions.

Test is never used to choose anything (no h, no threshold, no variant).
Test numbers are reference only.

Resumable: every (variant, month) prediction set is saved to
scratch/recency/<variant>/<YYYY-MM>.npz and skipped if present; delete the
directory to force a rerun.

Writes docs/router_recency.md (English), byte-identical on every run.

Run:
    .venv/bin/python scripts/router_recency.py
"""

import bisect
import datetime
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from profile_jira import load_env  # noqa: E402
from router_parent_context import components_asof, load_component_changes  # noqa: E402
from router_text_baseline import (  # noqa: E402
    CUTOFF, CV_CUT, RANDOM_STATE, extend_text, hit_counts, load_rows, pct)
from router_label_hygiene import build_vtoken_balanced_cv_model  # noqa: E402
from router_tossing import RESULTS_TOP15  # noqa: E402
from router_confidence import TARGETS, auto_stats, choose_threshold  # noqa: E402

# C = 8 fixed: the C selected for V-token+balanced in experiment 7
# (docs/router_label_hygiene.md). No C grid is run in this experiment.
C_FIXED = 8
HALFLIFE_HOURS = (1, 2, 4)
VARIANTS = ("static", "recency", "rolling", "rolling_recency")
SCRATCH = REPO / "scratch" / "recency"
DOC_PATH = REPO / "docs" / "router_recency.md"
YEAR_SECONDS = 365.25 * 86400.0


def month_start(y, m):
    return datetime.datetime(y, m, 1, tzinfo=datetime.timezone.utc)


def fit_balanced(train_rows, labels, weights, pkey_map, psum_map):
    """The V-token+balanced pipeline of
    router_label_hygiene.fit_variant_balanced, generalized in exactly two
    ways: (1) explicit per-row labels, so a rolling month can train on
    labels as-of the first day of the month instead of the final
    components; (2) an optional sample_weight (the recency weights).
    Vectorizer, solver, hyperparameters and random state are identical to
    fit_variant_balanced. Returns (vec, clf, mlb, fit_seconds)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    t0 = time.perf_counter()
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3,
                          ngram_range=(1, 2), max_features=200000)
    Xtr = vec.fit_transform(
        [extend_text(r, pkey_map, psum_map, False) for r in train_rows])
    mlb = MultiLabelBinarizer()
    ytr = mlb.fit_transform(labels)
    if weights is None:
        clf = OneVsRestClassifier(
            LogisticRegression(C=C_FIXED, max_iter=1000,
                               random_state=RANDOM_STATE,
                               class_weight="balanced"),
            n_jobs=-1)
        clf.fit(Xtr, ytr)
    else:
        # This sklearn version forwards fit kwargs to the per-class
        # estimators only via metadata routing.
        import sklearn
        sklearn.set_config(enable_metadata_routing=True)
        clf = OneVsRestClassifier(
            LogisticRegression(C=C_FIXED, max_iter=1000,
                               random_state=RANDOM_STATE,
                               class_weight="balanced")
            .set_fit_request(sample_weight=True),
            n_jobs=-1)
        clf.fit(Xtr, ytr, sample_weight=weights)
    return vec, clf, mlb, time.perf_counter() - t0


def make_predict(vec, clf, pkey_map, psum_map):
    def predict_proba(rows):
        X = vec.transform(
            [extend_text(r, pkey_map, psum_map, False) for r in rows])
        return clf.predict_proba(X)
    return predict_proba


def save_month(variant, ym, rows, proba, names):
    """Cache one (variant, month) prediction set. `rows` are the month's
    issues (prediction rows) in dataset order."""
    path = SCRATCH / variant / f"{ym[0]:04d}-{ym[1]:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    top1 = [names[int(np.argmax(proba[i]))] for i in range(n)]
    # second-highest probability: np.partition(a, -2) places the two
    # largest at the end (index -2 is the second-largest; the rest of
    # the array is unordered, so index [1] is NOT the second-largest)
    margin = np.array([
        float(proba[i].max() - np.partition(proba[i], -2)[-2]) for i in range(n)])
    np.savez_compressed(
        path,
        keys=np.array([r[0] for r in rows]),
        created=np.array([r[1].timestamp() for r in rows], dtype=float),
        actual=np.array([list(r[4]) for r in rows], dtype=object),
        top1=np.array(top1),
        margin=margin)
    return path


def load_month(variant, ym):
    path = SCRATCH / variant / f"{ym[0]:04d}-{ym[1]:02d}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        return {
            "actual": [list(a) for a in z["actual"]],
            "top1": z["top1"].tolist(),
            "margin": z["margin"],
        }


def month_window(ym, times, by_time):
    start = month_start(ym[0], ym[1])
    y2, m2 = (ym[0] + 1, 1) if ym[1] == 12 else (ym[0], ym[1] + 1)
    end = month_start(y2, m2)
    i0 = bisect.bisect_left(times, start)
    i1 = bisect.bisect_left(times, end)
    return start, by_time[i0:i1]


def recency_weights(rows, ref, h):
    return np.array([
        0.5 ** ((ref - r[1]).total_seconds() / YEAR_SECONDS / h)
        for r in rows], dtype=float)


def run_static(months, times, by_time, rows, parent_key_map, summary, t_start):
    missing = [ym for ym in months
               if not (SCRATCH / "static" / f"{ym[0]:04d}-{ym[1]:02d}.npz").exists()]
    names = None
    if missing:
        t0 = time.perf_counter()
        names, predict = build_vtoken_balanced_cv_model(
            rows, C_FIXED, parent_key_map, summary)
        print(f"[static] model fit in {time.perf_counter() - t0:.1f}s "
              f"(train = created_at < {CV_CUT:%Y-%m-%d}, final labels)")
    for ym in missing:
        t0 = time.perf_counter()
        _start, mrows = month_window(ym, times, by_time)
        save_month("static", ym, mrows, predict(mrows), names)
        print(f"[static] {ym[0]:04d}-{ym[1]:02d} n={len(mrows)} "
              f"month={time.perf_counter() - t0:.1f}s "
              f"elapsed={(time.perf_counter() - t_start) / 60:.0f}min")


def run_recency(months, times, by_time, rows, parent_key_map, summary, t_start):
    missing = [ym for ym in months
               if not (SCRATCH / "recency" / f"{ym[0]:04d}-{ym[1]:02d}.npz").exists()]
    i_cut = bisect.bisect_left(times, CV_CUT)
    i_test = bisect.bisect_left(times, CUTOFF)
    cv_train = by_time[:i_cut]
    cv_val = by_time[i_cut:i_test]
    val_actual = [list(r[4]) for r in cv_val]

    # ---- half-life h, chosen on VALIDATION top-1 (tie -> smaller h) ----
    h_rows = []
    models = {}
    for h in HALFLIFE_HOURS:
        t0 = time.perf_counter()
        w = recency_weights(cv_train, CV_CUT, h)
        vec, clf, mlb, fit_s = fit_balanced(
            cv_train, [list(r[4]) for r in cv_train], w,
            parent_key_map, summary)
        names = list(mlb.classes_)
        predict = make_predict(vec, clf, parent_key_map, summary)
        hit1, _hit3 = hit_counts(predict(cv_val), names, val_actual, 3)
        h_rows.append({"h": h, "n": len(cv_val), "hit1": hit1, "fit_s": fit_s})
        models[h] = (names, predict)
        print(f"[recency] h={h} fit={fit_s:.1f}s val_top1={hit1}/{len(cv_val)}")
    best = max(h_rows, key=lambda d: (d["hit1"], -d["h"]))
    best_h = best["h"]
    names, predict = models[best_h]
    print(f"[recency] chosen h = {best_h} (validation top-1, tie -> smaller h)")

    if missing:
        for ym in missing:
            t0 = time.perf_counter()
            _start, mrows = month_window(ym, times, by_time)
            save_month("recency", ym, mrows, predict(mrows), names)
            print(f"[recency] {ym[0]:04d}-{ym[1]:02d} n={len(mrows)} "
                  f"month={time.perf_counter() - t0:.1f}s "
                  f"elapsed={(time.perf_counter() - t_start) / 60:.0f}min")
    return h_rows, best_h


def run_rolling(months, times, by_time, changes, parent_key_map, summary,
                best_h, with_recency, t_start):
    variant = "rolling_recency" if with_recency else "rolling"
    for ym in months:
        path = SCRATCH / variant / f"{ym[0]:04d}-{ym[1]:02d}.npz"
        if path.exists():
            continue
        t0 = time.perf_counter()
        start, mrows = month_window(ym, times, by_time)
        pool = by_time[:bisect.bisect_left(times, start)]  # created < start
        # Training labels = components as known at the first day of the
        # month (step 1's as-of builder); empty as-of list -> not trained.
        tr_rows, tr_labels = [], []
        for r in pool:
            la = components_asof(sorted(r[4]), changes.get(r[0], []), start)
            if la:
                tr_rows.append(r)
                tr_labels.append(la)
        w = (recency_weights(tr_rows, start, best_h) if with_recency else None)
        vec, clf, mlb, fit_s = fit_balanced(
            tr_rows, tr_labels, w, parent_key_map, summary)
        predict = make_predict(vec, clf, parent_key_map, summary)
        save_month(variant, ym, mrows, predict(mrows), list(mlb.classes_))
        print(f"[{variant}] {ym[0]:04d}-{ym[1]:02d} pool={len(pool)} "
              f"train={len(tr_rows)} fit={fit_s:.1f}s "
              f"month={time.perf_counter() - t0:.1f}s "
              f"elapsed={(time.perf_counter() - t_start) / 60:.0f}min")


def gather(variant, months):
    """Load all cached months of one variant; validation = the 2023 months,
    test = everything from 2024 on."""
    val, test = {}, {}
    for ym in months:
        d = load_month(variant, ym)
        assert d is not None, f"missing {SCRATCH / variant / f'{ym[0]:04d}-{ym[1]:02d}.npz'}"
        (val if ym[0] == 2023 else test)[ym] = d
    return val, test


def combine(split):
    """Concatenate the per-month rows of one split in month order."""
    order = sorted(split)
    actual, top1 = [], []
    margin = np.concatenate([split[ym]["margin"] for ym in order])
    for ym in order:
        d = split[ym]
        actual.extend(d["actual"])
        top1.extend(d["top1"])
    n = len(actual)
    correct1 = [t in a for t, a in zip(top1, actual)]
    return {"n": n, "actual": actual, "correct1": correct1,
            "margin": margin, "hits": sum(correct1)}


def auto_row(A, t):
    """(n, auto-routed, coverage, precision, wrong-per-100) at threshold t."""
    n, n_auto, wrong, wp100 = auto_stats(A["margin"], A["correct1"], t)
    cov = n_auto / n if n else 0.0
    prec = (n_auto - wrong) / n_auto if n_auto else 0.0
    return n, n_auto, cov, prec, wp100


def variant_metrics(val, test):
    """Top-1 rates plus the 90%/80% thresholds chosen on VALIDATION only
    and applied to both splits."""
    out = {
        "val": val, "test": test,
        # threshold chosen on validation only (None if unreachable)
        "thr90": choose_threshold(val["margin"], val["correct1"], 0.90),
        "thr80": choose_threshold(val["margin"], val["correct1"], 0.80),
    }
    for target, tkey in ((0.90, "thr90"), (0.80, "thr80")):
        t = out[tkey]
        n = int(target * 100)
        if t is None:
            out[f"v{n}"] = out[f"t{n}"] = None
            continue
        out[f"v{n}"] = auto_row(val, t)
        out[f"t{n}"] = auto_row(test, t)
    return out


def quarter_stats(split):
    """{quarter label: (n, hits)} for the test months (2024+)."""
    out = {}
    for ym in sorted(split):
        key = f"{ym[0]}Q{(ym[1] - 1) // 3 + 1}"
        d = split[ym]
        n = len(d["actual"])
        hits = sum(t in a for t, a in zip(d["top1"], d["actual"]))
        cur = out.get(key)
        out[key] = (cur[0] + n, cur[1] + hits) if cur else (n, hits)
    return out


def build_doc(months, data_end, metrics, test_months, h_rows, best_h,
              best_rolling, selected):
    a = []
    ap = a.append
    ap("# Router aging and recency (experiment 3)")
    ap("")
    ap("Generated by `scripts/router_recency.py` on every run; never edit by")
    ap("hand.")
    ap("")
    ap("**Question.** The router was selected once, before 2024. As the data")
    ap("ages, does a statically trained model keep up — and do recency-")
    ap("weighted sampling or monthly retraining help?")
    ap("")
    ap("**Base model.** Every variant is V-token+balanced: tf-idf")
    ap("(sublinear_tf, min_df=3, 1-2 grams, 200k features) plus one-vs-rest")
    ap("logistic regression with `class_weight=\"balanced\"` (the pipeline of")
    ap("`scripts/router_label_hygiene.py`), with **C = 8 fixed**. No C grid is")
    ap("re-run in this experiment: C = 8 is the value experiment 7 selected")
    ap("for V-token+balanced (docs/router_label_hygiene.md). The only fitting")
    ap("code added here generalizes that pipeline to explicit per-row labels")
    ap("(rolling months) and to `sample_weight` (recency).")
    ap("")
    ap("**Splits and months.** train = created_at < 2023-01-01 (40018")
    ap("issues); validation = the 2023 slice of train (4733 issues); test =")
    ap(f"created_at >= 2024-01-01 (15754 issues, data ends {data_end}).")
    ap(f"Months covered: 2023-01 .. {months[-1][0]:04d}-{months[-1][1]:02d} "
       f"({len(months)} months).")
    ap("")
    ap("**Variants.**")
    ap("")
    ap("- `static`: trained once on created_at < 2023-01-01 (final labels)")
    ap("  and applied to every month; the aging baseline. Its validation")
    ap("  top-1 is asserted by the script to be 3298/4733 (69.7%), the")
    ap("  documented V-token+balanced number.")
    ap("- `recency`: same training set, `sample_weight = 0.5 ** (age_years /")
    ap("  h)` with age = 2023-01-01 - created_at; h in {1, 2, 4} chosen on")
    ap("  validation top-1 (tie -> smaller h).")
    ap("- `rolling`: for each month M, trained on the issues created before")
    ap("  the first day of M, predicting the issues created in M.")
    ap("- `rolling+recency`: rolling plus the recency weights, age measured")
    ap("  against the first day of M, h fixed at the h chosen by `recency`.")
    ap("")
    ap("**Label rule (step 1).** Step 1 found an existing as-of builder:")
    ap("`components_asof` in `scripts/router_parent_context.py:58` (the")
    ap("builder the earlier experiments used for \"labels as-of\"). So the")
    ap("training labels of every month M are the components as known at the")
    ap("first day of M — the final component list with every Component")
    ap("changelog entry on/after that day undone — and issues whose as-of")
    ap("list is empty are left out of that month's training set. The ground")
    ap("truth for the issues created in month M is their FINAL component")
    ap("list, the same label definition every other experiment uses (a top-1")
    ap("hit = predicted component is one of the issue's final components).")
    ap("The fallback rule (train only on issues RESOLVED before the first")
    ap("day of M, with their final labels) was not needed.")
    ap("")
    ap("### a) Top-1 and auto-routing at the confidence thresholds")
    ap("")
    ap("| variant | val top-1 | test top-1 |")
    ap("|---|---|---|")
    for v in VARIANTS:
        m = metrics[v]
        ap(f"| {v} | {pct(m['val']['hits'], m['val']['n'])} "
           f"({m['val']['hits']}/{m['val']['n']}) | {pct(m['test']['hits'], m['test']['n'])} "
           f"({m['test']['hits']}/{m['test']['n']}) |")
    ap("")
    ap("Margin thresholds are chosen on each variant's **validation**")
    ap("predictions only (lowest grid t reaching the target precision); the")
    ap("chosen thresholds are then applied to the test predictions.")
    ap("")
    for target in TARGETS:
        tgt100 = int(target * 100)
        ap(f"**Target precision {tgt100}% on validation:**")
        ap("")
        ap("| variant | margin ≥ | val auto-routed | val coverage | val precision | val wrong/100 | test auto-routed | test coverage | test precision | test wrong/100 |")
        ap("|---|---|---|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            m = metrics[v]
            t = m[f"thr{tgt100}"]  # threshold chosen on validation only
            row = m[f"v{tgt100}"]
            trow = m[f"t{tgt100}"]
            if row is None:
                ap(f"| {v} | — | — | — | — | — | — | — | — | — |")
                continue
            n_val, n_auto, cov, prec, wp100 = row
            n_test, n_auto2, cov2, prec2, wp1002 = trow
            ap(f"| {v} | {t:.2f} | {n_auto}/{n_val} | {100.0 * cov:.1f}% | "
               f"{100.0 * prec:.1f}% | {('%.1f' % wp100) if wp100 is not None else '—'} "
               f"| {n_auto2}/{n_test} | {100.0 * cov2:.1f}% | {100.0 * prec2:.1f}% | "
               f"{('%.1f' % wp1002) if wp1002 is not None else '—'} |")
        ap("")
    ap(f"### b) Test top-1 by quarter: static vs {best_rolling}")
    ap("")
    qs = quarter_stats(test_months["static"])
    qb = quarter_stats(test_months[best_rolling])
    ap("| quarter | n | static top-1 | "
       f"{best_rolling} top-1 |")
    ap("|---|---|---|---|")
    for k in sorted(qs):
        n, hits = qs[k]
        nb, hitsb = qb[k]
        assert n == nb
        ap(f"| {k} | {n} | {pct(hits, n)} | {pct(hitsb, n)} |")
    ap("")
    ap(f"### c) Test top-1 by component: static vs {selected}")
    ap("")
    ap(f"Variant chosen by validation top-1 (tie order: static, recency,")
    ap(f"rolling, rolling+recency): **{selected}** "
       f"({pct(metrics[selected]['val']['hits'], metrics[selected]['val']['n'])} "
       f"validation top-1).")
    ap("")
    test_actual = metrics["static"]["test"]["actual"]
    ap("| component | n (test) | static top-1 | "
       f"{selected} top-1 |")
    ap("|---|---|---|---|")
    cs = Counter()
    for comp_list in test_actual:
        cs.update(comp_list)
    data_top15 = sorted(cs.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    assert [c for c, _n in data_top15] == RESULTS_TOP15, (
        f"top-15 by test frequency {[c for c, _n in data_top15]} does not "
        f"match docs/router_results.md {RESULTS_TOP15}")
    stat_c1 = metrics["static"]["test"]["correct1"]
    sel_c1 = metrics[selected]["test"]["correct1"]
    for c in RESULTS_TOP15:
        idx = [i for i, comp_list in enumerate(test_actual) if c in comp_list]
        n = len(idx)
        ap(f"| {c} | {n} | {pct(sum(stat_c1[i] for i in idx), n)} | "
           f"{pct(sum(sel_c1[i] for i in idx), n)} |")
    ap("")
    ap(f"### d) Recency half-life h (chosen on validation top-1)")
    ap("")
    ap("| h (years) | val top-1 |")
    ap("|---|---|")
    for d in h_rows:
        ap(f"| {d['h']} | {pct(d['hit1'], d['n'])} ({d['hit1']}/{d['n']}) |")
    ap("")
    ap(f"Chosen h = **{best_h}** (highest validation top-1; tie -> smaller h).")
    ap("")
    ap("### Note on the test split")
    ap("")
    ap("The test split was never used to choose anything in this experiment:")
    ap("the half-life h, the margin thresholds and the variant selections")
    ap("above are all made on validation only; test is reported as")
    ap("reference.")
    ap("")
    ap("### Resumability")
    ap("")
    ap("Per-month prediction sets are cached at")
    ap("`scratch/recency/<variant>/<YYYY-MM>.npz` and skipped when present;")
    ap("delete the directory to force a rerun.")
    ap("")
    return "\n".join(a)


def main():
    t_start = time.perf_counter()
    rows, _desc, summary, parent_key_map, _cur_comps = load_rows()
    env = load_env(REPO / ".env")
    import psycopg2
    conn = psycopg2.connect(
        host=env["PGHOST"], port=env["PGPORT"], dbname=env["PGDATABASE"],
        user=env["PGUSER"], password=env["PGPASSWORD"])
    cur = conn.cursor()
    changes = load_component_changes(cur)
    conn.close()

    by_time = sorted(rows, key=lambda r: r[1])
    times = [r[1] for r in by_time]
    assert bisect.bisect_left(times, CV_CUT) == 40018, \
        "train slice (created_at < 2023) is not 40018 issues"
    months = sorted({(r[1].year, r[1].month) for r in rows if r[1] >= CV_CUT})
    assert months[0] == (2023, 1)
    n_val = 0
    n_test = 0
    for ym in months:
        _start, mrows = month_window(ym, times, by_time)
        if ym[0] == 2023:
            n_val += len(mrows)
        else:
            n_test += len(mrows)
    assert n_val == 4733 and n_test == 15754, \
        f"validation/test month counts {n_val}/{n_test} are not 4733/15754"
    data_end = by_time[-1][1].strftime("%Y-%m-%d")
    print(f"months: {months[0][0]:04d}-{months[0][1]:02d} .. "
          f"{months[-1][0]:04d}-{months[-1][1]:02d} ({len(months)}); "
          f"val={n_val} test={n_test}; data ends {data_end}")

    run_static(months, times, by_time, rows, parent_key_map, summary, t_start)
    h_rows, best_h = run_recency(months, times, by_time, rows, parent_key_map,
                                summary, t_start)
    run_rolling(months, times, by_time, changes, parent_key_map, summary,
                best_h, with_recency=False, t_start=t_start)
    run_rolling(months, times, by_time, changes, parent_key_map, summary,
                best_h, with_recency=True, t_start=t_start)

    # ---- aggregation from the cached month files ----
    metrics = {}
    test_months = {}
    for v in VARIANTS:
        val, test = gather(v, months)
        metrics[v] = variant_metrics(combine(val), combine(test))
        test_months[v] = test  # {ym: month data} for the quarter table
    assert metrics["static"]["val"]["hits"] == 3298 \
        and metrics["static"]["val"]["n"] == 4733, \
        (f"static validation top-1 is "
         f"{metrics['static']['val']['hits']}/{metrics['static']['val']['n']}, "
         "expected 3298/4733 (69.7%, docs/router_label_hygiene.md)")

    best_rolling = max(
        (("rolling", metrics["rolling"]["val"]["hits"]),
         ("rolling_recency", metrics["rolling_recency"]["val"]["hits"])),
        key=lambda p: (p[1], -VARIANTS.index(p[0])))[0]
    selected = max(
        ((v, metrics[v]["val"]["hits"]) for v in VARIANTS),
        key=lambda p: (p[1], -VARIANTS.index(p[0])))[0]

    DOC_PATH.write_text(build_doc(months, data_end, metrics, test_months,
                                  h_rows, best_h, best_rolling,
                                  selected), encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    print(f"total elapsed: {(time.perf_counter() - t_start) / 60:.1f}min")


if __name__ == "__main__":
    main()
