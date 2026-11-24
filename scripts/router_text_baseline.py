#!/usr/bin/env python3
"""Router text classifier baselines, learned only from train, measured only on
test. Random state fixed (42); writes docs/router_results.md, regenerated
identically on every run.

Population: router.text_dataset (built by scripts/router_text_dataset.py):
in-project issues with at least one component, cut on silver's corrected
created_at ('train' < 2024-01-01 UTC, 'test' >=).

Levels (the error signals are NOT inputs; they are only used to slice the
report):
  level 0  always predict the most frequent train components
           (top-1 = the first, top-3 = the first three).
  level 1  TfidfVectorizer(sublinear_tf=True, min_df=3,
                          ngram_range=(1,2), max_features=200000)
           + one-vs-rest logistic regression on the multi-component
           labels (MultiLabelBinarizer). C in {0.5, 2, 8} chosen on a time
           slice of train (fit on created_at < 2023-01-01, top-1 on the
           2023 slice); test is never used for selection. The final model
           is retrained on the whole train.
  package  (reference, from the signal era): on the java_trace subset of
           test, predict the most frequent train component per main trace
           package (profile_jira.main_package), falling back to level 0
           when the package never appeared in train.

Top-1 hit: the first predicted component is one of the issue's components.
Top-3 hit: any of the first three predicted components is one of the issue's
components (level 1: top three by predicted probability; ties keep the
stable, index-ordered sort, so results are deterministic).

Reads router.text_dataset and bronze.jira_issue_raw (descriptions, to recompute
the main package exactly as router.dataset does); writes only
docs/router_results.md. No writes to the database.

Run:
    .venv/bin/python scripts/router_text_baseline.py
"""
import datetime
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from profile_jira import load_env, main_package  # noqa: E402

CUTOFF = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)
CV_CUT = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
RANDOM_STATE = 42
C_CANDIDATES = (0.5, 2, 8)

DOC_PATH = REPO / "docs" / "router_results.md"


def load_rows():
    env = load_env(REPO / ".env")
    import psycopg2

    conn = psycopg2.connect(
        host=env["PGHOST"], port=env["PGPORT"], dbname=env["PGDATABASE"],
        user=env["PGUSER"], password=env["PGPASSWORD"])
    cur = conn.cursor()
    cur.execute(
        "SELECT issue_key, created_at, issue_type, text, components,"
        " has_java_trace, has_python_trace, has_error_class,"
        " has_exception_name, split FROM router.text_dataset"
        " WHERE split IN ('train', 'test') ORDER BY issue_key")
    rows = [tuple(r) for r in cur.fetchall()]
    cur.execute(
        "SELECT issue_key, payload->'fields'->>'description'"
        " FROM bronze.jira_issue_raw")
    desc = dict(cur.fetchall())
    conn.close()
    return rows, desc


def fit_level1(train_rows, C):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    t0 = time.perf_counter()
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3,
                          ngram_range=(1, 2), max_features=200000)
    Xtr = vec.fit_transform([r[3] for r in train_rows])
    labels = [list(r[4]) for r in train_rows]
    mlb = MultiLabelBinarizer()
    ytr = mlb.fit_transform(labels)
    clf = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE),
        n_jobs=-1)
    clf.fit(Xtr, ytr)
    # NOTE: with an indicator-matrix y, clf.classes_ is [0, 1, ..., K-1];
    # the label names are mlb.classes_, in the same column order as the
    # one-vs-rest estimators and the predict_proba columns.
    return vec, clf, mlb, time.perf_counter() - t0


def topk_idx(proba, k):
    """Per-row class indices of the k highest probabilities (stable ties)."""
    return np.argsort(-proba, axis=1, kind="stable")[:, :k]


def hit_counts(proba, names, actual, k):
    idx = topk_idx(proba, k)
    hit1 = hitk = 0
    for i, comps in enumerate(actual):
        cset = set(comps)
        if names[idx[i][0]] in cset:
            hit1 += 1
        if any(names[j] in cset for j in idx[i]):
            hitk += 1
    return hit1, hitk


def pct(hit, n):
    return f"{100.0 * hit / n:.1f}%" if n else "0.0%"


def group_hits(rows, names, proba_by_key, k):
    """(n, hit1, hitk) for a list of dataset rows, level 1 probabilities."""
    actual = [list(r[4]) for r in rows]
    proba = np.vstack([proba_by_key[r[0]] for r in rows]) if rows else np.zeros((0, len(names)))
    hit1, hitk = hit_counts(proba, names, actual, k)
    return len(rows), hit1, hitk


def main():
    rows, desc = load_rows()
    train = [r for r in rows if r[9] == "train"]
    test = [r for r in rows if r[9] == "test"]

    # ---- level 0 ------------------------------------------------------------
    comp_counts = Counter()
    for r in train:
        comp_counts.update(r[4])
    ranked = sorted(comp_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    l0_top1 = ranked[0][0]
    l0_top3 = [c for c, _ in ranked[:3]]

    def l0_hits(rs):
        hit1 = sum(1 for r in rs if l0_top1 in r[4])
        hit3 = sum(1 for r in rs if any(c in r[4] for c in l0_top3))
        return len(rs), hit1, hit3

    # ---- level 1: C selection on the 2023 slice of train (never test) -------
    cv_train = [r for r in train if r[1] < CV_CUT]
    cv_val = [r for r in train if CV_CUT <= r[1] < CUTOFF]
    val_results = []  # (C, val n, val top-1 hits)
    for C in C_CANDIDATES:
        vec, clf, mlb, fit_s = fit_level1(cv_train, C)
        names = list(mlb.classes_)
        val_proba = clf.predict_proba(vec.transform([r[3] for r in cv_val]))
        val_n = len(cv_val)
        val_hit1, _ = hit_counts(val_proba, names,
                                 [list(r[4]) for r in cv_val], 3)
        val_results.append((C, val_n, val_hit1, fit_s))
    best = max(val_results, key=lambda t: (t[2], -t[0]))  # most hits; ties -> smaller C
    best_C = best[0]

    # ---- level 1: final model on the whole train ----------------------------
    t0 = time.perf_counter()
    vec, clf, mlb, fit_s = fit_level1(train, best_C)
    train_secs = time.perf_counter() - t0
    names = list(mlb.classes_)
    names_set = set(names)

    t0 = time.perf_counter()
    test_proba = clf.predict_proba(vec.transform([r[3] for r in test]))
    predict_secs = time.perf_counter() - t0
    proba_by_key = {r[0]: test_proba[i] for i, r in enumerate(test)}

    # ---- package rule (reference) on the java_trace subset ------------------
    pkg_comps = {}
    for r in train:
        if not r[5]:
            continue
        pkg = main_package(desc.get(r[0]))
        if pkg is None:
            continue
        pkg_comps.setdefault(pkg, Counter()).update(r[4])
    pkg_best = {p: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                for p, c in pkg_comps.items()}
    test_jt = [r for r in test if r[5]]

    def pkg_hits(rs):
        hit = 0
        for r in rs:
            pkg = main_package(desc.get(r[0]))
            pred = pkg_best.get(pkg, l0_top1)
            if pred in r[4]:
                hit += 1
        return len(rs), hit

    # ---- evaluation ----------------------------------------------------------
    n_test = len(test)
    l0_all = l0_hits(test)
    l1_all = group_hits(test, names, proba_by_key, 3)
    l1_jt = group_hits(test_jt, names, proba_by_key, 3)
    pkg_all = pkg_hits(test_jt)

    type_counts = Counter(r[2] for r in test)
    top_types = sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:6]
    type_rows = []
    for t, n in top_types:
        rs = [r for r in test if r[2] == t]
        type_rows.append((t, n, l0_hits(rs), group_hits(rs, names, proba_by_key, 3)))

    sig = lambda r: any(r[i] for i in (5, 6, 7, 8))
    sig_rows = []
    for label, cond in (("has at least one signal", sig),
                        ("has no signal", lambda r: not sig(r))):
        rs = [r for r in test if cond(r)]
        sig_rows.append((label, l0_hits(rs), group_hits(rs, names, proba_by_key, 3)))

    comp_test_counts = Counter()
    for r in test:
        comp_test_counts.update(r[4])
    top15_comps = sorted(comp_test_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    comp_rows = []
    for c, n in top15_comps:
        rs = [r for r in test if c in r[4]]
        comp_rows.append((c, n, group_hits(rs, names, proba_by_key, 3)))

    # strongest features per top-10 component (largest logistic weights)
    feature_names = list(vec.get_feature_names_out())
    # OneVsRestClassifier keeps one binary model per label in estimators_,
    # in mlb.classes_ order (the indicator columns); each coef_ has shape
    # (1, n_features).
    coef = np.vstack([e.coef_[0] for e in clf.estimators_])
    top10_comps = top15_comps[:10]
    feature_rows = []
    for c, n in top10_comps:
        if c not in names_set:
            feature_rows.append((c, n, None))
            continue
        w = coef[list(names).index(c)]
        top10 = sorted(zip(feature_names, w.tolist()),
                       key=lambda t: (-t[1], t[0]))[:10]
        feature_rows.append((c, n, top10))

    # ---- docs/router_results.md ----------------------------------------------
    md = []
    md.append("# Router text classifier — baselines")
    md.append("")
    md.append("Regenerated by `scripts/router_text_baseline.py` over"
              " `router.text_dataset` (built by `scripts/router_text_dataset.py`);"
              " identical on every run. Learned only from train, measured only"
              " on test, `random_state=42`. The error signals are not model"
              " inputs; they only slice the report.")
    md.append("")
    md.append("## Cut")
    md.append("")
    md.append("- cut: `train` if `created_at < 2024-01-01` (UTC), `test` if `>=`"
              " (silver's corrected created_at)")
    md.append(f"- train: {len(train)} issues, test: {n_test} issues")
    md.append(f"- chosen C: {best_C} (highest top-1 on the 2023 validation"
              " slice of train; test never used for selection)")
    md.append("")
    md.append("## C selection (fit on train with `created_at < 2023-01-01`,"
              " top-1 evaluated on the 2023 slice)")
    md.append("")
    md.append("| C | top-1 | n |")
    md.append("|---:|------:|---:|")
    for C, n, h, _s in val_results:
        md.append(f"| {C} | {pct(h, n)} | {n} |")
    md.append("")
    md.append("## Overall test")
    md.append("")
    md.append(f"n = {n_test}")
    md.append("")
    md.append("| level | top-1 | top-3 |")
    md.append("|-------|------:|------:|")
    md.append(f"| level 0 (most frequent train components:"
              f" {', '.join('`' + c + '`' for c in l0_top3)}) |"
              f" {pct(l0_all[1], n_test)} | {pct(l0_all[2], n_test)} |")
    md.append(f"| level 1 (tf-idf + logistic regression, C={best_C}) |"
              f" {pct(l1_all[1], n_test)} | {pct(l1_all[2], n_test)} |")
    md.append("")
    md.append("## By issue type (6 most frequent in test)")
    md.append("")
    md.append("| issue_type | n | level 0 top-1 | level 0 top-3"
              " | level 1 top-1 | level 1 top-3 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for t, n, l0, l1 in type_rows:
        md.append(f"| {t} | {n} | {pct(l0[1], n)} | {pct(l0[2], n)}"
                  f" | {pct(l1[1], n)} | {pct(l1[2], n)} |")
    md.append("")
    md.append("## By error-signal presence (whole test)")
    md.append("")
    md.append("| group | n | level 0 top-1 | level 0 top-3"
              " | level 1 top-1 | level 1 top-3 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for label, l0, l1 in sig_rows:
        md.append(f"| {label} | {l0[0]} | {pct(l0[1], l0[0])}"
                  f" | {pct(l0[2], l0[0])} | {pct(l1[1], l1[0])}"
                  f" | {pct(l1[2], l1[0])} |")
    md.append("")
    md.append("## Java-trace subset of test: package rule vs level 1")
    md.append("")
    md.append(f"n = {len(test_jt)} (issues whose description carries a main"
              " Java trace package)")
    md.append("")
    md.append("| method | top-1 |")
    md.append("|---|------:|")
    md.append(f"| package rule (most frequent train component per package,"
              f" fallback level 0 `{l0_top1}`) | {pct(pkg_all[1], pkg_all[0])} |")
    md.append(f"| level 1 (tf-idf + logistic regression, C={best_C})"
              f" | {pct(l1_jt[1], l1_jt[0])} |")
    md.append("")
    md.append("## Top 15 components by test frequency (level 1)")
    md.append("")
    md.append("| component | n | top-1 | top-3 |")
    md.append("|---|---:|------:|------:|")
    for c, n, l1 in comp_rows:
        md.append(f"| {c} | {n} | {pct(l1[1], n)} | {pct(l1[2], n)} |")
    md.append("")
    md.append("## What level 1 learned: 10 strongest features per top-10"
              " component (by test frequency)")
    md.append("")
    md.append("Largest logistic-regression weights (over the tf-idf features,"
              " unigrams and bigrams) of each component's one-vs-rest model.")
    for c, n, feats in feature_rows:
        md.append("")
        md.append(f"### `{c}` (n = {n} in test)")
        md.append("")
        if feats is None:
            md.append("(component absent from train — no model, nothing to explain)")
            continue
        md.append("| # | feature | weight |")
        md.append("|---:|---|------:|")
        for i, (f, w) in enumerate(feats, 1):
            md.append(f"| {i} | {f} | {w:+.3f} |")
    md.append("")
    DOC_PATH.write_text("\n".join(md), encoding="utf-8")

    # ---- stdout summary ------------------------------------------------------
    print("=== router text baselines (train only; measured on test) ===")
    print(f"train: {len(train)}   test: {n_test}")
    print(f"level 0: top-1 `{l0_top1}`"
          f" | top-3 {', '.join('`' + c + '`' for c in l0_top3)}")
    print(f"C selection on the 2023 slice (n={best[1]}):")
    for C, n, h, s in val_results:
        mark = "  <-- chosen" if C == best_C else ""
        print(f"  C={C}: val top-1 {pct(h, n)}  (fit {s:.1f} s){mark}")
    print(f"overall test: level 0 top-1 {pct(l0_all[1], n_test)}"
          f" / top-3 {pct(l0_all[2], n_test)} | level 1 top-1"
          f" {pct(l1_all[1], n_test)} / top-3 {pct(l1_all[2], n_test)}")
    print(f"java-trace test subset (n={len(test_jt)}): package rule"
          f" {pct(pkg_all[1], pkg_all[0])} | level 1 {pct(l1_jt[1], l1_jt[0])}")
    print(f"training (final model on whole train): {train_secs:.1f} s")
    print(f"prediction (whole test): {predict_secs:.1f} s")
    print(f"wrote {DOC_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
