#!/usr/bin/env python3
"""Experiment 4 (journal 4.8): novelty detection for the selected variant.

Question. The `rolling` variant (monthly retraining; the variant selected
in experiment 3, docs/router_recency.md) can only predict components it
was trained on: its prediction classes are exactly the components present
in its training labels. Issues whose final components are ALL outside that
set are NOVEL - the router cannot route them correctly, yet its confidence
signals may still look high. This experiment quantifies that harm and
asks whether a nearest-training-example signal (nn_sim) can gate it out.

Method. For each month M (2023-01 .. last data month) the model is trained
exactly as the `rolling` variant of scripts/router_recency.py:
V-token+balanced, C = 8, training pool = issues created before the first
day of M, labels = components as known at that day (as-of; issues whose
as-of list is empty are dropped), prediction targets = the issues created
in M. The fit function (fit_balanced), the month loop, the as-of labels
and the validation/test split (validation = the 2023 months, test =
2024 onward) are REUSED from router_recency.py - no training code is
duplicated. choose_threshold and auto_stats are reused from
router_confidence.py.

Definitions (per month M).
  known(M) = the set of components that appear in at least one as-of
    training label of the month-M training set (exactly the model's
    class set).
  An issue created in M is NOVEL if none of its final components is in
    known(M). A novel issue auto-routed at any threshold is wrong by
    definition (the predicted component is a model class, hence in
    known(M), hence outside the issue's final components).

Signals per issue (no new model):
  top1    top-1 probability;
  margin  top1 - top2;
  nn_sim  max cosine similarity between the issue's tf-idf vector and
          the month-M training matrix (sparse dot product; tf-idf rows
          are L2-normalised, asserted within 1e-6 for every non-empty
          row; rows with no tf-idf features, if any, have nn_sim 0).

Thresholds. t90 = the 90% margin threshold of `rolling` in
docs/router_recency.md (parsed from that doc and asserted to equal
choose_threshold on the recomputed rolling validation margins). Gate
thresholds s = the 1st, 2nd and 5th percentile of nn_sim of the
KNOWN-and-auto-routed VALIDATION issues at t90 - never chosen with
novelty labels. Test is never used to choose anything.

Fit-function change (equivalence proof). fit_balanced in
router_recency.py gained an optional argument return_train=False; the
default keeps the previous 4-element return and identical behaviour.
Before the month loop the 2024-01 month is fitted once with the old call
(4 elements) and once with the new call (5 elements) and the predicted
probabilities are asserted identical with np.array_equal.

Resumable: each month is cached to scratch/novelty/<YYYY-MM>.npz
(keys, created, final components, top1, margin, top-1 probability,
nn_sim, novel flag) and skipped if present; delete the directory to force
a rerun. Progress is printed per month with elapsed time.

Writes docs/router_novelty.md (English), byte-identical on every run;
never edit by hand.

Run:
    .venv/bin/python scripts/router_novelty.py
"""
import bisect
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from profile_jira import load_env  # noqa: E402
from router_parent_context import components_asof, load_component_changes  # noqa: E402
from router_text_baseline import (  # noqa: E402
    CV_CUT, extend_text, load_rows, pct)
from router_confidence import auto_stats, choose_threshold  # noqa: E402
from router_recency import fit_balanced, make_predict, month_window  # noqa: E402

SCRATCH = REPO / "scratch" / "novelty"
DOC_PATH = REPO / "docs" / "router_novelty.md"
RECENCY_DOC = REPO / "docs" / "router_recency.md"
NOVEL_MIN = 5            # below this a split gets "too few novel issues"
PERCENTILES = (1, 2, 5)  # gate percentiles of nn_sim
DP = "Declarative Pipelines"


def load_data():
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
    return summary, parent_key_map, changes, by_time, times, months, data_end


def train_pool(ym, times, by_time, changes):
    """The exact `rolling` construction of router_recency.run_rolling:
    pool = issues created before the first day of M; training rows/labels =
    the pool issues with non-empty as-of labels at that day."""
    start, mrows = month_window(ym, times, by_time)
    pool = by_time[:bisect.bisect_left(times, start)]
    tr_rows, tr_labels = [], []
    for r in pool:
        la = components_asof(sorted(r[4]), changes.get(r[0], []), start)
        if la:
            tr_rows.append(r)
            tr_labels.append(la)
    return start, mrows, tr_rows, tr_labels


def known_set(ym, times, by_time, changes):
    """known(M) = components appearing in at least one as-of training label
    of the month-M training set (recomputed from the changelog; equals the
    model's class set used in the month loop)."""
    start, _mrows, _tr_rows, tr_labels = train_pool(ym, times, by_time, changes)
    known = set()
    for la in tr_labels:
        known.update(la)
    return known


def prove_fit_equivalence(times, by_time, changes, parent_key_map, summary):
    """Fit 2024-01 with the old call (4-element return, the previous
    behaviour) and with the new call (return_train=True, 5-element return)
    and assert the predicted probabilities are identical."""
    ym = (2024, 1)
    _start, mrows, tr_rows, tr_labels = train_pool(ym, times, by_time, changes)
    vec_o, clf_o, _mlb_o, fit_o = fit_balanced(
        tr_rows, tr_labels, None, parent_key_map, summary)
    vec_n, clf_n, _mlb_n, fit_n, Xtr = fit_balanced(
        tr_rows, tr_labels, None, parent_key_map, summary, return_train=True)
    p_o = np.array(make_predict(vec_o, clf_o, parent_key_map, summary)(mrows))
    p_n = np.array(make_predict(vec_n, clf_n, parent_key_map, summary)(mrows))
    assert p_o.shape == p_n.shape
    assert np.array_equal(p_o, p_n), \
        "return_train=True changed the predicted probabilities"
    max_diff = float(np.abs(p_o - p_n).max())
    print(f"[equivalence] 2024-01: train={len(tr_rows)} month={len(mrows)} "
          f"old call fit={fit_o:.1f}s, new call fit={fit_n:.1f}s, "
          f"proba shape {p_o.shape}, max abs diff {max_diff:.3e} "
          f"-> np.array_equal True")
    return {"n_train": len(tr_rows), "n_month": len(mrows),
            "fit_old": fit_o, "fit_new": fit_n, "shape": p_o.shape,
            "max_diff": max_diff}


def save_month(ym, mrows, top1, margin, top1prob, nn_sim, novel):
    path = SCRATCH / f"{ym[0]:04d}-{ym[1]:02d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keys=np.array([r[0] for r in mrows]),
        created=np.array([r[1].timestamp() for r in mrows], dtype=float),
        final=np.array([list(r[4]) for r in mrows], dtype=object),
        top1=np.array(top1),
        margin=margin,
        top1prob=top1prob,
        nn_sim=nn_sim,
        novel=novel)
    return path


def load_month(ym):
    path = SCRATCH / f"{ym[0]:04d}-{ym[1]:02d}.npz"
    assert path.exists(), f"missing {path}"
    with np.load(path, allow_pickle=True) as z:
        return {
            "keys": z["keys"].tolist(),
            "created": z["created"],
            "final": [list(a) for a in z["final"]],
            "top1": z["top1"].tolist(),
            "margin": z["margin"],
            "top1prob": z["top1prob"],
            "nn_sim": z["nn_sim"],
            "novel": z["novel"].astype(bool),
        }


def run_months(months, times, by_time, changes, parent_key_map, summary,
               t_start):
    for ym in months:
        path = SCRATCH / f"{ym[0]:04d}-{ym[1]:02d}.npz"
        if path.exists():
            print(f"[novelty] {ym[0]:04d}-{ym[1]:02d} cached "
                  f"(elapsed={(time.perf_counter() - t_start) / 60:.1f}min)")
            continue
        t0 = time.perf_counter()
        _start, mrows, tr_rows, tr_labels = train_pool(
            ym, times, by_time, changes)
        vec, clf, mlb, fit_s, Xtr = fit_balanced(
            tr_rows, tr_labels, None, parent_key_map, summary,
            return_train=True)
        names = list(mlb.classes_)
        known = set(names)
        texts = [extend_text(r, parent_key_map, summary, False)
                 for r in mrows]
        X = vec.transform(texts)
        proba = make_predict(vec, clf, parent_key_map, summary)(mrows)
        n = len(mrows)
        top1 = [names[int(np.argmax(proba[i]))] for i in range(n)]
        margin = np.array([
            float(proba[i].max() - np.partition(proba[i], -2)[-2])
            for i in range(n)])
        top1prob = np.array([float(proba[i].max()) for i in range(n)])

        def row_norms(Z):
            return np.sqrt((Z.multiply(Z)).sum(axis=1).A1())

        tr_nnz = np.diff(Xtr.indptr)
        q_nnz = np.diff(X.indptr)
        # tf-idf L2-normalised rows: assert unit norm within 1e-6 (every
        # non-empty row); empty rows keep nn_sim 0 below.
        assert np.all(np.abs(row_norms(Xtr)[tr_nnz > 0] - 1.0) <= 1e-6)
        assert np.all(np.abs(row_norms(X)[q_nnz > 0] - 1.0) <= 1e-6)
        nn_sim = np.zeros(n)
        if n and (tr_nnz > 0).any():
            for i0 in range(0, n, 256):
                D = (X[i0:i0 + 256] @ Xtr.T).toarray()
                nn_sim[i0:i0 + 256] = D.max(axis=1)
        novel = np.array([not (set(r[4]) & known) for r in mrows])
        save_month(ym, mrows, top1, margin, top1prob, nn_sim, novel)
        print(f"[novelty] {ym[0]:04d}-{ym[1]:02d} pool={len(by_time[:bisect.bisect_left(times, _start)])} "
              f"train={len(tr_rows)} known={len(known)} novel={int(novel.sum())} "
              f"empty_train={int((tr_nnz == 0).sum())} "
              f"empty_month={int((q_nnz == 0).sum())} fit={fit_s:.1f}s "
              f"month={time.perf_counter() - t0:.1f}s "
              f"elapsed={(time.perf_counter() - t_start) / 60:.1f}min")


def parse_t90():
    txt = RECENCY_DOC.read_text(encoding="utf-8")
    sec = txt.split("Target precision 90%")[1].split("Target precision 80%")[0]
    m = re.search(r"^\| rolling \| (\d+\.\d+) \|", sec, re.M)
    assert m, "rolling 90% row not found in docs/router_recency.md"
    return float(m.group(1))


def combine_split(data, years):
    order = sorted(ym for ym in data if ym[0] in years)
    out = {"n": 0, "keys": [], "created": [], "final": [], "top1": [],
           "margin": np.zeros(0), "top1prob": np.zeros(0),
           "nn_sim": np.zeros(0), "novel": np.zeros(0, dtype=bool)}
    for ym in order:
        d = data[ym]
        out["keys"].extend(d["keys"])
        out["created"].extend(d["created"])
        out["final"].extend(d["final"])
        out["top1"].extend(d["top1"])
        out["margin"] = np.concatenate([out["margin"], d["margin"]])
        out["top1prob"] = np.concatenate([out["top1prob"], d["top1prob"]])
        out["nn_sim"] = np.concatenate([out["nn_sim"], d["nn_sim"]])
        out["novel"] = np.concatenate([out["novel"], d["novel"]])
    out["n"] = len(out["keys"])
    out["correct1"] = [t in a for t, a in zip(out["top1"], out["final"])]
    return out


def auroc(novel, sig):
    """AUROC of novel (1) vs known (0) for one signal; None when the split
    has fewer than NOVEL_MIN novel issues."""
    n_novel = int(novel.sum())
    if n_novel < NOVEL_MIN or n_novel == len(novel):
        return None
    return roc_auc_score(novel.astype(int), sig)


def fmt_auroc(v):
    return "too few novel issues" if v is None else f"{v:.3f}"


def routed_string(counter, top=5):
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k} ({v})" for k, v in items[:top]) if items else "-"


def per_component_tables(data, months, times, by_time, changes):
    """Per-component novel counts (novel issues containing the component,
    per split) and first appearance = first month in which the component
    appears in the final components of an issue created in that month and
    is outside known(M)."""
    comp_first = {}
    for ym in months:
        d = data[ym]
        known = known_set(ym, times, by_time, changes)
        for f in d["final"]:
            for c in f:
                if c not in known and c not in comp_first:
                    comp_first[c] = ym
    return comp_first


def gate_row(split, s):
    """Auto-route if margin >= t90 and nn_sim >= s (s = 0.0 -> no gate)."""
    auto = (split["margin"] >= T90_G) & (split["nn_sim"] >= s)
    n_auto = int(auto.sum())
    wrong = int(sum(1 for i in np.flatnonzero(auto)
                    if not split["correct1"][i]))
    correct = n_auto - wrong
    n_novel_auto = int((auto & split["novel"]).sum())
    cov = n_auto / split["n"] if split["n"] else 0.0
    prec = correct / n_auto if n_auto else 0.0
    wp100 = (100.0 * wrong / n_auto) if n_auto else None
    return {"n_auto": n_auto, "n_novel_auto": n_novel_auto, "cov": cov,
            "prec": prec, "wp100": wp100}


T90_G = None


def fmt_wp100(v):
    return "-" if v is None else f"{v:.1f}"


def build_doc(ctx):
    L = []
    a = L.append
    a("# Router novelty detection (experiment 4)")
    a("")
    a("Generated by `scripts/router_novelty.py` on every run; never edit by")
    a("hand.")
    a("")
    a("**Question.** The selected variant of experiment 3, `rolling` (monthly")
    a("retraining, V-token+balanced, C = 8), can only predict components it")
    a("was trained on. Issues whose final components are all outside that")
    a("set are NOVEL and can never be routed correctly - yet their margin")
    a("and top-1 probabilities may still look high. How much of the")
    a("auto-routing at the 90% margin threshold is this, and does a")
    a("nearest-training-example signal (nn_sim) gate it out?")
    a("")
    a("**Base model and reuse.** For each month M the model is the `rolling")
    a("variant of docs/router_recency.md: trained on the issues created")
    a("before the first day of M, predicting the issues created in M. The")
    a("fit function (`fit_balanced`), the month loop, the as-of training")
    a("labels and the validation/test split are reused from")
    a("`scripts/router_recency.py`; `choose_threshold` and `auto_stats`")
    a("from `scripts/router_confidence.py`. No training code is duplicated.")
    a("")
    p = ctx["proof"]
    a("**Fit-function change (equivalence proof).** `fit_balanced` in")
    a("`scripts/router_recency.py` gained an optional argument")
    a("`return_train=False`; the default keeps the previous 4-element return")
    a("and identical behaviour. The 2024-01 month (train = "
      f"{p['n_train']} issues) was fitted with the old call (4 elements) "
      f"and with the new call (5 elements): predicted probabilities shape "
      f"{p['shape']}, max abs diff {p['max_diff']:.3e}, np.array_equal True.")
    a("")
    a("**Definitions (per month M).** known(M) = the set of components that")
    a("appear in at least one as-of training label of the month-M training")
    a("set (exactly the model's class set). An issue created in M is NOVEL")
    a("if none of its final components is in known(M). A novel issue")
    a("auto-routed at any threshold is wrong by definition (the predicted")
    a("component is a model class, hence in known(M), hence outside the")
    a("issue's final components).")
    a("")
    a("**Signals (per issue, no new model).** top1 = top-1 probability;")
    a("margin = top1 - top2; nn_sim = max cosine similarity between the")
    a("issue's tf-idf vector and the month-M training matrix (sparse dot")
    a("product; tf-idf rows are asserted L2-normalised within 1e-6).")
    a("")
    s = ctx["s_vals"]
    a("**Thresholds.** t90 = {:.2f}, the 90% margin threshold of `rolling`".format(
        ctx["t90"]))
    a("in docs/router_recency.md (parsed from that doc and asserted to")
    a("equal choose_threshold on the recomputed rolling validation")
    a("margins). Gates s = the 1st, 2nd and 5th percentile of nn_sim of the")
    a("KNOWN-and-auto-routed VALIDATION issues at t90 (s = {:.4f}, {:.4f},".format(
        s[0], s[1]))
    a("{:.4f} respectively) - never chosen with novelty labels. Test is".format(s[2]))
    a("never used to choose anything; test numbers are reference only.")
    a("")
    a("**Splits and months.** validation = the 2023 months ({} issues);".format(
        ctx["val"]["n"]))
    a("test = created_at >= 2024-01-01 ({} issues, data ends {}).".format(
        ctx["test"]["n"], ctx["data_end"]))
    a("")
    a("### a) Novel issues per split and per component")
    a("")
    a("| split | n issues | novel | share |")
    a("|---|---|---|---|")
    for name, sp in (("validation", ctx["val"]), ("test", ctx["test"])):
        a("| {} | {} | {} | {} |".format(
            name, sp["n"], int(sp["novel"].sum()),
            pct(int(sp["novel"].sum()), sp["n"])))
    a("")
    a("Per component: number of NOVEL issues containing it, per split, and")
    a("first appearance = the first month in which the component appears in")
    a("the final components of an issue created in that month and is")
    a("outside known(M).")
    a("")
    a("| component | novel (val) | novel (test) | first appearance |")
    a("|---|---|---|---|")
    cv, ct, cf = ctx["comp_val"], ctx["comp_test"], ctx["comp_first"]
    for c in sorted(cv | ct, key=lambda c: (-(cv[c] + ct[c]), c)):
        a("| {} | {} | {} | {}-{:02d} |".format(
            c, cv[c], ct[c], cf[c][0], cf[c][1]))
    a("")
    a("### c) AUROC of \"novel vs known\", per signal")
    a("")
    a("AUC > 0.5 means higher signal values indicate novel issues. A split")
    a("with fewer than 5 novel issues is reported as too few novel issues.")
    a("")
    a("| signal | validation | test |")
    a("|---|---|---|")
    for name in ("top1", "margin", "nn_sim"):
        a("| {} | {} | {} |".format(
            name, fmt_auroc(ctx["auroc"][("val", name)]),
            fmt_auroc(ctx["auroc"][("test", name)])))
    a("")
    a("### d) The harm at t90 = {:.2f}".format(ctx["t90"]))
    a("")
    a("Novel issues that get auto-routed (margin >= t90) - all of them are")
    a("wrong by definition - and where they were routed (top-1 component).")
    a("")
    a("| split | novel | novel auto-routed | share of novel | routed to (top 5) |")
    a("|---|---|---|---|---|")
    for name, sp in (("validation", ctx["val"]), ("test", ctx["test"])):
        h = ctx["harm"][name]
        a("| {} | {} | {} | {} | {} |".format(
            name, h["n_novel"], h["n_novel_auto"],
            pct(h["n_novel_auto"], h["n_novel"]), h["routed"]))
    a("")
    a("Per component (novel issues containing it; novel auto-routed at")
    a("t90; where those were routed):")
    a("")
    a("| component | novel (val) | auto (val) | routed (val) | novel (test) | auto (test) | routed (test) |")
    a("|---|---|---|---|---|---|---|")
    for row in ctx["harm_comps"]:
        a("| {} | {} | {} | {} | {} | {} | {} |".format(*row))
    a("")
    a("### e) Gate: auto-route only if margin >= t90 and nn_sim >= s")
    a("")
    a("s is the 1st / 2nd / 5th percentile of nn_sim of the KNOWN-and-")
    a("auto-routed VALIDATION issues at t90 ({} such issues).".format(
        ctx["n_known_auto_val"]))
    a("")
    a("| gate | s | val novel auto | val coverage | val precision | val wrong/100 | "
      "test novel auto | test coverage | test precision | test wrong/100 | "
      "cov loss vs no gate (val, pp) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in ctx["gate_rows"]:
        a("| {} |".format(" | ".join(row)))
    a("")
    a("**Gate selected by the validation rule** (the gate with the fewest")
    a("validation novel auto-routes whose validation coverage loss vs no")
    a("gate is at most 2 pp): " + ctx["gate_selected"])
    a("")
    a("**Test was never used to choose anything** (no variant, no threshold,")
    a("no gate); test numbers are reference only.")
    a("")
    a("### f) First 20 issues of Declarative Pipelines by created_at")
    a("")
    a("auto @ t90 = margin >= {:.2f}; blocked @ s2 = the 2nd-percentile gate".format(
        ctx["t90"]))
    a("(nn_sim < {:.4f}) stops an issue that would otherwise be".format(ctx["s_vals"][1]))
    a("auto-routed.")
    a("")
    a("| # | key | month | novel | top1 | top1 prob | margin | nn_sim | auto @ t90 | blocked @ s2 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for row in ctx["dp_rows"]:
        a("| {} |".format(" | ".join(row)))
    a("")
    return "\n".join(L) + "\n"


def main():
    global T90_G
    t_start = time.perf_counter()
    summary, parent_key_map, changes, by_time, times, months, data_end = \
        load_data()

    # ---- equivalence proof for the fit_balanced change ----
    proof = prove_fit_equivalence(times, by_time, changes, parent_key_map,
                                  summary)

    # ---- month loop (reused rolling construction, resumable) ----
    run_months(months, times, by_time, changes, parent_key_map, summary,
               t_start)
    data = {ym: load_month(ym) for ym in months}
    val = combine_split(data, {2023})
    test = combine_split(data, {2024, 2025, 2026})
    assert val["n"] == 4733 and test["n"] == 15754

    # ---- threshold: must match docs/router_recency.md ----
    t90_doc = parse_t90()
    t90 = choose_threshold(val["margin"], val["correct1"], 0.90)
    assert t90 == t90_doc, f"t90 {t90} != {t90_doc} in docs/router_recency.md"
    T90_G = t90
    # cross-checks against the documented `rolling` numbers (same model,
    # same data, same split): val top-1 3462/4733, test 11579/15754,
    # val auto at t90 = 1243/4733.
    assert sum(val["correct1"]) == 3462, sum(val["correct1"])
    assert sum(test["correct1"]) == 11579, sum(test["correct1"])
    n, n_auto, wrong, _wp100 = auto_stats(val["margin"], val["correct1"], t90)
    assert (n, n_auto) == (4733, 1243), (n, n_auto)
    print(f"[check] t90 = {t90} matches docs/router_recency.md; "
          f"val top-1 = 3462/4733, test top-1 = 11579/15754, "
          f"val auto at t90 = 1243/4733 (wrong {wrong})")

    # ---- 1a: novel counts per split and per component ----
    def novel_comp_count(sp):
        c = Counter()
        for i in np.flatnonzero(sp["novel"]):
            for comp in set(sp["final"][i]):
                c[comp] += 1
        return c

    comp_val = novel_comp_count(val)
    comp_test = novel_comp_count(test)
    comp_first = per_component_tables(data, months, times, by_time, changes)
    auroc_map = {}
    for sp_name, sp in (("val", val), ("test", test)):
        for sig_name in ("top1", "margin", "nn_sim"):
            sig = {"top1": sp["top1prob"], "margin": sp["margin"],
                   "nn_sim": sp["nn_sim"]}[sig_name]
            auroc_map[(sp_name, sig_name)] = auroc(sp["novel"], sig)
    harm = {}
    harm_comps = []
    for sp_name, sp in (("validation", val), ("test", test)):
        novel_auto_idx = [i for i in np.flatnonzero(sp["novel"])
                          if sp["margin"][i] >= t90]
        harm[sp_name] = {
            "n_novel": int(sp["novel"].sum()),
            "n_novel_auto": len(novel_auto_idx),
            "routed": routed_string(Counter(
                sp["top1"][i] for i in novel_auto_idx))}
    for c in sorted(set(comp_val) | set(comp_test),
                    key=lambda c: (-(comp_val[c] + comp_test[c]), c)):
        row = [c]
        for sp in (val, test):
            idx = [i for i in np.flatnonzero(sp["novel"])
                   if c in set(sp["final"][i])]
            auto = [i for i in idx if sp["margin"][i] >= t90]
            row += [len(idx), len(auto), routed_string(
                Counter(sp["top1"][i] for i in auto), top=3)]
        harm_comps.append(row)

    # ---- 1e: percentile gates (validation nn_sim, no novelty labels) ----
    known_auto_val = [i for i in range(len(val["novel"]))
                      if not val["novel"][i] and val["margin"][i] >= t90]
    s_vals = np.percentile(
        np.array([val["nn_sim"][i] for i in known_auto_val]), PERCENTILES)
    val_nogate = gate_row(val, 0.0)
    assert val_nogate["n_auto"] == 1243  # matches docs/router_recency.md
    test_nogate = gate_row(test, 0.0)
    gate_rows = []

    def gate_row_list(p_label, s, gv, gt, loss_pp):
        return [
            p_label, f"{s:.4f}", str(gv["n_novel_auto"]),
            pct(gv["n_auto"], val["n"]),
            pct(int(gv["prec"] * gv["n_auto"]), gv["n_auto"]) if gv["n_auto"] else "-",
            fmt_wp100(gv["wp100"]), str(gt["n_novel_auto"]),
            pct(gt["n_auto"], test["n"]),
            pct(int(gt["prec"] * gt["n_auto"]), gt["n_auto"]) if gt["n_auto"] else "-",
            fmt_wp100(gt["wp100"]), f"{loss_pp:.1f}"]

    gate_rows.append(gate_row_list("no gate", 0.0, val_nogate, test_nogate, 0.0))
    loss_by_p = {}
    for p in PERCENTILES:
        s = float(s_vals[p - 1])
        gv = gate_row(val, s)
        gt = gate_row(test, s)
        loss_pp = (val_nogate["cov"] - gv["cov"]) * 100.0
        loss_by_p[p] = loss_pp
        gate_rows.append(gate_row_list(f"{p}th percentile", s, gv, gt,
                                       loss_pp))
    ORDS = {1: "1st", 2: "2nd", 5: "5th"}
    eligible = [p for p in PERCENTILES if loss_by_p[p] <= 2.0]
    if eligible:
        best_p = min(eligible, key=lambda p: (
            int(gate_row(val, float(s_vals[p - 1]))["n_novel_auto"]), p))
        gb = gate_row(val, float(s_vals[best_p - 1]))
        gate_selected = (
            f"the {ORDS[best_p]} percentile (s = {s_vals[best_p - 1]:.4f}): "
            f"{gb['n_novel_auto']} validation novel issues still auto-routed "
            f"vs no gate, coverage loss {loss_by_p[best_p]:.1f} pp "
            f"(tie-break, if ever used: the smaller percentile)")
    else:
        gate_selected = ("none - every gate loses more than 2 pp of "
                        "validation coverage vs no gate, so no gate "
                        "qualifies under that rule")

    # ---- 1f: first 20 Declarative Pipelines issues by created_at ----
    s2 = float(s_vals[1])
    dp = []
    for ym in months:
        d = data[ym]
        for i, f in enumerate(d["final"]):
            if DP in f:
                dp.append((d["created"][i], d["keys"][i], ym, i, d))
    dp.sort(key=lambda t: (t[0], t[1]))
    dp_rows = []
    for rank, (created, key, ym, i, d) in enumerate(dp[:20], 1):
        auto_t90 = d["margin"][i] >= t90
        blocked = auto_t90 and d["nn_sim"][i] < s2
        dp_rows.append([
            str(rank), key, f"{ym[0]:04d}-{ym[1]:02d}",
            "yes" if d["novel"][i] else "no", d["top1"][i],
            f"{d['top1prob'][i]:.3f}", f"{d['margin'][i]:.4f}",
            f"{d['nn_sim'][i]:.4f}", "yes" if auto_t90 else "no",
            "yes" if blocked else "no"])

    ctx = {"proof": proof, "t90": t90, "val": val, "test": test,
           "data_end": data_end, "comp_val": comp_val, "comp_test": comp_test,
           "comp_first": comp_first, "auroc": auroc_map, "harm": harm,
           "harm_comps": harm_comps, "n_known_auto_val": len(known_auto_val),
           "s_vals": s_vals, "gate_rows": gate_rows,
           "gate_selected": gate_selected, "dp_rows": dp_rows}
    DOC_PATH.write_text(build_doc(ctx), encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    print(f"total elapsed: {(time.perf_counter() - t_start) / 60:.1f}min")


if __name__ == "__main__":
    main()
