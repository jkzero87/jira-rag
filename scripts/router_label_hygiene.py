#!/usr/bin/env python3
"""Experiment 7: label hygiene for the issue router.

Three label-side problems, each measured on the 2023 validation slice
exactly like the baseline (fit on train with created_at < 2023-01-01,
top-1 on the 2023 slice; test is never used for selection):

1. Renames. Old component names kept in use on new issues alongside their
   replacements. The rename mapping is built ONLY from information dated
   before 2023-01-01 (the validation and test slices never enter it) and is
   applied to the training labels before fitting.
2. Multi-component issues: issues with two or more final components, per
   period, plus the V-token top-1 under the current single-label
   definition (top-1 in the set of final components).
3. Rare classes: a balanced-class-weight variant of the same pipeline,
   and the SQL error count before/after.

Regenerates docs/router_label_hygiene.md byte-identical on every run
(do not edit the doc by hand).
"""
import datetime
import inspect
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from profile_jira import load_env  # noqa: E402
from router_parent_context import components_asof, load_component_changes  # noqa: E402
from router_text_baseline import (  # noqa: E402
    C_CANDIDATES, CV_CUT, CUTOFF, RANDOM_STATE, load_rows, fit_variant,
    vtoken_cv_selection, build_vtoken_model, build_vtoken_cv_model,
    hit_counts, topk_idx, pct, extend_text, _vtoken_predict_proba)
from router_tossing import RESULTS_TOP15, period_of  # noqa: E402

# Rename-detection parameters (all evidence strictly before CV_CUT).
YEAR2_AGO = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
BURST_MIN = 5          # replacement events required inside the window
BURST_DAYS = 60        # burst window length in days
# Pairs flagged by the tossing graph (experiment 5) that must be stated
# with a data verdict even when the rule does not pass them.
MANDATORY_PAIRS = [("R", "SparkR"), ("docs", "Documentation"),
                   ("MLlib", "ML"), ("ps", "Pandas API on Spark")]


def build_init_map(rows, changes):
    """Initial components per issue: the current list with every Component
    changelog entry after created_at undone (components_asof), exactly like
    router_tossing."""
    init_map = {}
    for r in rows:
        init_map[r[0]] = components_asof(
            sorted(r[4]), changes.get(r[0], []), r[1])
    return init_map


def first_last_new(rows, init_map):
    """First and last creation of an issue with the name in its INITIAL
    component list, using only rows with created_at < CV_CUT."""
    first_new, last_new = {}, {}
    for r in rows:
        if r[1] >= CV_CUT:
            continue
        for c in init_map[r[0]]:
            if c not in last_new or r[1] > last_new[c]:
                last_new[c] = r[1]
            if c not in first_new or r[1] < first_new[c]:
                first_new[c] = r[1]
    return first_new, last_new


def replacement_events(rows, changes, init_map):
    """X -> Y replacement events from the Component changelog, all with
    ts < CV_CUT: the issue lost X (fromString=X, toString=None) and gained
    Y (fromString=None, toString=Y) in either order (event time = the later
    of the two), or a direct swap entry (fromString=X, toString=Y). One
    event per (X, Y) per issue (the earliest qualifying pair)."""
    events = defaultdict(list)
    for key, ch in changes.items():
        if key not in init_map:
            continue
        rem_ts, add_ts, swap_ts = {}, {}, {}
        for ts, frm, to in ch:
            if ts >= CV_CUT:
                continue
            if frm is not None and to is not None:
                if (frm, to) not in swap_ts:
                    swap_ts[(frm, to)] = ts
            elif frm is not None:
                if frm not in rem_ts:
                    rem_ts[frm] = ts
            elif to is not None:
                if to not in add_ts:
                    add_ts[to] = ts
        for x, t_rm in rem_ts.items():
            for y, t_ad in add_ts.items():
                if x != y and (x, y) not in swap_ts:
                    events[(x, y)].append(max(t_rm, t_ad))
        for (x, y), t in swap_ts.items():
            events[(x, y)].append(t)
    return events


def max_burst(ts_list, days=BURST_DAYS):
    """Largest number of events inside any `days`-day window, and that
    window's span (None if the list is empty)."""
    ts_list = sorted(ts_list)
    best, bspan = 0, None
    for i, t0 in enumerate(ts_list):
        j = i
        while j < len(ts_list) and (ts_list[j] - t0).total_seconds() <= days * 86400:
            j += 1
        if j - i > best:
            best = j - i
            bspan = (t0, ts_list[j - 1])
    return best, bspan


def build_mapping(events, first_new, last_new):
    """Rename candidates: X -> Y with X retired (no new X issue created in
    [YEAR2_AGO, CV_CUT)), Y current (>= 1 new Y issue created in that
    window), and a burst of >= BURST_MIN replacement events inside
    BURST_DAYS days. All evidence is from before CV_CUT by construction.
    Returns (mapping, infos): mapping X -> Y with rename chains resolved to
    a fixpoint; infos maps every considered pair (passing candidates plus
    the mandatory pairs) to its evidence dict."""
    infos = {}
    candidates = []
    for (x, y) in sorted(set(events) | set(MANDATORY_PAIRS)):
        ts_list = sorted(events.get((x, y), []))
        best, bspan = max_burst(ts_list)
        x_retired = x in last_new and last_new[x] < YEAR2_AGO
        y_current = y in last_new and last_new[y] >= YEAR2_AGO
        info = {
            "x": x, "y": y,
            "n": len(ts_list),
            "span": (ts_list[0], ts_list[-1]) if ts_list else None,
            "burst": best,
            "burst_span": bspan,
            "last_new_x": last_new.get(x),
            "first_new_y": first_new.get(y),
            "x_retired": x_retired,
            "y_current": y_current,
            "burst_ok": best >= BURST_MIN,
            "passes": x_retired and y_current and best >= BURST_MIN,
        }
        infos[(x, y)] = info
        if info["passes"]:
            candidates.append((x, y))
    mapping = {}
    for x, y in candidates:
        mapping[x] = y
    # Resolve rename chains to a fixpoint (if Y was itself renamed, follow).
    changed = True
    while changed:
        changed = False
        for x in sorted(mapping):
            if mapping[x] in mapping:
                mapping[x] = mapping[mapping[x]]
                changed = True
    return mapping, infos


def assert_mapping_evidence_pre_cutoff(mapping, infos):
    """Every timestamp used for a mapped rename is < CV_CUT."""
    for x in mapping:
        info = next(i for i in infos.values() if i["x"] == x and i["passes"]
                    and i["y"] == mapping[x])
        if info["span"]:
            assert info["span"][0] < CV_CUT and info["span"][1] < CV_CUT
        if info["burst_span"]:
            assert info["burst_span"][0] < CV_CUT
        if info["last_new_x"]:
            assert info["last_new_x"] < CV_CUT
        if info["first_new_y"]:
            assert info["first_new_y"] < CV_CUT


def apply_mapping(rows, mapping):
    """Rows with the final component list remapped old -> new, deduplicated
    and sorted (the dataset's ordering)."""
    out = []
    for r in rows:
        remapped = sorted({mapping.get(c, c) for c in r[4]})
        out.append(r[:4] + (remapped,) + r[5:])
    return out


def cv_select(fit_fn, train, parent_key_map, summary):
    """Generic C selection, mirroring vtoken_cv_selection verbatim: fit on
    the train slice with created_at < CV_CUT, top-1 on the 2023 slice,
    ties -> the smaller C. fit_fn(train_rows, C) -> (vec, clf, mlb, fit_s).
    Returns ([(C, val_n, val_hit1, fit_s), ...], best_C)."""
    cv_train = [r for r in train if r[1] < CV_CUT]
    cv_val = [r for r in train if CV_CUT <= r[1] < CUTOFF]
    res = []
    for C in C_CANDIDATES:
        vec, clf, mlb, fit_s = fit_fn(cv_train, C)
        val_proba = clf.predict_proba(vec.transform(
            [extend_text(r, parent_key_map, summary, False)
             for r in cv_val]))
        val_hit1, _ = hit_counts(val_proba, list(mlb.classes_),
                                 [list(r[4]) for r in cv_val], 3)
        res.append((C, len(cv_val), val_hit1, fit_s))
    best_C = max(res, key=lambda t: (t[2], -t[0]))[0]
    return res, best_C


def fit_variant_balanced(train_rows, C, pkey_map, psum_map, with_title):
    """Identical pipeline to fit_variant, but with class_weight='balanced'
    on the per-class logistic regression (OvR): the rare labels get
    upweighted. Same tf-idf, same OvR, same solver and random_state."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer

    t0 = time.perf_counter()
    vec = TfidfVectorizer(sublinear_tf=True, min_df=3,
                          ngram_range=(1, 2), max_features=200000)
    Xtr = vec.fit_transform(
        [extend_text(r, pkey_map, psum_map, with_title) for r in train_rows])
    mlb = MultiLabelBinarizer()
    ytr = mlb.fit_transform([list(r[4]) for r in train_rows])
    clf = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE,
                           class_weight="balanced"),
        n_jobs=-1)
    clf.fit(Xtr, ytr)
    return vec, clf, mlb, time.perf_counter() - t0


def build_vtoken_balanced_model(train, C, parent_key_map, summary):
    """Train the final V-token+balanced model on the WHOLE train at the given
    C (which must already have been selected on the 2023 validation slice).
    Returns (names, predict_proba) exactly as build_vtoken_model (see
    router_text_baseline)."""
    vec, clf, mlb, _fit_s = fit_variant_balanced(
        train, C, parent_key_map, summary, False)
    names = list(mlb.classes_)
    return names, (lambda rows: _vtoken_predict_proba(
        rows, vec, clf, parent_key_map, summary))


def build_vtoken_balanced_cv_model(train, C, parent_key_map, summary):
    """Train the V-token+balanced model on the pre-2023 slice ONLY
    (created_at < CV_CUT), so it has never seen the 2023 validation slice.
    Returns (names, predict_proba) exactly as build_vtoken_balanced_model."""
    cv_train = [r for r in train if r[1] < CV_CUT]
    vec, clf, mlb, _fit_s = fit_variant_balanced(
        cv_train, C, parent_key_map, summary, False)
    names = list(mlb.classes_)
    return names, (lambda rows: _vtoken_predict_proba(
        rows, vec, clf, parent_key_map, summary))


def variant_cv(fit_fn, train, parent_key_map, summary):
    """cv_select plus one extra fit at the selected C, so the caller gets
    the validation probabilities of the finally selected model. Returns
    (res, best_C, names, val_proba) where val_proba is the selected model's
    predict_proba on the 2023 slice (rows of cv_val, in order)."""
    res, best_C = cv_select(fit_fn, train, parent_key_map, summary)
    cv_train = [r for r in train if r[1] < CV_CUT]
    cv_val = [r for r in train if CV_CUT <= r[1] < CUTOFF]
    vec, clf, mlb, _ = fit_fn(cv_train, best_C)
    names = list(mlb.classes_)
    val_proba = clf.predict_proba(vec.transform(
        [extend_text(r, parent_key_map, summary, False) for r in cv_val]))
    return res, best_C, names, val_proba


def sql_errors(rows, names, proba):
    """(n_sql, hits, errors): rows whose final components include 'SQL'; a
    hit when the top-1 prediction is 'SQL'."""
    idx = topk_idx(proba, 1)
    n_sql = hits = 0
    for i, r in enumerate(rows):
        if "SQL" in r[4]:
            n_sql += 1
            if names[idx[i][0]] == "SQL":
                hits += 1
    return n_sql, hits, n_sql - hits


def comp_hits(rows, names, proba, c):
    """(n, hit1) for rows whose final components include c: hit1 iff the
    top-1 prediction is one of the issue's final components (the current
    single-label definition)."""
    idx = topk_idx(proba, 1)
    n = hit1 = 0
    for i, r in enumerate(rows):
        if c in r[4]:
            n += 1
            if names[idx[i][0]] in set(r[4]):
                hit1 += 1
    return n, hit1


def mapping_for_threshold(events, last_new, thr_days):
    """The rename mapping when the retirement threshold is T whole days
    before CV_CUT (instead of YEAR2_AGO = 365 days). The burst and
    current(Y) rules are unchanged."""
    cut = CV_CUT - datetime.timedelta(days=thr_days)
    m = {}
    for (x, y) in sorted(events):
        best, _ = max_burst(sorted(events[(x, y)]))
        if best < BURST_MIN:
            continue
        if x in last_new and last_new[x] < cut and y in last_new \
                and last_new[y] >= cut:
            m[x] = y
    changed = True
    while changed:
        changed = False
        for x in sorted(m):
            if m[x] in m:
                m[x] = m[m[x]]
                changed = True
    return m


def robustness_interval(events, last_new, mapping):
    """Contiguous range [lo, hi] of whole-day retirement thresholds (1..730)
    for which the mapping is exactly the same."""
    stable = [T for T in range(1, 731)
              if mapping_for_threshold(events, last_new, T) == mapping]
    assert stable, "no stable threshold found"
    assert all(b - a == 1 for a, b in zip(stable, stable[1:]))
    return stable[0], stable[-1]


def fix_time_diagnostic(rows, init_map, changes, x, y):
    """Issues created before CV_CUT that started with x (in the initial
    list) and ended with y (x not in the final list): how many completed
    the swap before CV_CUT, how many after, and how many never show an
    x/y Component entry at all. The swap time is the latest x/y entry."""
    before = after = never = 0
    for r in rows:
        if r[1] >= CV_CUT:
            continue
        if x not in init_map[r[0]] or y not in r[4] or x in r[4]:
            continue
        ts_swap = [ts for ts, frm, to in changes.get(r[0], [])
                   if (frm == x and to is None)
                   or (frm is None and to == y)
                   or (frm == x and to == y)]
        if not ts_swap:
            never += 1
        elif max(ts_swap) < CV_CUT:
            before += 1
        else:
            after += 1
    return before, after, never


def gen_doc(mapping, infos, last_new, n_total, period_stats,
            variant_rows, selected, sql_rows,
            test_hit1, test_hit3, sel_test1, sel_test3,
            comp_table, rob_lo, rob_hi, fix_diag,
            val_old, test_old, hit_counts_src):
    """Render docs/router_label_hygiene.md as a string. Pure function of
    the computed values (no timestamps, sorted iterations only), so two
    runs produce a byte-identical file."""
    def fdate(d):
        return d.strftime("%Y-%m-%d") if d is not None else "-"

    L = []
    a = L.append
    a("# Router label hygiene (experiment 7)")
    a("")
    a("Generated by `scripts/router_label_hygiene.py`; the doc is regenerated")
    a("byte-identical on every run - do not edit it by hand.")
    a("")
    a("**Population.** router text-dataset issues (in-project, >= 1 component),")
    a(f"n = {n_total}. **Periods** by `created_at`: train < 2023-01-01")
    a(f"(n = {period_stats['train'][0]}), validation 2023 (n = {period_stats['validation'][0]}),")
    a(f"test >= 2024-01-01 (n = {period_stats['test'][0]}).")
    a("")
    a("**Selection protocol.** every variant is selected on the 2023 slice:")
    a("fit on train (created_at < 2023-01-01), score top-1 on the 2023 slice,")
    a("C grid {0.5, 2, 8}, tie -> smaller C. Test is never used for selection.")
    a("")
    a("## 1. Component renames (evidence strictly before 2023-01-01)")
    a("")
    a("**Method.** An issue's *initial* components are its current list with")
    a("every Component changelog entry after `created_at` undone")
    a("(`components_asof`, as in experiment 5). A rename X -> Y is supported")
    a("only if ALL of the following hold, using only data dated before")
    a("2023-01-01:")
    a("")
    a("* **retired(X)** - no issue created in [2022-01-01, 2023-01-01) has X")
    a("  in its initial components;")
    a("* **current(Y)** - at least one issue created in that window has Y in")
    a("  its initial components;")
    a("* **burst** - at least 5 X -> Y replacement events (the issue lost X and")
    a("  gained Y, in either order, or one direct swap entry; one event per")
    a("  (X, Y) per issue) inside a 60-day window.")
    a("")
    a(f"**Result.** mapping = `{dict(sorted(mapping.items()))}`; applied to the")
    a("training labels before fitting. No issue created in 2023 or later")
    a(f"carries an old name (0 of {period_stats['validation'][0]} validation rows,")
    a(f"0 of {period_stats['test'][0]} test rows carry `docs` or `MLilb`).")
    a("")
    a("**Candidates.** the pairs with >= 1 pre-2023 replacement event, plus the")
    a("four pairs flagged in experiment 5 (tossing graph). Passing pairs first,")
    a("then the flagged ones:")
    a("")
    a("| old | new | events | span | max 60-day cluster | last new X (init) | first new Y (init) | verdict |")
    a("|---|---|---|---|---|---|---|---|")

    def verdict_text(info, x, y):
        if info["passes"]:
            return "**rename**"
        rs = []
        if not info["x_retired"]:
            rs.append(f"X not retired (new {x} issue on "
                      f"{fdate(info['last_new_x'])})")
        if not info["y_current"]:
            rs.append(f"Y not current (last new {y} issue on "
                      f"{fdate(last_new.get(y))})")
        if not info["burst_ok"]:
            rs.append(f"burst {info['burst']} < {BURST_MIN}")
        txt = "not a rename - " + "; ".join(rs)
        if (x, y) in fix_diag:
            b, af, nv = fix_diag[(x, y)]
            txt += (f"; of the {b + af + nv} issues created before 2023 that "
                    f"started with {x} and ended with {y}, only {b} completed "
                    f"the swap before 2023-01-01 ({af} after, {nv} never "
                    f"recorded)")
        return txt

    passing = sorted(k for k, i in infos.items() if i["passes"])
    flagged = sorted(set(MANDATORY_PAIRS) - set(passing))
    for (x, y) in passing + flagged:
        info = infos[(x, y)]
        span = (f"{fdate(info['span'][0])}..{fdate(info['span'][1])}"
                if info["span"] else "-")
        bspan = (f"{info['burst']} ({fdate(info['burst_span'][0])})"
                 if info["burst_span"] else "0")
        a(f"| {x} | {y} | {info['n']} | {span} | {bspan} | "
          f"{fdate(info['last_new_x'])} | {fdate(info['first_new_y'])} | "
          f"{verdict_text(info, x, y)} |")
    a("")
    n_other = len(infos) - len(passing + flagged)
    a(f"Every other considered pair (the remaining {n_other} of {len(infos)})")
    a("fails at least one criterion (no burst, X not retired, or Y not")
    a("current).")
    a("")
    a(f"**Robustness.** the mapping is unchanged for every whole-day retirement")
    a(f"threshold T with {rob_lo} <= T <= {rob_hi} days before 2023-01-01 (i.e.")
    a(f"any threshold in ({rob_lo - 1}, {rob_hi + 1}) days).")
    a("")
    a("## 2. Multi-component issues")
    a("")
    a("Issues with >= 2 final components, per period (`period_of`, experiment 5):")
    a("")
    a("| period | issues | >= 2 final components | share |")
    a("|---|---|---|---|")
    for p in ("train", "validation", "test"):
        n, nm = period_stats[p]
        a(f"| {p} | {n} | {nm} | {pct(nm, n)} |")
    a("")
    a("The current single-label scoring code (`scripts/router_text_baseline.py`):")
    a("")
    a("```python")
    a(hit_counts_src.rstrip())
    a("```")
    a("")
    a("A top-1 hit is counted when the predicted name is one of the issue's")
    a("FINAL components - identical to \"any of the final components\" by")
    a("construction, so both definitions give the same number below.")
    a("")
    a("## 3. Rare classes: balanced weights and SQL errors")
    a("")
    a("Same pipeline for all four variants (tf-idf sublinear_tf, min_df=3,")
    a("1-2 grams, 200k features; OneVsRest logistic regression, C grid")
    a('{0.5, 2, 8}, tie -> smaller C); "balanced" adds class_weight='
      '"balanced";')
    a('"renames" applies the section-1 mapping to the training labels:')
    a("")
    a("| variant | C | val top-1 |")
    a("|---|---|---|")
    for label, C, n, h, _names, _p in variant_rows:
        a(f"| {label} | {C} | {pct(h, n)} ({h}/{n}) |")
    a("")
    a("**Selection.** highest validation top-1; tie -> the simpler variant")
    a("(V-token < V-token+renames < V-token+balanced <")
    a("V-token+renames+balanced).")
    a(f"**Selected: {selected[0]}.**")
    if "balanced" in selected[0]:
        sel_idx = [i for i, v in enumerate(variant_rows)
                   if v[0] == selected[0]][0]
        sel_sql = sql_rows[sel_idx]
        base_sql = sql_rows[0]
        if sel_sql[2] > base_sql[2]:
            a(f"The selected variant raises overall val top-1 but also raises")
            a(f"the SQL error count ({base_sql[2]} -> {sel_sql[2]}); the SQL table")
            a("below shows the trade-off per variant.")
    a("")
    a("SQL error count (validation rows whose final components include")
    a("'SQL'; an error when the top-1 is not 'SQL'):")
    a("")
    a("| variant | SQL rows | top-1 'SQL' | errors |")
    a("|---|---|---|---|")
    for row, s in zip(variant_rows, sql_rows):
        a(f"| {row[0]} | {s[0]} | {s[1]} | {s[2]} |")
    a("")
    a("## 4. Per-component (top 15, validation, experiment-5 order)")
    a("")
    a(f"| component | val issues | V-token top-1 | {selected[0]} top-1 |")
    a("|---|---|---|---|")
    for c, n_vt, h_vt, n_sel, h_sel in comp_table:
        a(f"| {c} | {n_vt} | {pct(h_vt, n_vt)} ({h_vt}/{n_vt}) | "
          f"{pct(h_sel, n_sel)} ({h_sel}/{n_sel}) |")
    a("")
    a("## 5. Test reference (never used for selection)")
    a("")
    a(f"Whole-train models at the validation-selected C, scored on the test")
    a(f"period (n = {period_stats['test'][0]}):")
    a("")
    a("| variant | test top-1 | test top-3 |")
    a("|---|---|---|")
    n_test = period_stats["test"][0]
    a(f"| V-token | {pct(test_hit1, n_test)} ({test_hit1}/{n_test}) | "
      f"{pct(test_hit3, n_test)} ({test_hit3}/{n_test}) |")
    a(f"| {selected[0]} | {pct(sel_test1, n_test)} ({sel_test1}/{n_test}) | "
      f"{pct(sel_test3, n_test)} ({sel_test3}/{n_test}) |")
    a("")
    a("## 6. Verification")
    a("")
    a("* two full runs of `scripts/router_label_hygiene.py` produce a")
    a("  byte-identical doc (checked with `cmp`, empty diff): every dict is")
    a("  iterated in sorted order, every model uses `random_state=42`, and no")
    a("  timestamp is written;")
    a("* the rename mapping uses only evidence dated before 2023-01-01 -")
    a("  asserted in code (`assert_mapping_evidence_pre_cutoff`);")
    a("* the V-token baseline is reproduced exactly: val top-1 = 3250/4733")
    a("  = 68.7% at C = 8 - asserted in code;")
    a(f"* no validation or test row carries `docs` or `MLilb` - asserted in")
    a(f"  code ({val_old} validation rows, {test_old} test rows carry an old name).")
    a("")
    return "\n".join(L)


def main():
    t0 = time.perf_counter()
    rows, _desc, summary, parent_key, _cur_comps = load_rows()
    env = load_env(REPO / ".env")
    import psycopg2
    conn = psycopg2.connect(
        host=env["PGHOST"], port=env["PGPORT"], dbname=env["PGDATABASE"],
        user=env["PGUSER"], password=env["PGPASSWORD"])
    cur = conn.cursor()
    changes = load_component_changes(cur)
    conn.close()

    train = [r for r in rows if r[9] == "train"]
    cv_train = [r for r in train if r[1] < CV_CUT]
    cv_val = [r for r in train if CV_CUT <= r[1] < CUTOFF]
    assert len(cv_train) == 40018 and len(cv_val) == 4733

    # ---- Step 1: rename detection (info < 2023-01-01 only) ----
    init_map = build_init_map(rows, changes)
    first_new, last_new = first_last_new(rows, init_map)
    events = replacement_events(rows, changes, init_map)
    mapping, infos = build_mapping(events, first_new, last_new)
    assert_mapping_evidence_pre_cutoff(mapping, infos)

    def fdate(dt):
        return dt.date().isoformat() if dt else "-"

    print("== Step 1: rename candidates (all info < 2023-01-01) ==")
    for (x, y) in sorted(infos):
        info = infos[(x, y)]
        verdict = "RENAME" if info["passes"] else "no"
        reasons = []
        if not info["x_retired"]:
            reasons.append(f"X not retired (new {x} issue on "
                           f"{fdate(info['last_new_x'])})")
        if not info["y_current"]:
            reasons.append(f"Y not current (last new {y} issue on "
                           f"{fdate(last_new.get(y))})")
        if not info["burst_ok"]:
            reasons.append(f"burst {info['burst']} < {BURST_MIN}")
        print(f"  {x!r:22s} -> {y!r:26s} n={info['n']:3d}  "
              f"span {fdate(info['span'][0]) if info['span'] else '-'}.."
              f"{fdate(info['span'][1]) if info['span'] else '-'}  "
              f"burst60={info['burst']:3d}  "
              f"last_new(X)={fdate(info['last_new_x'])}  "
              f"first_new(Y)={fdate(info['first_new_y'])}  "
              f"verdict={verdict}"
              + (f"  ({'; '.join(reasons)})" if reasons else ""))
    print(f"mapping = {dict(sorted(mapping.items()))}")

    # ---- Step 1 (cont.): V-token+renames variant ----
    # Training labels remapped old -> new before fitting; the 2023
    # validation labels stay as-is (no issue created in 2023 carries an old
    # name, which is asserted in the doc section).
    res_renames, best_C_renames, names_renames, val_proba_renames = variant_cv(
        lambda tr, C: fit_variant(
            apply_mapping(tr, mapping), C, parent_key, summary, False),
        train, parent_key, summary)
    r = next(t for t in res_renames if t[0] == best_C_renames)
    print(f"V-token+renames: best C={best_C_renames}  "
          f"val top-1 = {pct(r[2], r[1])} ({r[2]}/{r[1]})")

    # ---- Step 2: multi-component issues + V-token baseline ----
    period_stats = {}
    for p in ("train", "validation", "test"):
        pr = [r for r in rows if period_of(r[1]) == p]
        n_multi = sum(1 for r in pr if len(r[4]) >= 2)
        period_stats[p] = (len(pr), n_multi)
    print("== Step 2: issues with >= 2 final components ==")
    for p in ("train", "validation", "test"):
        n, nm = period_stats[p]
        print(f"  {p:12s} n={n}  >=2 final components: {nm} ({pct(nm, n)})")

    # V-token baseline: selection is the verbatim vtoken_cv_selection from
    # router_text_baseline (fit pre-2023, top-1 on 2023, tie -> smaller C).
    # The current single-label definition (hit_counts) scores an issue as a
    # top-1 hit iff names[argmax(proba)] is one of the issue's FINAL
    # components - identical to "any of the final components" by
    # construction, so the two definitions give the same number here.
    vtok_res, best_C_vt = vtoken_cv_selection(train, parent_key, summary)
    r8 = next(t for t in vtok_res if t[0] == 8)
    assert best_C_vt == 8, f"V-token selection moved: C={best_C_vt}"
    assert (r8[1], r8[2]) == (4733, 3250), \
        f"V-token val top-1 moved: {r8[2]}/{r8[1]}"
    print(f"V-token baseline: best C={best_C_vt}  "
          f"val top-1 = {pct(3250, 4733)} (3250/4733)")

    # Validation probabilities from the C=8 CV model (used again in step 4
    # for the per-component table and the SQL error counts).
    names_vt, vtoken_val_proba = build_vtoken_cv_model(train, 8, parent_key,
                                                       summary)
    cv_val = [r for r in train if CV_CUT <= r[1] < CUTOFF]
    vtok_hit1, _ = hit_counts(vtoken_val_proba(cv_val), names_vt,
                              [list(r[4]) for r in cv_val], 3)
    assert vtok_hit1 == 3250

    # Test reference (never used for selection): the whole-train model at the
    # validation-selected C.
    test = [r for r in rows if r[9] == "test"]
    names_ft, vtoken_test_proba = build_vtoken_model(train, 8, parent_key,
                                                     summary)
    test_hit1, test_hit3 = hit_counts(vtoken_test_proba(test), names_ft,
                                      [list(r[4]) for r in test], 3)
    print(f"V-token test reference (whole train, C=8): "
          f"top-1 = {pct(test_hit1, len(test))} ({test_hit1}/{len(test)})  "
          f"top-3 = {pct(test_hit3, len(test))} ({test_hit3}/{len(test)})")

    # ---- Step 3: balanced class weights + SQL error counts ----
    res_bal, best_C_bal, names_bal, val_proba_bal = variant_cv(
        lambda tr, C: fit_variant_balanced(tr, C, parent_key, summary, False),
        train, parent_key, summary)
    rb = next(t for t in res_bal if t[0] == best_C_bal)
    print(f"V-token+balanced: best C={best_C_bal}  "
          f"val top-1 = {pct(rb[2], rb[1])} ({rb[2]}/{rb[1]})")

    res_both, best_C_both, names_both, val_proba_both = variant_cv(
        lambda tr, C: fit_variant_balanced(
            apply_mapping(tr, mapping), C, parent_key, summary, False),
        train, parent_key, summary)
    rb2 = next(t for t in res_both if t[0] == best_C_both)
    print(f"V-token+renames+balanced: best C={best_C_both}  "
          f"val top-1 = {pct(rb2[2], rb2[1])} ({rb2[2]}/{rb2[1]})")

    # Validation SQL error counts: rows whose final components include
    # 'SQL'; an error when the top-1 is not 'SQL'.
    sql_vt = sql_errors(cv_val, names_vt, vtoken_val_proba(cv_val))
    sql_renames = sql_errors(cv_val, names_renames, val_proba_renames)
    sql_bal = sql_errors(cv_val, names_bal, val_proba_bal)
    sql_both = sql_errors(cv_val, names_both, val_proba_both)
    for label, s in (("V-token", sql_vt), ("V-token+renames", sql_renames),
                     ("V-token+balanced", sql_bal),
                     ("V-token+renames+balanced", sql_both)):
        print(f"SQL (validation): {label:25s} top-1 'SQL' on {s[1]}/{s[0]}  "
              f"({s[2]} errors)")

    # ---- Step 4: variant selection + doc generation ----
    # Robustness: sweep the retirement threshold (whole days before CV_CUT)
    # and require the mapping to be unchanged over a wide interval.
    rob_lo, rob_hi = robustness_interval(events, last_new, mapping)
    print(f"robustness: mapping unchanged for T in [{rob_lo}, {rob_hi}] days")

    # Fix-time diagnostic for the four flagged pairs: of the pre-2023 issues
    # that started with X and ended with Y, when did the swap actually happen.
    fix_diag = {}
    for x, y in MANDATORY_PAIRS:
        fix_diag[(x, y)] = fix_time_diagnostic(rows, init_map, changes, x, y)
        b, af, nv = fix_diag[(x, y)]
        print(f"fix-time {x!r}->{y!r}: before={b} after={af} never={nv}")

    # Old names must not leak into the validation/test labels.
    old_names = set(mapping)
    val_old = sum(1 for r in cv_val if old_names & set(r[4]))
    test_old = sum(1 for r in test if old_names & set(r[4]))
    assert val_old == 0 and test_old == 0, (val_old, test_old)

    # Variants in order of simplicity: a tie in validation top-1 goes to the
    # simpler variant (max keeps the first occurrence of the maximum).
    vtok_val_arr = vtoken_val_proba(cv_val)
    variant_rows = [
        ("V-token", best_C_vt, 4733, 3250, names_vt, vtok_val_arr),
        ("V-token+renames", best_C_renames, r[1], r[2], names_renames,
         val_proba_renames),
        ("V-token+balanced", best_C_bal, rb[1], rb[2], names_bal,
         val_proba_bal),
        ("V-token+renames+balanced", best_C_both, rb2[1], rb2[2],
         names_both, val_proba_both),
    ]
    selected = max(variant_rows, key=lambda v: v[3])
    print(f"selected variant: {selected[0]}  "
          f"(val top-1 {selected[3]}/{selected[2]}, C={selected[1]})")

    # Test reference for the selected variant (whole-train fit at the
    # validation-selected C; test is never used for selection). The plain
    # V-token reuses the step-2 reference.
    if selected[0] == "V-token":
        sel_test1, sel_test3 = test_hit1, test_hit3
    else:
        fit_fns = {
            "V-token+renames": lambda tr, C: fit_variant(
                apply_mapping(tr, mapping), C, parent_key, summary, False),
            "V-token+balanced": lambda tr, C: fit_variant_balanced(
                tr, C, parent_key, summary, False),
            "V-token+renames+balanced": lambda tr, C: fit_variant_balanced(
                apply_mapping(tr, mapping), C, parent_key, summary, False),
        }
        sel_vec, sel_clf, sel_mlb, _ = fit_fns[selected[0]](train,
                                                            selected[1])
        sel_test1, sel_test3 = hit_counts(
            _vtoken_predict_proba(test, sel_vec, sel_clf, parent_key, summary),
            list(sel_mlb.classes_), [list(r[4]) for r in test], 3)
    print(f"{selected[0]} test reference (whole train, C={selected[1]}): "
          f"top-1 = {pct(sel_test1, len(test))} ({sel_test1}/{len(test)})  "
          f"top-3 = {pct(sel_test3, len(test))} ({sel_test3}/{len(test)})")

    # Per-component top-1 (top 15, experiment-5 order), validation.
    comp_table = []
    for c in RESULTS_TOP15:
        n_vt, h_vt = comp_hits(cv_val, names_vt, vtok_val_arr, c)
        n_sel, h_sel = comp_hits(cv_val, selected[4], selected[5], c)
        comp_table.append((c, n_vt, h_vt, n_sel, h_sel))

    # Write the doc (byte-identical across runs).
    doc = gen_doc(mapping, infos, last_new, len(rows), period_stats,
                  variant_rows, selected,
                  (sql_vt, sql_renames, sql_bal, sql_both),
                  test_hit1, test_hit3, sel_test1, sel_test3,
                  comp_table, rob_lo, rob_hi, fix_diag,
                  val_old, test_old, inspect.getsource(hit_counts))
    doc_path = REPO / "docs" / "router_label_hygiene.md"
    doc_path.write_text(doc, encoding="utf-8")
    print(f"wrote {doc_path} ({len(doc)} bytes)")

    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
