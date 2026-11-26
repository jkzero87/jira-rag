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

Parent-context variants for sub-tasks (all use the parent's components AS OF
the sub-task's created_at — router.text_dataset.parent_components_asof —
never the parent's current components, which would leak future changes):
  V-rule   for sub-tasks, predict the parent's as-of components ordered by
           train frequency; sub-tasks whose parent had no components then,
           and all non-sub-tasks, fall back to level 1.
  V-token  level 1 (same tf-idf + LR) with the parent's as-of component
           tokens appended to the sub-task's text, e.g. __parentcomp_SQL
           (spaces and dots replaced by _).
  V-title  V-token plus the parent's summary (bronze) appended as well.
  V-blend  p = (1-w) * p_model + w * p_rule: p_model is the V-token
           probability over all components; p_rule weights the parent's
           as-of components by the same <2023 train frequencies V-rule's
           validation rule uses, normalized to sum 1 (uniform when none
           of them appeared in train < 2023). Rows whose parent had no
           as-of components have no p_rule and keep the V-token
           prediction for every w. w in {0.0, 0.1, ..., 1.0} is chosen on
           the 2023 slice (top-1; ties -> smallest w); test is never
           used for selection.
  Each model variant chooses C in {0.5, 2, 8} with the level-1 procedure
  (fit on train < 2023-01-01, top-1 on the 2023 slice); test is never used
  for selection.

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
PASOF_IDX = 10  # index of parent_components_asof in a dataset row

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
        " has_exception_name, split, parent_components_asof"
        " FROM router.text_dataset"
        " WHERE split IN ('train', 'test') ORDER BY issue_key")
    rows = [tuple(r) for r in cur.fetchall()]
    # NOTE: parent_components_now is deliberately NOT loaded here; it is only
    # used by the dataset script's report (step 1). The variants below use
    # parent_components_asof exclusively.
    cur.execute(
        "SELECT issue_key, payload->'fields'->>'description',"
        " COALESCE(payload->'fields'->>'summary', '')"
        " FROM bronze.jira_issue_raw")
    desc, summary = {}, {}
    for k, d, s in cur.fetchall():
        desc[k] = d
        summary[k] = s
    cur.execute(
        "SELECT issue_key, parent_key FROM silver.issue"
        " WHERE parent_key IS NOT NULL")
    parent_key = dict(cur.fetchall())
    # Parent's CURRENT components (silver) — only for the step-1 report
    # number (now vs asof), never fed to any variant.
    cur.execute(
        "SELECT issue_key, array_agg(component ORDER BY component)"
        " FROM silver.issue_component GROUP BY issue_key")
    cur_comps = dict(cur.fetchall())
    conn.close()
    return rows, desc, summary, parent_key, cur_comps


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


def parent_tokens(pasof):
    """One token per parent as-of component, e.g. 'SQL' -> __parentcomp_SQL
    (spaces and dots replaced by _). The default tf-idf tokenizer keeps each
    as a single token (underscore is a word character)."""
    return [f"__parentcomp_{c.replace(' ', '_').replace('.', '_')}"
            for c in (pasof or [])]


def extend_text(row, pkey_map, psum_map, with_title):
    """Row text plus the sub-task's parent as-of component tokens (and, for
    with_title, the parent's summary). Non-sub-tasks (pasof is null) are
    returned unchanged."""
    text = row[3]
    parts = []
    pasof = row[PASOF_IDX]
    if pasof:
        parts.append(" ".join(parent_tokens(pasof)))
    if with_title:
        pk = pkey_map.get(row[0])
        psum = psum_map.get(pk, "") if pk else ""
        if psum:
            parts.append(psum)
    if parts:
        text = text + "\n" + "\n".join(parts)
    return text


def fit_variant(train_rows, C, pkey_map, psum_map, with_title):
    """Level 1 (same tf-idf + LR) over parent-context-extended texts."""
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
        LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE),
        n_jobs=-1)
    clf.fit(Xtr, ytr)
    return vec, clf, mlb, time.perf_counter() - t0


def main():
    rows, desc, summary, parent_key_map, cur_comps = load_rows()
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
    cv_models = {}
    for C in C_CANDIDATES:
        vec, clf, mlb, fit_s = fit_level1(cv_train, C)
        cv_models[C] = (vec, clf, mlb)
        names = list(mlb.classes_)
        val_proba = clf.predict_proba(vec.transform([r[3] for r in cv_val]))
        val_n = len(cv_val)
        val_hit1, _ = hit_counts(val_proba, names,
                                 [list(r[4]) for r in cv_val], 3)
        val_results.append((C, val_n, val_hit1, fit_s))
    best = max(val_results, key=lambda t: (t[2], -t[0]))  # most hits; ties -> smaller C
    best_C = best[0]

    # ---- V-rule on the 2023 validation slice (train < 2023 only) -------------
    # Both the frequency order and the level-1 fallback come from train
    # < 2023-01-01, so no 2023/test information enters the V-rule score.
    cv_vec, cv_clf, cv_mlb = cv_models[best_C]
    cv_names = list(cv_mlb.classes_)
    cv_val_proba_l1 = cv_clf.predict_proba(
        cv_vec.transform([r[3] for r in cv_val]))
    cv_comp_counts = Counter()
    for r in cv_train:
        cv_comp_counts.update(r[4])
    cv_freq_rank = {c: i for i, (c, _n) in enumerate(
        sorted(cv_comp_counts.items(), key=lambda kv: (-kv[1], kv[0])))}

    def vrule_val_top1(i, r):
        pasof = r[PASOF_IDX]
        if pasof:
            return sorted(pasof,
                          key=lambda c: (cv_freq_rank.get(c,
                                                         len(cv_freq_rank)), c))[0]
        return cv_names[int(np.argmax(cv_val_proba_l1[i]))]

    vrule_val_hits1 = sum(1 for i, r in enumerate(cv_val)
                          if vrule_val_top1(i, r) in list(r[4]))

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

    # ---- parent context (sub-tasks) -------------------------------------------
    st_train = [r for r in train if r[PASOF_IDX] is not None]
    st_test = [r for r in test if r[PASOF_IDX] is not None]
    nonst_test = [r for r in test if r[PASOF_IDX] is None]
    missing_parents = sum(
        1 for r in train + test
        if parent_key_map.get(r[0]) and r[PASOF_IDX] is None)

    def changed_stats(rs):
        """(changed, n, pct) — parent's CURRENT components (from silver,
        the same source the dataset used for parent_components_now) differ
        from the dataset's as-of list. Never used by any variant below."""
        ch = 0
        for r in rs:
            pk = parent_key_map.get(r[0])
            now = cur_comps.get(pk)
            if now is not None and set(now) != set(r[PASOF_IDX]):
                ch += 1
        return ch, len(rs), pct(ch, len(rs)) if rs else "0.0%"

    ch_tr, n_st_tr, pct_tr = changed_stats(st_train)
    ch_te, n_st_te, pct_te = changed_stats(st_test)
    ch_all = ch_tr + ch_te
    n_all_st = n_st_tr + n_st_te
    pct_all = pct(ch_all, n_all_st) if n_all_st else "0.0%"

    # ---- V-rule: parent's as-of components, ordered by train frequency -------
    freq_rank = {c: i for i, (c, _n) in enumerate(ranked)}

    def vrule_pred(r):
        """Ordered component list for a sub-task; None -> level 1 fallback."""
        pasof = r[PASOF_IDX]
        if not pasof:
            return None
        return sorted(pasof, key=lambda c: (freq_rank.get(c, len(freq_rank)), c))

    def vrule_top1(r):
        p = vrule_pred(r)
        if p is None:
            return names[int(np.argmax(proba_by_key[r[0]]))]
        return p[0]

    def vrule_top3(r):
        p = vrule_pred(r)
        if p is None:
            idx = np.argsort(-proba_by_key[r[0]], kind="stable")[:3]
            return [names[i] for i in idx]
        return p[:3]

    def vrule_hits(rs):
        hit1 = sum(1 for r in rs if vrule_top1(r) in r[4])
        hit3 = sum(1 for r in rs if any(c in r[4] for c in vrule_top3(r)))
        return len(rs), hit1, hit3

    # ---- V-token / V-token+title: C selection, then final model ---------------
    def variant_cv(with_title):
        res = []
        for C in C_CANDIDATES:
            vec, clf, mlb, fit_s = fit_variant(
                cv_train, C, parent_key_map, summary, with_title)
            val_proba = clf.predict_proba(vec.transform(
                [extend_text(r, parent_key_map, summary, with_title)
                 for r in cv_val]))
            val_hit1, _ = hit_counts(val_proba, list(mlb.classes_),
                                     [list(r[4]) for r in cv_val], 3)
            res.append((C, len(cv_val), val_hit1, fit_s))
        return res

    vtok_val = variant_cv(False)
    vtok_best_C = max(vtok_val, key=lambda t: (t[2], -t[0]))[0]
    vtit_val = variant_cv(True)
    vtit_best_C = max(vtit_val, key=lambda t: (t[2], -t[0]))[0]

    # ---- V-blend on the 2023 validation slice (w chosen here, never test) --
    # p = (1-w) * p_model + w * p_rule. p_model: V-token probabilities over
    # all components (model fitted on train < 2023-01-01, C=vtok_best_C).
    # p_rule: the parent's as-of components weighted by the same <2023
    # train frequencies V-rule's validation rule uses (cv_comp_counts),
    # normalized to sum 1; uniform when none of them appeared in train
    # < 2023. Rows whose parent had no as-of components have no p_rule and
    # keep the V-token prediction for every w.
    vtok_cv_vec, vtok_cv_clf, vtok_cv_mlb, _fit_s = fit_variant(
        cv_train, vtok_best_C, parent_key_map, summary, False)
    vtok_cv_names = list(vtok_cv_mlb.classes_)
    vtok_cv_names_set = set(vtok_cv_names)
    vtok_cv_val_proba = vtok_cv_clf.predict_proba(vtok_cv_vec.transform(
        [extend_text(r, parent_key_map, summary, False) for r in cv_val]))

    def rule_weights(pasof):
        """Normalized distribution over the parent's as-of components,
        proportional to their <2023 train frequency (the frequencies
        V-rule's validation rule uses); None when there is no parent
        context, uniform when none of them appeared in train < 2023."""
        if not pasof:
            return None
        weights = [cv_comp_counts.get(c, 0) for c in pasof]
        total = sum(weights)
        if total > 0:
            return {c: wt / total for c, wt in zip(pasof, weights)}
        return {c: 1.0 / len(pasof) for c in pasof}

    def blend_rank(w, proba_row, names_row, names_set_row, rw):
        """(p, tie_key, component) of (1-w)*p_model + w*p_rule over the
        model vocabulary plus the parent's as-of components missing from
        it, in descending p. Ties keep the stable, index-ordered sort
        (and, among the out-of-vocabulary parent components, name order),
        so the top-1 is deterministic. With rw None it is the plain model
        ranking, so the top-1 equals the V-token prediction for every w."""
        items = []
        for j, (c, p) in enumerate(zip(names_row, proba_row)):
            pr = rw.get(c) if rw is not None else None
            items.append((float(p) * (1.0 - w)
                          + (w * pr if pr is not None else 0.0),
                          j, c))
        if rw is not None:
            for c in sorted(rw):
                if c not in names_set_row:
                    items.append((w * rw[c], len(names_row), c))
        items.sort(key=lambda t: (-t[0], t[1]))
        return items

    def blend_topk(w, proba_row, names_row, names_set_row, rw, k):
        return [c for _p, _j, c in
                blend_rank(w, proba_row, names_row, names_set_row, rw)[:k]]

    W_GRID = [i / 10 for i in range(11)]
    val_rule_weights = [rule_weights(r[PASOF_IDX]) for r in cv_val]
    blend_val_hits = {}
    for w in W_GRID:
        preds = [blend_topk(w, vtok_cv_val_proba[i], vtok_cv_names,
                            vtok_cv_names_set, rw, 1)[0]
                 for i, rw in enumerate(val_rule_weights)]
        blend_val_hits[w] = sum(
            1 for r, p in zip(cv_val, preds) if p in list(r[4]))
    # ties -> the smallest w
    best_w = max(W_GRID, key=lambda w: (blend_val_hits[w], -w))
    # w=0 must reproduce the V-token validation score exactly (same model,
    # same argmax convention).
    vtok_cv_best_hits = vtok_val[C_CANDIDATES.index(vtok_best_C)][2]
    assert blend_val_hits[0.0] == vtok_cv_best_hits, \
        (f"V-blend w=0 differs from V-token on validation: "
         f"{blend_val_hits[0.0]} vs {vtok_cv_best_hits}")

    # sub-task / non-sub-task validation slices for V-token and V-blend
    cv_st_idx = [i for i, r in enumerate(cv_val) if r[PASOF_IDX]]
    cv_nonst_idx = [i for i, r in enumerate(cv_val) if not r[PASOF_IDX]]
    vtok_cv_preds = [vtok_cv_names[int(np.argmax(row))]
                     for row in vtok_cv_val_proba]
    blend_cv_preds = [blend_topk(best_w, vtok_cv_val_proba[i], vtok_cv_names,
                                 vtok_cv_names_set, rw, 1)[0]
                      for i, rw in enumerate(val_rule_weights)]
    # rows without a parent: V-blend must be IDENTICAL to V-token
    assert all(vtok_cv_preds[i] == blend_cv_preds[i] for i in cv_nonst_idx), \
        "V-blend differs from V-token on validation rows without a parent"
    vtok_val_st_hits = sum(
        1 for i in cv_st_idx if vtok_cv_preds[i] in list(cv_val[i][4]))
    vtok_val_nonst_hits = sum(
        1 for i in cv_nonst_idx if vtok_cv_preds[i] in list(cv_val[i][4]))
    blend_val_st_hits = sum(
        1 for i in cv_st_idx if blend_cv_preds[i] in list(cv_val[i][4]))
    blend_val_nonst_hits = sum(
        1 for i in cv_nonst_idx if blend_cv_preds[i] in list(cv_val[i][4]))

    # ---- variant selection: top-1 on the 2023 slice ONLY, never test -------
    # Fixed order below; the highest validation top-1 wins, ties -> the
    # earlier (simpler) variant. Test numbers are reference only.
    vtok_val_best = vtok_val[C_CANDIDATES.index(vtok_best_C)]
    vtit_val_best = vtit_val[C_CANDIDATES.index(vtit_best_C)]
    var_val = (
        ("level 1", best[2], best[1]),
        ("V-rule", vrule_val_hits1, len(cv_val)),
        ("V-token", vtok_val_best[2], vtok_val_best[1]),
        ("V-token+title", vtit_val_best[2], vtit_val_best[1]),
        ("V-blend", blend_val_hits[best_w], len(cv_val)))
    selected_variant = None
    selected_hits = -1
    for vname, h, _n in var_val:
        if h > selected_hits:
            selected_variant, selected_hits = vname, h

    def variant_final(with_title, C):
        vec, clf, mlb, _fit_s = fit_variant(
            train, C, parent_key_map, summary, with_title)
        names_v = list(mlb.classes_)
        proba_v = clf.predict_proba(vec.transform(
            [extend_text(r, parent_key_map, summary, with_title)
             for r in test]))
        proba_by_key_v = {r[0]: proba_v[i] for i, r in enumerate(test)}
        return names_v, proba_by_key_v

    vtok_names, vtok_proba = variant_final(False, vtok_best_C)
    vtit_names, vtit_proba = variant_final(True, vtit_best_C)

    # ---- V-blend on the whole test (reference only, never for selection) ---
    # Same rule as on validation: p_model = final V-token model, p_rule =
    # parent's as-of components with the <2023 train frequencies
    # (cv_comp_counts); rows without a parent keep the V-token prediction.
    vtok_names_set = set(vtok_names)
    vblend_pred1 = {}
    vblend_pred3 = {}
    for r in test:
        rw = rule_weights(r[PASOF_IDX])
        vblend_pred1[r[0]] = blend_topk(
            best_w, vtok_proba[r[0]], vtok_names, vtok_names_set, rw, 1)[0]
        vblend_pred3[r[0]] = blend_topk(
            best_w, vtok_proba[r[0]], vtok_names, vtok_names_set, rw, 3)

    def vblend_hits(rs):
        hit1 = sum(1 for r in rs if vblend_pred1[r[0]] in r[4])
        hit3 = sum(1 for r in rs
                   if any(c in r[4] for c in vblend_pred3[r[0]]))
        return len(rs), hit1, hit3

    vblend_all = vblend_hits(test)
    vblend_st = vblend_hits(st_test)
    vblend_nonst = vblend_hits(nonst_test)
    # rows without a parent: V-blend must be IDENTICAL to V-token
    assert all(
        vblend_pred1[r[0]] == vtok_names[int(np.argmax(vtok_proba[r[0]]))]
        for r in nonst_test), \
        "V-blend differs from V-token on test rows without a parent"

    # ---- evaluation ----------------------------------------------------------
    n_test = len(test)
    l0_all = l0_hits(test)
    l1_all = group_hits(test, names, proba_by_key, 3)
    l1_jt = group_hits(test_jt, names, proba_by_key, 3)
    l1_st = group_hits(st_test, names, proba_by_key, 3)
    l1_nonst = group_hits(nonst_test, names, proba_by_key, 3)
    pkg_all = pkg_hits(test_jt)

    vrule_all = vrule_hits(test)
    vrule_st = vrule_hits(st_test)
    vrule_nonst = vrule_hits(nonst_test)
    vtok_all = group_hits(test, vtok_names, vtok_proba, 3)
    vtok_st = group_hits(st_test, vtok_names, vtok_proba, 3)
    vtok_nonst = group_hits(nonst_test, vtok_names, vtok_proba, 3)
    vtit_all = group_hits(test, vtit_names, vtit_proba, 3)
    vtit_st = group_hits(st_test, vtit_names, vtit_proba, 3)
    vtit_nonst = group_hits(nonst_test, vtit_names, vtit_proba, 3)

    # V-rule on non-sub-tasks must be IDENTICAL to level 1 (fallback path).
    assert l1_nonst == vrule_nonst, \
        f"V-rule non-sub-tasks differ from level 1: {l1_nonst} vs {vrule_nonst}"

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

    # top-1 per top-15 component, per variant; no per-row best is chosen —
    # the selected variant (from the 2023 validation slice) is the same for
    # every row
    def variant_top1_hits(rs, proba_by_key_v, names_v):
        return sum(1 for r in rs
                   if names_v[int(np.argmax(proba_by_key_v[r[0]]))] in r[4])

    comp_var_rows = []
    for c, n in top15_comps:
        rs = [r for r in test if c in r[4]]
        comp_var_rows.append((
            c, n,
            variant_top1_hits(rs, proba_by_key, names),
            sum(1 for r in rs if vrule_top1(r) in r[4]),
            variant_top1_hits(rs, vtok_proba, vtok_names),
            variant_top1_hits(rs, vtit_proba, vtit_names),
            sum(1 for r in rs if vblend_pred1[r[0]] in r[4])))

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
    md.append("## Parent context for sub-tasks")
    md.append("")
    md.append("For sub-tasks the parent issue's components are taken AS OF"
              " the sub-task's created_at (the parent's Component changelog"
              " undone from the current state), never the parent's current"
              " components, which would leak future changes."
              " `router.text_dataset.parent_components_now` is read only by"
              " the dataset script's report (step 1), never by any variant."
              " The step-1 change rate below is recomputed here from"
              " `silver.issue_component` (same source) against the dataset's"
              " as-of column.")
    md.append("")
    md.append("Caveat: the captured changelog does not record every"
              " component addition (e.g. initial values at creation — many"
              " issues have a current component with no 'add' entry). Such"
              " components are treated as having existed since before the"
              " latest recorded change; the undo still never uses information"
              " from after the sub-task's created_at.")
    md.append("")
    md.append(f"- sub-tasks in dataset: train {n_st_tr}, test {n_st_te};"
              f" parents missing from bronze: {missing_parents}")
    md.append(f"- parent components changed after the sub-task was created"
              f" (now != asof): train {ch_tr} of {n_st_tr} ({pct_tr}),"
              f" test {ch_te} of {n_st_te} ({pct_te}),"
              f" total {ch_all} of {n_all_st} ({pct_all})")
    md.append("")
    # (a) V-blend weight grid on the 2023 validation slice
    md.append("### V-blend weight w selection on the 2023 validation slice"
              f" (n={len(cv_val)})")
    md.append("")
    md.append(f"p = (1-w) * p_model + w * p_rule, where p_model is the"
              f" V-token (C={vtok_best_C}) model fitted on train <"
              " 2023-01-01 and p_rule weights the parent's as-of"
              " components by the same <2023 train frequencies V-rule's"
              " validation rule uses, normalized to sum 1 (uniform when"
              " none of them appeared in train < 2023). Rows whose parent"
              " had no as-of components have no p_rule and keep the V-token"
              " prediction for every w. w is chosen on the 2023 slice"
              " only (top-1; ties -> smallest w); test is never used.")
    md.append("")
    md.append("| w | val top-1 | n |")
    md.append("|---:|------:|---:|")
    for w in W_GRID:
        mark = "  <-- chosen" if w == best_w else ""
        md.append(f"| {w:.1f} | {pct(blend_val_hits[w], len(cv_val))}"
                  f" | {len(cv_val)}{mark} |")
    md.append("")
    md.append(f"chosen w = {best_w:.1f} (highest top-1 on the 2023 slice;"
              " ties -> smallest w; test never used)")
    md.append("")
    # (b) variant selection on validation, now five variants
    var_val_disp = (
        (f"level 1 (C={best_C})", best[2], best[1]),
        ("V-rule (parent as-of components, <2023 train frequency;"
         " <2023 level-1 fallback)", vrule_val_hits1, len(cv_val)),
        (f"V-token (C={vtok_best_C})", vtok_val_best[2], vtok_val_best[1]),
        (f"V-token+title (C={vtit_best_C})", vtit_val_best[2],
         vtit_val_best[1]),
        (f"V-blend (w={best_w:.1f}; V-token + <2023 parent-frequency"
         " rule)", blend_val_hits[best_w], len(cv_val)))
    md.append("### Variant selection on the 2023 validation slice (fit on"
              " train < 2023-01-01, top-1 on the 2023 slice; V-rule uses the"
              " <2023 train component frequency and the <2023 level-1 model"
              " as fallback; V-blend blends the V-token model with the"
              " <2023 parent-frequency rule at the w chosen above)")
    md.append("")
    md.append("| variant | val top-1 | n |")
    md.append("|---|---:|---:|")
    for vname, h, n in var_val_disp:
        md.append(f"| {vname} | {pct(h, n)} | {n} |")
    md.append("")
    md.append(f"Selected variant (by 2023 validation): {selected_variant}"
              " (highest top-1; ties -> the earlier/simpler variant)")
    md.append("")
    # (e) validation top-1 on sub-tasks / non-sub-tasks for V-token & V-blend
    md.append("### V-token vs V-blend on sub-tasks / non-sub-tasks (2023"
              " validation slice, top-1)")
    md.append("")
    md.append("| variant | sub-tasks n | sub-tasks top-1"
              " | non-sub-tasks n | non-sub-tasks top-1 |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| V-token (C={vtok_best_C}) | {len(cv_st_idx)}"
              f" | {pct(vtok_val_st_hits, len(cv_st_idx))}"
              f" | {len(cv_nonst_idx)}"
              f" | {pct(vtok_val_nonst_hits, len(cv_nonst_idx))} |")
    md.append(f"| V-blend (w={best_w:.1f}) | {len(cv_st_idx)}"
              f" | {pct(blend_val_st_hits, len(cv_st_idx))}"
              f" | {len(cv_nonst_idx)}"
              f" | {pct(blend_val_nonst_hits, len(cv_nonst_idx))} |")
    md.append("")
    # (c) test reference: selected variant and V-token only
    test_variant_rows = {
        "level 1": (l1_st, l1_nonst, l1_all),
        "V-rule": (vrule_st, vrule_nonst, vrule_all),
        "V-token": (vtok_st, vtok_nonst, vtok_all),
        "V-token+title": (vtit_st, vtit_nonst, vtit_all),
        "V-blend": (vblend_st, vblend_nonst, vblend_all)}
    test_variant_labels = {
        "level 1": "level 1",
        "V-rule": ("V-rule (parent as-of components, ordered by train"
                   " frequency; level 1 fallback when the parent had none)"),
        "V-token": (f"V-token (level 1 + `__parentcomp_*` tokens,"
                    f" C={vtok_best_C})"),
        "V-token+title": (f"V-token+title (V-token + parent summary,"
                          f" C={vtit_best_C})"),
        "V-blend": (f"V-blend (w={best_w:.1f}; V-token + <2023"
                    " parent-frequency rule)")}
    ref_order = [selected_variant]
    if "V-token" not in ref_order:
        ref_order.append("V-token")
    md.append("### Variants (measured on test only — reference: selected"
              " variant and V-token)")
    md.append("")
    md.append("| method | sub-tasks n | sub-tasks top-1 | sub-tasks top-3"
              " | non-sub-tasks n | non-sub-tasks top-1 | non-sub-tasks top-3"
              " | whole test n | whole test top-1 | whole test top-3 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for vname in ref_order:
        stg, nonst, allg = test_variant_rows[vname]
        mark = " <- selected" if vname == selected_variant else ""
        md.append(f"| {test_variant_labels[vname]}{mark} | {stg[0]}"
                  f" | {pct(stg[1], stg[0])} | {pct(stg[2], stg[0])}"
                  f" | {nonst[0]} | {pct(nonst[1], nonst[0])}"
                  f" | {pct(nonst[2], nonst[0])} | {allg[0]}"
                  f" | {pct(allg[1], allg[0])} | {pct(allg[2], allg[0])} |")
    md.append("test numbers are reference only; selection used the 2023"
              " validation slice")
    md.append("")
    md.append("### C selection for the model variants (same procedure as"
              " level 1: fit on train < 2023-01-01, top-1 on the 2023 slice)")
    md.append("")
    md.append("| C | V-token top-1 | V-token+title top-1 | n |")
    md.append("|---:|---:|---:|---:|")
    for i, C in enumerate(C_CANDIDATES):
        _t, n, h1, _s1 = vtok_val[i]
        _u, _n2, h2, _s2 = vtit_val[i]
        md.append(f"| {C} | {pct(h1, n)} | {pct(h2, n)} | {n} |")
    md.append("")
    md.append(f"chosen: V-token C={vtok_best_C}, V-token+title C={vtit_best_C}"
              " (most top-1 on the 2023 slice; ties -> smaller C)")
    md.append("")
    var_headers = ["level 1", "V-rule"]
    for v in ("V-token", "V-token+title", "V-blend"):
        var_headers.append(v + (" (selected)" if v == selected_variant else ""))
    md.append("### Top 15 components by test frequency: top-1 per variant"
              " (the selected variant is the same for every row; no per-row"
              " best is chosen)")
    md.append("")
    md.append("| component | n | " + " | ".join(var_headers) + " |")
    md.append("|---|---:|" + "---:|" * len(var_headers))
    for c, n, h_l1, h_vr, h_vt, h_vtt, h_vb in comp_var_rows:
        md.append(f"| {c} | {n} | {pct(h_l1, n)} | {pct(h_vr, n)}"
                  f" | {pct(h_vt, n)} | {pct(h_vtt, n)} | {pct(h_vb, n)} |")
    md.append("")
    md.append("V-rule on non-sub-tasks uses the level 1 fallback, so its"
              " non-sub-task hit counts are identical to level 1 (asserted"
              " by the script).")
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
    print("parent context (sub-tasks):")
    print(f"  sub-tasks in dataset: train {n_st_tr}, test {n_st_te};"
          f" parents missing from bronze: {missing_parents}")
    print(f"  parent components changed after creation (now != asof):"
          f" train {ch_tr} of {n_st_tr} ({pct_tr}), test {ch_te} of {n_st_te}"
          f" ({pct_te}), total {ch_all} of {n_all_st} ({pct_all})")
    print("  " + "  ".join(f"C={C}: val top-1 {pct(h, n)}"
                           for C, n, h, _s in vtok_val)
          + f"  -> V-token C={vtok_best_C}")
    print("  " + "  ".join(f"C={C}: val top-1 {pct(h, n)}"
                           for C, n, h, _s in vtit_val)
          + f"  -> V-token+title C={vtit_best_C}")
    print(f"V-blend weight selection on the 2023 slice (n={len(cv_val)}):")
    for w in W_GRID:
        mark = "  <-- chosen" if w == best_w else ""
        print(f"  w={w:.1f}: val top-1 {pct(blend_val_hits[w], len(cv_val))}{mark}")
    print(f"  validation sub-tasks / non-sub-tasks top-1 (V-token, V-blend):")
    print(f"    V-token: sub-tasks {pct(vtok_val_st_hits, len(cv_st_idx))}"
          f" (n={len(cv_st_idx)}) | non-sub-tasks"
          f" {pct(vtok_val_nonst_hits, len(cv_nonst_idx))} (n={len(cv_nonst_idx)})")
    print(f"    V-blend (w={best_w:.1f}): sub-tasks"
          f" {pct(blend_val_st_hits, len(cv_st_idx))} | non-sub-tasks"
          f" {pct(blend_val_nonst_hits, len(cv_nonst_idx))}")
    print(f"variant selection on the 2023 slice (n={len(cv_val)}):")
    for vname, h, n in var_val:
        mark = "  <-- selected" if vname == selected_variant else ""
        print(f"  {vname}: val top-1 {pct(h, n)}{mark}")
    for vname in ref_order:
        stg, nonst, allg = test_variant_rows[vname]
        mark = "  <-- selected" if vname == selected_variant else ""
        print(f"  test reference {test_variant_labels[vname]}{mark}:"
              f" sub-tasks top-1 {pct(stg[1], stg[0])} (n={stg[0]})"
              f" | non-sub-tasks top-1 {pct(nonst[1], nonst[0])} (n={nonst[0]})"
              f" | whole test top-1 {pct(allg[1], allg[0])}"
              f" / top-3 {pct(allg[2], allg[0])}")
    print("  check: V-blend identical to V-token on rows without a parent (asserted)")
    print("  check: V-rule non-sub-task results identical to level 1 (asserted)")
    print(f"training (final model on whole train): {train_secs:.1f} s")
    print(f"prediction (whole test): {predict_secs:.1f} s")
    print(f"wrote {DOC_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
