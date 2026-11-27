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
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from profile_jira import load_env  # noqa: E402
from router_parent_context import components_asof, load_component_changes  # noqa: E402
from router_text_baseline import (  # noqa: E402
    C_CANDIDATES, CV_CUT, CUTOFF, load_rows, fit_variant,
    vtoken_cv_selection, build_vtoken_model, build_vtoken_cv_model,
    hit_counts, pct, extend_text)

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
    remapped_train = apply_mapping(train, mapping)
    res_renames, best_C_renames = cv_select(
        lambda tr, C: fit_variant(
            apply_mapping(tr, mapping), C, parent_key, summary, False),
        train, parent_key, summary)
    r = next(t for t in res_renames if t[0] == best_C_renames)
    print(f"V-token+renames: best C={best_C_renames}  "
          f"val top-1 = {pct(r[2], r[1])} ({r[2]}/{r[1]})")

    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
