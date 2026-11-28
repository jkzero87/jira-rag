#!/usr/bin/env python3
"""Experiment 2 (journal 4.9): hippocampus - an embedding memory next to the router.

Question. The `rolling` router (monthly retraining, as-of labels,
V-token+balanced, C = 8; the variant selected in experiment 3,
docs/router_recency.md) learns components only through tf-idf weights. Does
an episodic memory of the training issues' embeddings - nearest neighbours
or per-component prototypes - add routing accuracy when blended with it?

Method. For each month M (2023-01 .. last data month) the month-M training
set is the `rolling` one: issues created before the first day of M with
non-empty as-of labels at that day (router_novelty.train_pool, the exact
construction of router_recency.run_rolling). Reused without change:
fit_balanced, make_predict and month_window (router_recency.py), load_data
and train_pool (router_novelty.py), choose_threshold and auto_stats
(router_confidence.py). make_predict already returns the full
probability matrix, so fit_balanced is called with its default arguments
and is not modified.

Scores per issue created in M, over the month-M class set (the components
of the as-of training labels):
  LR          the `rolling` probability matrix.
  kNN(k)      embeddings of strategy summary_desc (jira.issue_chunks,
              L2-normalised; norms asserted within 1e-5). The k most
              similar training issues by dot product; component score =
              sum of the similarities of the neighbours whose as-of label
              contains the component / sum of all k similarities.
              k in {10, 30, 100}.
  prototypes  per component, the mean embedding of its training issues
              (as-of labels), re-normalised; score = softmax over
              components of (cosine x 20).
  combined    (1 - w) * LR + w * MEMORY, MEMORY = kNN(k) or prototypes,
              w in {0, 0.25, 0.5, 0.75, 1.0}. A component missing from one
              source scores 0 there. w = 0 is asserted to reproduce
              rolling: validation top-1 3462/4733, test top-1 11579/15754.

Selection. (source, k, w) is chosen on VALIDATION top-1 only (the 2023
months); ties -> smaller w, then prototypes before kNN, then smaller k.
The 90% and 80% margin thresholds (top1 - top2 of the combined score) are
chosen on validation with choose_threshold and applied to test. Test is
never used to choose anything.

Resumable: each month is cached to scratch/hippocampus/<YYYY-MM>.npz (keys,
created, final components, LR proba, class names, kNN top-100 neighbour
indices and similarities, kNN(k) scores, prototype scores) and skipped if
present; delete the directory to force a rerun. Progress is printed per
month with elapsed time.

Writes docs/router_hippocampus.md (English), byte-identical on every run;
never edit by hand. Reads docs/router_embedding_gap.json (written by
scripts/router_embed_gap.py) for the embedding gap-fill section.

Run:
    .venv/bin/python scripts/router_hippocampus.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from profile_jira import load_env  # noqa: E402
from router_text_baseline import pct  # noqa: E402
from router_confidence import TARGETS, auto_stats, choose_threshold  # noqa: E402
from router_recency import fit_balanced, make_predict, month_window  # noqa: E402
from router_novelty import load_data, parse_t90, train_pool  # noqa: E402
from router_tossing import RESULTS_TOP15  # noqa: E402

SCRATCH = REPO / "scratch" / "hippocampus"
DOC_PATH = REPO / "docs" / "router_hippocampus.md"
GAP_JSON = REPO / "docs" / "router_embedding_gap.json"
STRATEGY = "summary_desc"
KS = (10, 30, 100)
K_MAX = max(KS)
WS = (0.0, 0.25, 0.5, 0.75, 1.0)
PROTO_SCALE = 20.0
NORM_TOL = 1e-5
# Sources in tie-break order: prototypes before kNN, then smaller k.
SOURCES = ("prototypes",) + tuple(f"knn{k}" for k in KS)
HIGHLIGHT = ("Declarative Pipelines", "Pandas API on Spark", "Optimizer",
             "Connect", "Spark Core", "Graph")
ROLLING_VAL = (3462, 4733)
ROLLING_TEST = (11579, 15754)


def source_label(src):
    return "prototypes" if src == "prototypes" else f"kNN (k={src[3:]})"


def load_embeddings(keys):
    """{issue_key: float64 vector} for strategy summary_desc; asserts every key
    is present and every norm is 1 within NORM_TOL."""
    env = load_env(REPO / ".env")
    import psycopg2
    conn = psycopg2.connect(
        host=env["PGHOST"], port=env["PGPORT"], dbname=env["PGDATABASE"],
        user=env["PGUSER"], password=env["PGPASSWORD"])
    cur = conn.cursor()
    cur.execute("SELECT issue_key, embedding::text FROM jira.issue_chunks"
                " WHERE strategy = %s AND chunk_index = 0 AND issue_key = ANY(%s)",
                (STRATEGY, list(keys)))
    emb = {k: np.array(v[1:-1].split(","), dtype=np.float64) for k, v in cur.fetchall()}
    conn.close()
    missing = set(keys) - set(emb)
    assert not missing, f"{len(missing)} issues have no {STRATEGY} embedding"
    norms = np.array([np.linalg.norm(v) for v in emb.values()])
    assert np.all(np.abs(norms - 1.0) <= NORM_TOL), \
        f"embedding norms outside 1 +- {NORM_TOL}: max dev {np.abs(norms - 1).max():.2e}"
    print(f"embeddings: {len(emb)} ({STRATEGY}), max |norm - 1| = "
          f"{np.abs(norms - 1).max():.2e}", flush=True)
    return emb


def indicator(labels, names):
    col = {c: j for j, c in enumerate(names)}
    Y = np.zeros((len(labels), len(names)))
    for i, la in enumerate(labels):
        for c in la:
            Y[i, col[c]] = 1.0
    return Y


def memory_scores(Eq, Etr, Ytr):
    """kNN top-K_MAX neighbours (indices, similarities), kNN(k) scores and
    prototype scores of the query rows Eq against the training rows Etr."""
    S = Eq @ Etr.T
    n = S.shape[0]
    nbr = np.zeros((n, K_MAX), dtype=np.int64)
    for i in range(n):
        cand = np.argpartition(-S[i], K_MAX - 1)[:K_MAX]
        # order by similarity desc, ties by training index: deterministic
        nbr[i] = cand[np.lexsort((cand, -S[i, cand]))]
    sim = np.take_along_axis(S, nbr, axis=1)
    knn = {}
    for k in KS:
        s = sim[:, :k]
        assert np.all(s.sum(axis=1) > 0), "non-positive kNN similarity sum"
        knn[k] = np.einsum("nk,nkc->nc", s, Ytr[nbr[:, :k]]) / s.sum(axis=1, keepdims=True)
    P = (Ytr.T @ Etr) / Ytr.sum(axis=0)[:, None]
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    z = PROTO_SCALE * (Eq @ P.T)
    z -= z.max(axis=1, keepdims=True)
    proto = np.exp(z)
    proto /= proto.sum(axis=1, keepdims=True)
    return nbr, sim, knn, proto


def run_months(months, times, by_time, changes, parent_key_map, summary, t_start):
    todo = [ym for ym in months if not (SCRATCH / f"{ym[0]:04d}-{ym[1]:02d}.npz").exists()]
    for ym in months:
        if ym not in todo:
            print(f"[hippocampus] {ym[0]:04d}-{ym[1]:02d} cached "
                  f"(elapsed={(time.perf_counter() - t_start) / 60:.1f}min)", flush=True)
    if not todo:
        return
    emb = load_embeddings([r[0] for r in by_time])
    for ym in todo:
        t0 = time.perf_counter()
        _start, mrows, tr_rows, tr_labels = train_pool(ym, times, by_time, changes)
        vec, clf, mlb, fit_s = fit_balanced(tr_rows, tr_labels, None, parent_key_map, summary)
        names = list(mlb.classes_)
        lr = np.asarray(make_predict(vec, clf, parent_key_map, summary)(mrows))
        Etr = np.stack([emb[r[0]] for r in tr_rows])
        Eq = np.stack([emb[r[0]] for r in mrows])
        Ytr = indicator(tr_labels, names)
        nbr, sim, knn, proto = memory_scores(Eq, Etr, Ytr)
        path = SCRATCH / f"{ym[0]:04d}-{ym[1]:02d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            keys=np.array([r[0] for r in mrows]),
            created=np.array([r[1].timestamp() for r in mrows], dtype=float),
            final=np.array([list(r[4]) for r in mrows], dtype=object),
            lr=lr, names=np.array(names), nbr_idx=nbr, nbr_sim=sim,
            proto=proto, **{f"knn{k}": knn[k] for k in KS})
        print(f"[hippocampus] {ym[0]:04d}-{ym[1]:02d} train={len(tr_rows)} "
              f"month={len(mrows)} classes={len(names)} fit={fit_s:.1f}s "
              f"month={time.perf_counter() - t0:.1f}s "
              f"elapsed={(time.perf_counter() - t_start) / 60:.1f}min", flush=True)


def load_month(ym):
    path = SCRATCH / f"{ym[0]:04d}-{ym[1]:02d}.npz"
    assert path.exists(), f"missing {path}"
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in ("lr", *(f"knn{k}" for k in KS))}
        d["prototypes"] = z["proto"]
        d["names"] = z["names"].tolist()
        d["final"] = [list(a) for a in z["final"]]
        d["keys"] = z["keys"].tolist()
    return d


def align(scores, names, union):
    """Place a (n, len(names)) score matrix on the union column order; absent
    components score 0."""
    out = np.zeros((scores.shape[0], len(union)))
    col = {c: j for j, c in enumerate(union)}
    out[:, [col[c] for c in names]] = scores
    return out


def month_eval(d, src, w):
    """(top-1 component per issue, margin) of the combined score."""
    names = d["names"]  # LR and memory share the month-M class set
    union = sorted(set(names))
    combined = (1.0 - w) * align(d["lr"], names, union) + w * align(d[src], names, union)
    top = combined.argmax(axis=1)
    part = np.partition(combined, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    return [union[j] for j in top], margin


def split_eval(data, months, src, w):
    """{'val'|'test': {final, top1, correct1, margin, hits, n, month}}."""
    out = {}
    for split, pick in (("val", lambda ym: ym[0] == 2023), ("test", lambda ym: ym[0] >= 2024)):
        final, top1, margin, month = [], [], [], []
        for ym in months:
            if not pick(ym):
                continue
            t, m = month_eval(data[ym], src, w)
            final.extend(data[ym]["final"])
            top1.extend(t)
            margin.append(m)
            month.extend([ym] * len(t))
        correct1 = [t in f for t, f in zip(top1, final)]
        out[split] = {"final": final, "top1": top1, "correct1": correct1,
                      "margin": np.concatenate(margin), "hits": sum(correct1),
                      "n": len(final), "month": month}
    return out


def auto_row(A, t):
    n, n_auto, wrong, wp100 = auto_stats(A["margin"], A["correct1"], t)
    return {"n": n, "n_auto": n_auto, "cov": n_auto / n if n else 0.0,
            "prec": (n_auto - wrong) / n_auto if n_auto else 0.0, "wp100": wp100}


def confidence(ev):
    """90%/80% thresholds chosen on VALIDATION only, applied to both splits."""
    out = {}
    for target in TARGETS:
        t = choose_threshold(ev["val"]["margin"], ev["val"]["correct1"], target)
        out[target] = (t, None if t is None else auto_row(ev["val"], t),
                       None if t is None else auto_row(ev["test"], t))
    return out


def fmt_wp(v):
    return "—" if v is None else f"{v:.1f}"


# True components whose wrong top-1 predictions are dissected (SQL magnet).
RARE_TRUE = ("Kubernetes", "Connect", "Declarative Pipelines", "Optimizer", "Graph")


def magnet_stats(ev):
    """Wrong top-1 predictions: total, count predicted to SQL, and for each
    rare true component the top-3 wrong predicted components with counts."""
    out = {}
    for split in ("val", "test"):
        A = ev[split]
        wrong = [i for i in range(A["n"]) if not A["correct1"][i]]
        pred_sql = sum(1 for i in wrong if A["top1"][i] == "SQL")
        per_comp = {}
        for c in RARE_TRUE:
            cnt = Counter(A["top1"][i] for i in wrong if c in A["final"][i])
            per_comp[c] = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        out[split] = {"n": A["n"], "wrong": len(wrong), "pred_sql": pred_sql,
                      "per_comp": per_comp}
    return out


def build_doc(ctx):
    a = []
    ap = a.append
    g = ctx["gap"]
    sel = ctx["selected"]
    sel_label = f"{source_label(sel[0])}, w = {sel[1]}"
    ap("# Router hippocampus: embedding memory (experiment 2)")
    ap("")
    ap("Generated by `scripts/router_hippocampus.py` on every run; never edit by hand.")
    ap("")
    ap("**Question.** The `rolling` router (monthly retraining, as-of labels, V-token+balanced, C = 8; the variant selected in experiment 3, docs/router_recency.md) learns components only through tf-idf weights. Does an episodic memory of the training issues' embeddings, nearest neighbours or per-component prototypes, add routing accuracy when blended with it?")
    ap("")
    ap("**Method.** For each month M (2023-01 .. "
       f"{ctx['months'][-1][0]:04d}-{ctx['months'][-1][1]:02d}, {len(ctx['months'])} months) the training set is the `rolling` one: issues created before the first day of M whose as-of labels at that day are non-empty. Validation = the 2023 months ({ctx['n_val']} issues); test = 2024 onward ({ctx['n_test']} issues, data ends {ctx['data_end']}). Ground truth = final components; a top-1 hit = the predicted component is one of them.")
    ap("")
    ap("- **LR**: the `rolling` probability matrix (fit_balanced and make_predict of `scripts/router_recency.py`, unchanged).")
    ap(f"- **kNN(k)**: `{STRATEGY}` embeddings (Qwen3-Embedding-4B, 1024 dims, L2-normalised; norms asserted within {NORM_TOL:g}). The k most similar training issues by dot product; component score = sum of the similarities of the neighbours whose as-of label contains the component, divided by the sum of all k similarities. k in {{{', '.join(map(str, KS))}}}.")
    ap(f"- **Prototypes**: per component, the mean embedding of its training issues (as-of labels), re-normalised; score = softmax over components of (cosine x {PROTO_SCALE:g}).")
    ap(f"- **Combined**: score = (1 - w) x LR + w x MEMORY, w in {{{', '.join(f'{w:g}' for w in WS)}}}. All three sources score the same month-M class set (the components of the as-of training labels); a component missing from a source would score 0 there.")
    ap("")
    ap(f"**Reproduction check.** w = 0 is asserted to reproduce `rolling` exactly: validation top-1 {ROLLING_VAL[0]}/{ROLLING_VAL[1]} and test top-1 {ROLLING_TEST[0]}/{ROLLING_TEST[1]} (docs/router_recency.md).")
    ap("")
    ap("### a) Validation top-1 grid")
    ap("")
    ap("| memory source | " + " | ".join(f"w = {w:g}" for w in WS) + " |")
    ap("|---|" + "---|" * len(WS))
    for src in SOURCES:
        cells = []
        for w in WS:
            h = ctx["grid"][(src, w)]
            c = f"{pct(h, ctx['n_val'])} ({h})"
            cells.append(f"**{c}**" if (src, w) == sel else c)
        ap(f"| {source_label(src)} | " + " | ".join(cells) + " |")
    ap("")
    ap(f"Selected on validation top-1 (ties -> smaller w, then prototypes before kNN, then smaller k): **{sel_label}**.")
    ap("")
    ap("### b) Selected configuration vs rolling")
    ap("")
    ap("| configuration | val top-1 | test top-1 |")
    ap("|---|---|---|")
    for label, ev in (("rolling (w = 0)", ctx["roll"]), (f"selected ({sel_label})", ctx["sel"])):
        ap(f"| {label} | {pct(ev['val']['hits'], ev['val']['n'])} ({ev['val']['hits']}/{ev['val']['n']}) "
           f"| {pct(ev['test']['hits'], ev['test']['n'])} ({ev['test']['hits']}/{ev['test']['n']}) |")
    ap("")
    ap("Margin = top1 - top2 of the configuration's score. Thresholds are chosen on validation only (lowest grid t reaching the target precision) and applied to test.")
    ap("")
    for target in TARGETS:
        ap(f"**Target precision {int(target * 100)}% on validation:**")
        ap("")
        ap("| configuration | margin ≥ | val coverage | val precision | val wrong/100 | test coverage | test precision | test wrong/100 |")
        ap("|---|---|---|---|---|---|---|---|")
        for label, conf in (("rolling", ctx["roll_conf"]), ("selected", ctx["sel_conf"])):
            t, v, s = conf[target]
            if t is None:
                ap(f"| {label} | — | — | — | — | — | — | — |")
                continue
            ap(f"| {label} | {t:.2f} | {100 * v['cov']:.1f}% ({v['n_auto']}/{v['n']}) | {100 * v['prec']:.1f}% | {fmt_wp(v['wp100'])} "
               f"| {100 * s['cov']:.1f}% ({s['n_auto']}/{s['n']}) | {100 * s['prec']:.1f}% | {fmt_wp(s['wp100'])} |")
        ap("")
    ap("### c) Test top-1 by component")
    ap("")
    ap("The 15 most frequent test components (docs/router_results.md), plus any highlighted component outside them. Highlighted components are in bold. n train = issues carrying the component in the as-of training labels of month 2024-01 (labels as of 2024-01-01).")
    ap("")
    ap("| component | n train (as of 2024-01-01) | n test | rolling top-1 | selected top-1 | difference (pp) |")
    ap("|---|---|---|---|---|---|")
    for c, n_tr, n, rh, sh in ctx["per_comp"]:
        name = f"**{c}**" if c in HIGHLIGHT else c
        extra = "" if c in RESULTS_TOP15 else " (outside top 15)"
        diff = 100.0 * (sh - rh) / n if n else 0.0
        ap(f"| {name}{extra} | {n_tr} | {n} | {pct(rh, n)} | {pct(sh, n)} | {diff:+.1f} |")
    ap("")
    ap("### d) SQL magnet: wrong top-1 predictions")
    ap("")
    ap("Hypothesis under test: the kNN vote is frequency-weighted, so the memory pulls wrong top-1 predictions toward SQL (the \"SQL magnet\" that class_weight = \"balanced\" removed from the LR in experiment 7, docs/router_label_hygiene.md). Wrong top-1 = predicted component not among the final components.")
    ap("")
    ap("| configuration | split | n | wrong top-1 | wrong -> SQL |")
    ap("|---|---|---|---|---|")
    for label, st in ctx["magnet"]:
        for split in ("val", "test"):
            s = st[split]
            ap(f"| {label} | {split} | {s['n']} | {s['wrong']} | {s['pred_sql']} |")
    ap("")
    ap("| true component | configuration | split | top-3 wrong predictions |")
    ap("|---|---|---|---|---|")
    for c in RARE_TRUE:
        for label, st in ctx["magnet"]:
            for split in ("val", "test"):
                top3 = st[split]["per_comp"][c]
                cell = ", ".join(f"{p} ({n})" for p, n in top3) if top3 else "—"
                ap(f"| {c} | {label} | {split} | {cell} |")
    ap("")
    ap("### e) Test top-1 by quarter")
    ap("")
    ap("| quarter | n | rolling top-1 | selected top-1 |")
    ap("|---|---|---|---|")
    for q, n, rh, sh in ctx["quarters"]:
        ap(f"| {q} | {n} | {pct(rh, n)} | {pct(sh, n)} |")
    ap("")
    ap("### f) Embedding gap fill")
    ap("")
    ap(f"Before this experiment, {g['gap_issues']} router issues (created {g['gap_created_min'][:10]} .. {g['gap_created_max'][:10]}) had no embedding: they existed only in `bronze.jira_issue_raw`, and `jira.issue_chunks` references `jira.issues`. `scripts/router_embed_gap.py` inserted them into `jira.issues` with `src/jira_rag/ingest.py --bronze-keys-file` (flatten and UPSERT of ingest.py; existing rows skipped) and embedded them with `src/jira_rag/embed.py --keys-file` for both strategies. The {g['existing_rows']} existing `jira.issues` rows kept the same count and (issue_key, summary, description) md5 `{g['existing_md5']}`.")
    ap("")
    ap("| table | before | after |")
    ap("|---|---|---|")
    for k in g["before"]:
        ap(f"| `{k}` | {g['before'][k]} | {g['after'][k]} |")
    ap("")
    cc = g["consistency_check"]
    ap(f"Consistency check before writing (`scripts/embed_consistency_check.py`): the 20 smallest issue keys created in 2026-09 ({cc['keys'][0]} .. {cc['keys'][-1]}) re-embedded with the embed.py code path and compared with their stored vectors.")
    ap("")
    ap("| strategy | min cosine | mean cosine |")
    ap("|---|---|---|")
    for s in ("summary_only", "summary_desc"):
        ap(f"| {s} | {cc[s]['min_cosine']:.6f} | {cc[s]['mean_cosine']:.6f} |")
    ap("")
    ap(f"After the fill every one of the {g['load_rows']} `load_rows()` issues has an embedding for both strategies (asserted).")
    ap("")
    ap("### Note on the test split")
    ap("")
    ap("The test split was never used to choose anything in this experiment: the memory source, k, w and the margin thresholds are all chosen on validation only; test is reported as reference.")
    ap("")
    ap("### Resumability")
    ap("")
    ap("Per-month scores are cached at `scratch/hippocampus/<YYYY-MM>.npz` and skipped when present; delete the directory to force a rerun.")
    ap("")
    return "\n".join(a)


def main():
    t_start = time.perf_counter()
    summary, parent_key_map, changes, by_time, times, months, data_end = load_data()
    run_months(months, times, by_time, changes, parent_key_map, summary, t_start)
    data = {ym: load_month(ym) for ym in months}

    # ---- grid: validation top-1 for every (source, w) ----
    evals, grid = {}, {}
    for src in SOURCES:
        for w in WS:
            ev = split_eval(data, months, src, w)
            evals[(src, w)] = ev
            grid[(src, w)] = ev["val"]["hits"]
    n_val = evals[(SOURCES[0], 0.0)]["val"]["n"]
    n_test = evals[(SOURCES[0], 0.0)]["test"]["n"]
    for src in SOURCES:
        ev = evals[(src, 0.0)]
        assert (ev["val"]["hits"], ev["val"]["n"]) == ROLLING_VAL, \
            f"w=0 ({src}) validation top-1 {ev['val']['hits']}/{ev['val']['n']} != {ROLLING_VAL}"
        assert (ev["test"]["hits"], ev["test"]["n"]) == ROLLING_TEST, \
            f"w=0 ({src}) test top-1 {ev['test']['hits']}/{ev['test']['n']} != {ROLLING_TEST}"
    print(f"[assert] w=0 reproduces rolling: val {ROLLING_VAL[0]}/{ROLLING_VAL[1]}, "
          f"test {ROLLING_TEST[0]}/{ROLLING_TEST[1]}")

    # ---- selection on VALIDATION top-1 only ----
    selected = min(grid, key=lambda sw: (-grid[sw], sw[1], SOURCES.index(sw[0])))
    print(f"[select] {selected} val top-1 {grid[selected]}/{n_val}")
    roll = evals[(SOURCES[0], 0.0)]
    sel = evals[selected]
    roll_conf = confidence(roll)
    sel_conf = confidence(sel)
    assert roll_conf[0.90][0] == parse_t90(), \
        f"rolling t90 {roll_conf[0.90][0]} differs from docs/router_recency.md {parse_t90()}"

    # ---- per component (test) ----
    _s, _m, _tr, tr_labels_2024 = train_pool((2024, 1), times, by_time, changes)
    n_train = Counter(c for la in tr_labels_2024 for c in la)
    test_final = roll["test"]["final"]
    cs = Counter(c for f in test_final for c in f)
    top15 = [c for c, _n in sorted(cs.items(), key=lambda kv: (-kv[1], kv[0]))[:15]]
    assert top15 == RESULTS_TOP15, top15
    comps = RESULTS_TOP15 + [c for c in HIGHLIGHT if c not in RESULTS_TOP15]
    per_comp = []
    for c in comps:
        idx = [i for i, f in enumerate(test_final) if c in f]
        assert idx, f"highlighted component {c} has no test issue"
        per_comp.append((c, n_train[c], len(idx),
                         sum(roll["test"]["correct1"][i] for i in idx),
                         sum(sel["test"]["correct1"][i] for i in idx)))

    # ---- per quarter (test) ----
    qagg = {}
    for i, ym in enumerate(roll["test"]["month"]):
        q = f"{ym[0]}Q{(ym[1] - 1) // 3 + 1}"
        n, rh, sh = qagg.get(q, (0, 0, 0))
        qagg[q] = (n + 1, rh + roll["test"]["correct1"][i], sh + sel["test"]["correct1"][i])
    quarters = [(q, *qagg[q]) for q in sorted(qagg)]

    # ---- SQL magnet: wrong top-1 predictions ----
    magnet = [("rolling", magnet_stats(roll)),
              (f"{source_label(selected[0])}, w = {selected[1]:g}", magnet_stats(sel))]

    gap = json.loads(GAP_JSON.read_text(encoding="utf-8"))
    DOC_PATH.write_text(build_doc({
        "months": months, "data_end": data_end, "n_val": n_val, "n_test": n_test,
        "grid": grid, "selected": selected, "roll": roll, "sel": sel,
        "roll_conf": roll_conf, "sel_conf": sel_conf, "per_comp": per_comp,
        "quarters": quarters, "magnet": magnet, "gap": gap}), encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    print(f"total elapsed: {(time.perf_counter() - t_start) / 60:.1f}min")


if __name__ == "__main__":
    main()
