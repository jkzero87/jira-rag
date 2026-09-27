#!/usr/bin/env python3
"""Shared retrieval metrics (single source of truth).

recall_at_k(keys, expected, k)  = |top-k ∩ expected| / |expected|
mrr_at_k(keys, expected, k)     = 1 / rank of the FIRST expected key in
                                   the top-k (0.0 if none)
"""


def recall_at_k(keys, expected, k=10):
    expected = list(expected)
    if not expected:
        return 0.0
    topk = list(keys)[:k]
    return len(set(topk) & set(expected)) / len(expected)


def mrr_at_k(keys, expected, k=10):
    expected = set(expected)
    for i, key in enumerate(list(keys)[:k]):
        if key in expected:
            return 1.0 / (i + 1)
    return 0.0


def recall_mrr(keys, expected, k=10):
    """(recall_at_k, mrr_at_k) — the pair every eval script needs."""
    return recall_at_k(keys, expected, k), mrr_at_k(keys, expected, k)


def group_metrics(entries):
    """Mean of per-question {"r10": ..., "mrr": ...} dicts over a list.

    Returns {"n": ..., "r10": ..., "mrr": ...}; empty list gives zeros.
    """
    n = len(entries)
    if not n:
        return {"n": 0, "r10": 0.0, "mrr": 0.0}
    return {
        "n": n,
        "r10": sum(e["r10"] for e in entries) / n,
        "mrr": sum(e["mrr"] for e in entries) / n,
    }
