#!/usr/bin/env python3
"""Reciprocal Rank Fusion (single source of truth for every RRF blend).

rrf_term(i, k) = 1 / (k + rank), rank = i + 1 is the 1-BASED rank of the
item at 0-based list position i. Every fusion in the repo uses this term,
so vector, rerank and keyword lists are always scored on the same scale.
"""

RRF_K = 60


def rrf_term(i, k=RRF_K):
    """RRF contribution of the item at 0-based position i (1-based rank i + 1)."""
    return 1.0 / (k + i + 1)


def blend_vector_rerank(cand_keys, rerank_scores, k=RRF_K):
    """Blend the vector order with the reranker order by RRF.

    cand_keys: issue keys in vector order (nearest first).
    rerank_scores: reranker scores, parallel to cand_keys (higher = better).
    Rerank ties keep vector order (stable sort); final ties are broken by
    vector rank. Returns [(issue_key, rrf_score)], best first.
    """
    rank_vec = {key: i for i, key in enumerate(cand_keys)}
    by_score = sorted(zip(cand_keys, rerank_scores), key=lambda ks: -ks[1])
    rank_rr = {key: i for i, (key, _) in enumerate(by_score)}
    rrf = {key: rrf_term(rank_vec[key], k) + rrf_term(rank_rr[key], k)
           for key in cand_keys}
    order = sorted(cand_keys, key=lambda key: (-rrf[key], rank_vec[key]))
    return [(key, rrf[key]) for key in order]
