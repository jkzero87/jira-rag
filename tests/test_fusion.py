"""fusion.py: the RRF blend used by search_rerank and build_contexts."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "jira_rag"))

from fusion import blend_vector_rerank, rrf_term


def reference_blend(cand_keys, scores, k=60):
    """The blend as search_rerank wrote it before fusion.py (commit 9098340)."""
    rank_vec = {key: i for i, key in enumerate(cand_keys)}
    rank_rr = {key: i for i, key in enumerate(
        sorted(cand_keys, key=lambda c: -scores[cand_keys.index(c)]))}
    rrf = {key: 1.0 / (k + rank_vec[key] + 1) + 1.0 / (k + rank_rr[key] + 1)
           for key in cand_keys}
    order = sorted(cand_keys, key=lambda c: (-rrf[c], rank_vec[c]))
    return [(key, rrf[key]) for key in order]


def test_rrf_term_is_one_based():
    assert rrf_term(0) == 1 / 61
    assert rrf_term(9, k=10) == 1 / 20


def test_matches_previous_implementation_with_ties():
    rng = random.Random(0)
    for _ in range(200):
        keys = [f"SPARK-{i}" for i in rng.sample(range(1000), 20)]
        scores = [rng.choice([0.1, 0.5, 0.9, rng.random()]) for _ in keys]  # many ties
        assert blend_vector_rerank(keys, scores) == reference_blend(keys, scores)


def test_reranker_agreement_keeps_vector_order():
    keys = ["A", "B", "C"]
    out = blend_vector_rerank(keys, [3.0, 2.0, 1.0])
    assert [key for key, _ in out] == keys
    assert out[0][1] == 2 * rrf_term(0)


def test_full_disagreement_ties_broken_by_vector_rank():
    # A: ranks (1, 2), B: ranks (2, 1) -> equal RRF; A wins on vector rank.
    out = blend_vector_rerank(["A", "B"], [1.0, 2.0])
    assert [key for key, _ in out] == ["A", "B"]
    assert out[0][1] == out[1][1]
