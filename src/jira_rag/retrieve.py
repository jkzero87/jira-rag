#!/usr/bin/env python3
"""Retrieve jira.issue_chunks with Qwen3-Embedding-4B (CUDA if available, else CPU).

Queries are embedded with the Qwen3-Embedding retrieval instruction
prefix, then the SAME tokenizer settings (padding_side="left"),
last-token pooling, Matryoshka truncation and L2 normalization as
embed.py (constants and the pooling helper are imported from embed.py,
not copied). Search is exact pgvector cosine distance (no ANN index: a
sequential scan over ~59k rows), always through vector_rows().

Filtered retrieval: parse a question into a structured filter form, build a
WHERE clause from it, and restrict the vector search to matching issues.
The rejected keyword strategies (hybrid, rescue) live in experimental.py.
"""
import sys
import time
import psycopg2
import torch
from transformers import AutoModel, AutoTokenizer

from embed import DSN, MODEL, MAX_TOKENS, last_token_pool
from fusion import RRF_K, blend_vector_rerank
from query_parse import parse as parse_question, to_sql


QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
DIM = 1024  # stored Matryoshka dimension; must match jira.issue_chunks.embedding
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_tokenizer = None
_model = None
_dtype = None


def init():
    """Load tokenizer + model on DEVICE (lazy). Tries bfloat16, falls back to float32."""
    global _tokenizer, _model, _dtype
    if _model is not None:
        return _tokenizer, _model
    t0 = time.monotonic()
    print("loading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", trust_remote_code=True)
    model = None
    for dtype, label in ((torch.bfloat16, "bfloat16"), (torch.float32, "float32")):
        try:
            print(f"loading {MODEL} on {DEVICE} ({label}) ...", flush=True)
            cand = AutoModel.from_pretrained(MODEL, dtype=dtype, trust_remote_code=True,
                                             attn_implementation="sdpa").to(DEVICE)
            with torch.no_grad():
                cand(torch.zeros(1, 4, dtype=torch.long, device=DEVICE))  # smoke test
            model = cand
            break
        except Exception as exc:
            print(f"{label} on {DEVICE} failed ({type(exc).__name__}: {exc}); "
                  "trying next dtype", file=sys.stderr)
    if model is None:
        sys.exit(f"error: could not load model on {DEVICE}")
    print(f"model loaded on {DEVICE} ({label}) in {time.monotonic() - t0:.1f}s", flush=True)
    _tokenizer, _model, _dtype = tokenizer, model, label
    return tokenizer, model


def embed_query(q, dim=DIM):
    """Embed one query -> list of `dim` floats.

    Prefixes QUERY_PREFIX, tokenizes (truncated at MAX_TOKENS), left-pads,
    last-token pools, truncates to `dim` and L2-normalizes after truncation
    (identical pipeline to embed.py).
    """
    tokenizer, model = init()
    enc = tokenizer(QUERY_PREFIX + q, max_length=MAX_TOKENS, truncation=True)
    b = tokenizer.pad({"input_ids": [enc["input_ids"]]}, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**b)
    e = last_token_pool(out.last_hidden_state, b["attention_mask"])
    del out
    e = e.float()[:, :dim]
    e = e / torch.linalg.norm(e, dim=1, keepdim=True)
    return e[0].tolist()


VECTOR_SQL = """
SELECT ic.issue_key, ic.embedding <=> %s::vector AS distance
FROM jira.issue_chunks ic
JOIN jira.issues i ON i.issue_key = ic.issue_key
WHERE ic.strategy = %s{and_where}
ORDER BY distance, ic.issue_key
LIMIT %s
"""

COUNT_SQL = """
SELECT COUNT(*)
FROM jira.issue_chunks ic
JOIN jira.issues i ON i.issue_key = ic.issue_key
WHERE ic.strategy = %s{and_where}
"""


def _and_where(where):
    """SQL suffix for an optional WHERE fragment from query_parse.to_sql."""
    return f"\n  AND {where}" if where else ""


def vector_rows(cur, qvec, strategy, limit, where="", params=()):
    """The one vector search every strategy uses: cosine top `limit` over
    jira.issue_chunks for `strategy`, restricted by an optional WHERE
    fragment on jira.issues (alias i). Ties are broken by issue_key.

    Returns [(issue_key, distance)], nearest first."""
    cur.execute(VECTOR_SQL.format(and_where=_and_where(where)),
                [str(qvec), strategy, *params, limit])
    return cur.fetchall()


def _count_chunks(cur, strategy, where="", params=()):
    """Count issue_chunks for a strategy matching an optional WHERE."""
    cur.execute(COUNT_SQL.format(and_where=_and_where(where)), [strategy, *params])
    return cur.fetchone()[0]


def search_vec(qvec, strategy, k):
    """Cosine search over an already-embedded query (list of floats)."""
    conn = psycopg2.connect(DSN)
    try:
        return vector_rows(conn.cursor(), qvec, strategy, k)
    finally:
        conn.close()


def search(q, strategy, k):
    """Embed the query, then cosine-search jira.issue_chunks (top k)."""
    return search_vec(embed_query(q), strategy, k)


# ---------------------------------------------------------------------------
# Filtered retrieval: parse → WHERE → vector rank
# ---------------------------------------------------------------------------

def search_filtered(q, strategy, k=10):
    """Parse the question into a filter form, then run a vector search
    restricted to issues matching the WHERE clause.

    Returns (results, form, where_clause, filter_row_count) where:
      results: [(issue_key, distance)], nearest first (top k)
      form: the parsed form dict
      where_clause: the SQL WHERE fragment ("" if empty form)
      filter_row_count: number of issue_chunks matching the filter
                       (for correctness checking)

    If the form is empty (no filters), this is identical to search().
    """
    form = parse_question(q)
    where, params = to_sql(form)
    qvec = embed_query(q)

    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()
        filter_row_count = _count_chunks(cur, strategy, where, params)
        results = vector_rows(cur, qvec, strategy, k, where, params)
    finally:
        conn.close()

    return results, form, where, filter_row_count


# ---------------------------------------------------------------------------
# Rerank retrieval: vector top-N + bge-reranker-v2-m3 (fp32, 6 threads)
# ---------------------------------------------------------------------------

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_N = 20        # candidates to rerank
RERANK_DESC_CHARS = 500  # summary + first 500 chars of description
RERANK_RRF_K = RRF_K  # RRF constant for blending vector rank + rerank rank
RERANK_THREADS = 6   # torch threads

_reranker_model = None


def _init_reranker():
    """Load bge-reranker-v2-m3 ONCE per process (lazy)."""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    t0 = time.monotonic()
    print(f"loading bge-reranker-v2-m3 (fp32, {DEVICE}) ...", flush=True)
    torch.set_num_threads(RERANK_THREADS)
    from sentence_transformers import CrossEncoder
    _reranker_model = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=512)
    print(f"reranker loaded in {time.monotonic() - t0:.1f}s", flush=True)
    return _reranker_model


def search_rerank(q, strategy, k=10):
    """Parse the question, run a vector search for top RERANK_N candidates,
    then rerank with bge-reranker-v2-m3 (fp32, 6 threads).

    Routed: only rerank when the parser produced NO WHERE clause.
    Blended: RRF k=60 of vector rank and rerank rank (fusion.blend_vector_rerank).
    When WHERE is present, return the vector order unchanged (filtered path).

    Returns (results, form, where_clause, stage_seconds) where
      results: [(issue_key, score)], highest score first (top k)
      form: the parsed form dict
      where_clause: the SQL WHERE fragment ("" if empty form)
      stage_seconds: dict with 'parse', 'vector', 'rerank' times
    """
    stages = {}
    t = time.monotonic()
    form = parse_question(q)
    where, params = to_sql(form)
    stages["parse"] = time.monotonic() - t

    t = time.monotonic()
    qvec = embed_query(q)
    # Vector search: top RERANK_N (same WHERE as search_filtered)
    conn = psycopg2.connect(DSN)
    try:
        vec_rows = vector_rows(conn.cursor(), qvec, strategy, RERANK_N, where, params)
    finally:
        conn.close()
    stages["vector"] = time.monotonic() - t

    # Routed: only rerank when no WHERE
    if where:
        # Filtered path: return vector order unchanged
        stages["rerank"] = 0.0
        return vec_rows[:k], form, where, stages

    # Rerank: score each candidate with bge-reranker-v2-m3
    t = time.monotonic()
    model = _init_reranker()
    cand_keys = [key for key, _ in vec_rows]

    # Fetch doc texts in one query
    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT issue_key, summary, description FROM jira.issues "
            "WHERE issue_key = ANY(%s)", (cand_keys,))
        doc_map = {k: (s or "") + "\n" + (d or "")[:RERANK_DESC_CHARS]
                   for k, s, d in cur.fetchall()}
    finally:
        conn.close()

    pairs = [(q, doc_map.get(key, "")) for key in cand_keys]
    rerank_scores = model.predict(pairs, convert_to_scores=True)
    stages["rerank"] = time.monotonic() - t

    blended = blend_vector_rerank(cand_keys, list(rerank_scores), RERANK_RRF_K)
    return blended[:k], form, where, stages


def main():
    for key, dist in search("Upgrade ZooKeeper", "summary_only", 5):
        print(key, round(float(dist), 6))


if __name__ == "__main__":
    main()
