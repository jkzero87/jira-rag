#!/usr/bin/env python3
"""Retrieve jira.issue_chunks with Qwen3-Embedding-4B on CPU.

Queries are embedded with the Qwen3-Embedding retrieval instruction
prefix, then the SAME tokenizer settings (padding_side="left"),
last-token pooling, Matryoshka truncation and L2 normalization as
embed.py (constants and the pooling helper are imported from embed.py,
not copied). Search is plain pgvector cosine distance; no index, no filters.

Filtered retrieval: parse a question into a structured filter form, build a
WHERE clause from it, and restrict the vector search to matching issues.
"""
import sys
import time
import psycopg2
import torch
from transformers import AutoModel, AutoTokenizer

from embed import DSN, MODEL, MAX_TOKENS, last_token_pool
from query_parse import parse as parse_question, to_sql

QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
DIM = 1024  # stored Matryoshka dimension; must match jira.issue_chunks.embedding

_tokenizer = None
_model = None
_dtype = None


def init():
    """Load tokenizer + model on CPU (lazy). Tries bfloat16, falls back to float32."""
    global _tokenizer, _model, _dtype
    if _model is not None:
        return _tokenizer, _model
    t0 = time.monotonic()
    print("loading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", trust_remote_code=True)
    model = None
    for dtype, label in ((torch.bfloat16, "bfloat16"), (torch.float32, "float32")):
        try:
            print(f"loading {MODEL} on CPU ({label}) ...", flush=True)
            cand = AutoModel.from_pretrained(MODEL, dtype=dtype, trust_remote_code=True,
                                             attn_implementation="sdpa")
            with torch.no_grad():
                cand(torch.zeros(1, 4, dtype=torch.long))  # CPU smoke test
            model = cand
            break
        except Exception as exc:
            print(f"{label} on CPU failed ({type(exc).__name__}: {exc}); "
                  "trying next dtype", file=sys.stderr)
    if model is None:
        sys.exit("error: could not load model on CPU")
    print(f"model loaded on CPU ({label}) in {time.monotonic() - t0:.1f}s", flush=True)
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
    b = tokenizer.pad({"input_ids": [enc["input_ids"]]}, return_tensors="pt")
    with torch.no_grad():
        out = model(**b)
    e = last_token_pool(out.last_hidden_state, b["attention_mask"])
    del out
    e = e.float()[:, :dim]
    e = e / torch.linalg.norm(e, dim=1, keepdim=True)
    return e[0].tolist()


SEARCH_SQL = """
SELECT issue_key, embedding <=> %s::vector AS distance
FROM jira.issue_chunks
WHERE strategy = %s
ORDER BY distance
LIMIT %s
"""


def search_vec(qvec, strategy, k):
    """Cosine search over an already-embedded query (list of floats)."""
    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()
        cur.execute(SEARCH_SQL, (str(qvec), strategy, k))
        return cur.fetchall()  # [(issue_key, distance)], nearest first
    finally:
        conn.close()


def search(q, strategy, k):
    """Embed the query, then cosine-search jira.issue_chunks (top k)."""
    return search_vec(embed_query(q), strategy, k)


# ---------------------------------------------------------------------------
# Filtered retrieval: parse → WHERE → vector rank
# ---------------------------------------------------------------------------

SEARCH_FILTERED_SQL = """
SELECT ic.issue_key, ic.embedding <=> %s::vector AS distance
FROM jira.issue_chunks ic
JOIN jira.issues i ON i.issue_key = ic.issue_key
WHERE ic.strategy = %s
{where_prefix}{where}
ORDER BY distance
LIMIT %s
"""


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

    if not where:
        # No filters — identical to plain search
        results = search_vec(embed_query(q), strategy, k)
        return results, form, "", _count_chunks(strategy, "", [])

    sql = SEARCH_FILTERED_SQL.format(where_prefix=" AND ", where=where)
    qvec = embed_query(q)

    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()

        # Count how many chunks match the filter (for correctness check)
        count_sql = (
            "SELECT COUNT(*) FROM jira.issue_chunks ic "
            "JOIN jira.issues i ON i.issue_key = ic.issue_key "
            "WHERE ic.strategy = %s"
        )
        if where:
            count_sql += " AND " + where
        cur.execute(count_sql, [strategy] + list(params))
        filter_row_count = cur.fetchone()[0]

        # Run the filtered vector search
        cur.execute(sql, [str(qvec), strategy] + list(params) + [k])
        results = cur.fetchall()
    finally:
        conn.close()

    return results, form, where, filter_row_count


def _count_chunks(strategy, where, params):
    """Count issue_chunks matching a strategy (+ optional WHERE)."""
    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()
        sql = "SELECT COUNT(*) FROM jira.issue_chunks ic WHERE ic.strategy = %s"
        if where:
            sql += " AND " + where
        cur.execute(sql, [strategy] + list(params))
        return cur.fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Hybrid retrieval: keyword (FTS) + vector, fused with RRF
# ---------------------------------------------------------------------------

# Keyword list: top 100 issues by ts_rank_cd against the generated fts column.
# The query string is the raw question with '&' replaced by '|' so that a
# conjunction becomes an OR (plainto_tsquery would AND every word, which
# over-constrains for natural-language questions).
KEYWORD_SQL = """
SELECT i.issue_key,
       ts_rank_cd(i.fts, plainto_tsquery('english', %s)) AS rank
FROM jira.issues i
JOIN jira.issue_chunks ic ON ic.issue_key = i.issue_key
WHERE ic.strategy = %s
{where_prefix}{where}
  AND i.fts @@ plainto_tsquery('english', %s)
ORDER BY rank DESC
LIMIT 100
"""


def search_hybrid(q, strategy, k=10):
    """Parse the question, run a vector list and a keyword (FTS) list — both
    restricted by the SAME WHERE clause — then fuse with Reciprocal Rank
    Fusion:  score = 1/(60 + rank_vec) + 1/(60 + rank_kw)  (0 if absent).

    Returns (results, form, where_clause) where
      results: [(issue_key, rrf_score)], highest score first (top k)
      form: the parsed form dict
      where_clause: the SQL WHERE fragment ("" if empty form)
    text_terms is NOT used as a filter (it only ranks via the vector query).
    """
    form = parse_question(q)
    where, params = to_sql(form)
    where_prefix = " AND " if where else ""

    # Keyword query: '&' → '|' so words are OR'd instead of AND'd.
    kw_query = q.replace("&", "|")

    qvec = embed_query(q)

    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()

        # --- Vector list: top 100 (same WHERE as search_filtered) ---
        vec_sql = (
            "SELECT ic.issue_key, ic.embedding <=> %s::vector AS distance\n"
            "FROM jira.issue_chunks ic\n"
            "JOIN jira.issues i ON i.issue_key = ic.issue_key\n"
            "WHERE ic.strategy = %s"
            + (where_prefix + where if where else "")
            + "\nORDER BY distance\nLIMIT 100"
        )
        cur.execute(vec_sql, [str(qvec), strategy] + list(params))
        vec_rows = cur.fetchall()  # [(issue_key, distance)]

        # --- Keyword list: top 100 by ts_rank_cd ---
        kw_sql = KEYWORD_SQL.format(where_prefix=where_prefix, where=where)
        kw_params = [kw_query, strategy] + list(params) + [kw_query]
        cur.execute(kw_sql, kw_params)
        kw_rows = cur.fetchall()  # [(issue_key, rank)]
    finally:
        conn.close()

    # --- RRF fusion ---
    rank_vec = {key: i for i, (key, _) in enumerate(vec_rows)}  # 0-based
    rank_kw = {key: i for i, (key, _) in enumerate(kw_rows)}
    all_keys = set(rank_vec) | set(rank_kw)
    fused = []
    for key in all_keys:
        score = (1.0 / (60 + rank_vec[key]) if key in rank_vec else 0.0) + \
                (1.0 / (60 + rank_kw[key]) if key in rank_kw else 0.0)
        fused.append((key, score))
    fused.sort(key=lambda x: -x[1])
    return fused[:k], form, where


def main():
    for key, dist in search("Upgrade ZooKeeper", "summary_only", 5):
        print(key, round(float(dist), 6))


if __name__ == "__main__":
    main()
