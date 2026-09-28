#!/usr/bin/env python3
"""Retrieve jira.issue_chunks with Qwen3-Embedding-4B (CUDA if available, else CPU).

Queries are embedded with the Qwen3-Embedding retrieval instruction
prefix, then the SAME tokenizer settings (padding_side="left"),
last-token pooling, Matryoshka truncation and L2 normalization as
embed.py (constants and the pooling helper are imported from embed.py,
not copied). Search is plain pgvector cosine distance; no index, no filters.

Filtered retrieval: parse a question into a structured filter form, build a
WHERE clause from it, and restrict the vector search to matching issues.
"""
import logging
import sys
import time
import psycopg2
import torch
from transformers import AutoModel, AutoTokenizer

from embed import DSN, MODEL, MAX_TOKENS, last_token_pool
from query_parse import parse as parse_question, to_sql

logger = logging.getLogger(__name__)

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


SEARCH_SQL = """
SELECT issue_key, embedding <=> %s::vector AS distance
FROM jira.issue_chunks
WHERE strategy = %s
ORDER BY distance, issue_key
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
ORDER BY distance, ic.issue_key
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
# Rerank retrieval: vector top-N + bge-reranker-v2-m3 (fp32, 6 threads)
# ---------------------------------------------------------------------------

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_N = 20        # candidates to rerank
RERANK_DESC_CHARS = 500  # summary + first 500 chars of description
RERANK_RRF_K = 60    # RRF constant for blending vector rank + rerank rank
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


def _rerank_doc_text(issue_key):
    """summary + first RERANK_DESC_CHARS of description."""
    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT summary, description FROM jira.issues WHERE issue_key = %s",
            (issue_key,))
        row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return ""
    s, d = row
    return (s or "") + "\n" + (d or "")[:RERANK_DESC_CHARS]


def search_rerank(q, strategy, k=10):
    """Parse the question, run a vector search for top RERANK_N candidates,
    then rerank with bge-reranker-v2-m3 (fp32, 6 threads).

    Routed: only rerank when the parser produced NO WHERE clause.
    Blended: RRF k=60 of vector rank and rerank rank.
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
    where_prefix = " AND " if where else ""
    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()
        vec_sql = (
            "SELECT ic.issue_key, ic.embedding <=> %s::vector AS distance\n"
            "FROM jira.issue_chunks ic\n"
            "JOIN jira.issues i ON i.issue_key = ic.issue_key\n"
            "WHERE ic.strategy = %s"
            + (where_prefix + where if where else "")
            + "\nORDER BY distance, ic.issue_key\nLIMIT "
            + str(RERANK_N)
        )
        cur.execute(vec_sql, [str(qvec), strategy] + list(params))
        vec_rows = cur.fetchall()  # [(issue_key, distance)]
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

    # Blended RRF: vector rank + rerank rank (1-based, matching offline eval)
    rank_vec = {key: i for i, key in enumerate(cand_keys)}
    rank_rr = {key: i for i, key in enumerate(
        sorted(cand_keys, key=lambda k: -rerank_scores[cand_keys.index(k)]))}
    rrf = {key: 1.0 / (RERANK_RRF_K + rank_vec[key] + 1)
              + 1.0 / (RERANK_RRF_K + rank_rr[key] + 1)
           for key in cand_keys}
    order = sorted(cand_keys, key=lambda k: (-rrf[k], rank_vec[k]))

    return [(key, rrf[key]) for key in order[:k]], form, where, stages


# ---------------------------------------------------------------------------
# Hybrid retrieval: keyword (FTS) + vector, fused with RRF
# ---------------------------------------------------------------------------

# Keyword list: top 100 issues by ts_rank_cd against the generated fts column.
# plainto_tsquery turns the question into a tsquery with ' & ' between every
# token (AND semantics, which matches nothing for natural-language questions).
# The tsquery TEXT is rewritten: every ' & ' becomes ' | ' so the tokens are
# OR'd instead.  If plainto_tsquery comes back empty, the whole keyword list
# is skipped (no tsquery to match against).
#
# Rare-lexeme filter: a pure-OR tsquery matches 29k+ rows for a generic
# question because the common tokens (the/with/spark/...) match nearly every
# issue and dilute RRF.  We therefore keep only the question lexemes whose
# document frequency (jira.lexeme_df.ndoc, from ts_stat over jira.issues.fts)
# is below 2% of the total number of issues.  The tsquery text is rewritten
# in Python — every ' lexeme ' that is NOT rare is removed from the string —
# and the result is passed as %s so the same string is used for both the
# ts_rank_cd query and the @@ match.  If no rare lexemes remain, the keyword
# list is skipped entirely.
KEYWORD_SQL = """
SELECT i.issue_key,
       ts_rank_cd(i.fts, to_tsquery('english', %s)) AS rank
FROM jira.issues i
JOIN jira.issue_chunks ic ON ic.issue_key = i.issue_key
WHERE ic.strategy = %s
{where_prefix}{where}
  AND i.fts @@ to_tsquery('english', %s)
ORDER BY rank DESC
LIMIT 100
"""


def _rare_kw_tsquery(q, threshold=0.02):
    """Build the OR keyword tsquery string, keeping only the question's rare
    lexemes (ndoc < threshold * total issues in jira.lexeme_df).

    Returns (tsquery_string, kept_words) where tsquery_string is "" if no rare
    lexemes survive (caller skips the keyword list).
    """
    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM jira.issues")
        n_issues = cur.fetchone()[0]
        limit = threshold * n_issues

        cur.execute("SELECT plainto_tsquery('english', %s)::text", (q,))
        plain = cur.fetchone()[0]
        if not plain or not plain.strip():
            return "", []

        # plainto_tsquery text is 'lex1' & 'lex2' & ... — each lexeme is quoted
        # (single-quoted string literal form).  Strip the surrounding quotes to
        # get bare lexemes for the lexeme_df lookup, then re-quote for the
        # tsquery OR string we hand to to_tsquery().
        quoted = [t.strip() for t in plain.split('&') if t.strip()]
        bare = [t.strip("'") for t in quoted]
        kept = []
        if bare:
            cur.execute(
                "SELECT word, ndoc FROM jira.lexeme_df WHERE word = ANY(%s)",
                (bare,))
            ndoc = {w: d for w, d in cur.fetchall()}
            # A lexeme absent from lexeme_df never appears in any issue's fts,
            # so it matches nothing and can only add noise: drop it too.
            kept = [q_ for q_ in quoted
                    if ndoc.get(q_.strip("'"), 0) < limit]
    finally:
        conn.close()
    return (" | ".join(kept) if kept else ""), kept


def search_hybrid(q, strategy, k=10):
    """Parse the question, run a vector list and a keyword (FTS) list — both
    restricted by the SAME WHERE clause — then fuse with weighted Reciprocal
    Rank Fusion:  score = 1/(60 + rank_vec) + KW_WEIGHT * 1/(60 + rank_kw).

    KW_WEIGHT is set to 0.25 (Part E: no weight > 0 kept lookup MRR within
    0.05 of the w=0 baseline, so the default of 0.25 is retained).

    Returns (results, form, where_clause) where
      results: [(issue_key, rrf_score)], highest score first (top k)
      form: the parsed form dict
      where_clause: the SQL WHERE fragment ("" if empty form)
    text_terms is NOT used as a filter (it only ranks via the vector query).
    """
    form = parse_question(q)
    where, params = to_sql(form)
    where_prefix = " AND " if where else ""

    qvec = embed_query(q)

    # Keyword list: OR of the question's RARE lexemes only (ndoc < 2% of
    # issues), so common tokens (the/with/spark/...) don't dilute RRF.
    # If no rare lexemes remain, the keyword list is skipped entirely.
    kw_tsquery, kept_words = _rare_kw_tsquery(q)
    if not kw_tsquery:
        logger.warning("no rare question lexemes for %r (kept %r); "
                       "keyword list skipped", q, kept_words)
        run_kw = False
    else:
        run_kw = True
        logger.info("keyword lexemes for %r: kept %s (dropped the rest)",
                    q[:40], kept_words)

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
            + "\nORDER BY distance, ic.issue_key\nLIMIT 100"
        )
        cur.execute(vec_sql, [str(qvec), strategy] + list(params))
        vec_rows = cur.fetchall()  # [(issue_key, distance)]

        # --- Keyword list: top 100 by ts_rank_cd over rare lexemes ---
        kw_rows = []
        if run_kw:
            kw_sql = KEYWORD_SQL.format(where_prefix=where_prefix, where=where)
            kw_params = [kw_tsquery, strategy] + list(params) + [kw_tsquery]
            cur.execute(kw_sql, kw_params)
            kw_rows = cur.fetchall()  # [(issue_key, rank)]
    finally:
        conn.close()

    # --- Weighted RRF fusion (deterministic: tie-break by issue_key) ---
    KW_WEIGHT = 0.25  # Part E: best weight that keeps lookup MRR within 0.05
    # of the w=0 baseline.  No w>0 satisfied that constraint, so the
    # default 0.25 is retained (keyword list contributes but is secondary).
    rank_vec = {key: i for i, (key, _) in enumerate(vec_rows)}  # 0-based
    rank_kw = {key: i for i, (key, _) in enumerate(kw_rows)}
    all_keys = set(rank_vec) | set(rank_kw)
    fused = []
    for key in all_keys:
        score = (1.0 / (60 + rank_vec[key]) if key in rank_vec else 0.0) + \
                KW_WEIGHT * (1.0 / (60 + rank_kw[key]) if key in rank_kw else 0.0)
        fused.append((key, score))
    fused.sort(key=lambda x: -x[1])
    return fused[:k], form, where


# ---------------------------------------------------------------------------
# Rescue retrieval: vector order untouched, keyword hits fill the last m slots
# ---------------------------------------------------------------------------

RESCUE_M = 1  # offline tail-rescue sweep: m=1 (and m=2) tie for best overall
# r@10 with lookup/filtered MRR unchanged vs the w=0 baseline; m=1 is minimal.


def search_rescue(q, strategy, k=10, m=RESCUE_M):
    """Parse the question, run the SAME vector list and keyword list as
    search_hybrid (both restricted by the same WHERE clause), then return
    the vector order UNTOUCHED with the last m slots taken from the keyword
    list:

      final top-k = vector top-(k-m) + first m keyword hits not already in
      that head (fill from the vector list if the keyword list runs out).

    Unlike search_hybrid this never re-ranks or blends: the keyword list can
    only rescue gold the vector list missed, never demote a vector hit.
    m=0 is exactly summary_desc_filtered.

    Returns (results, form, where_clause) where
      results: [(issue_key, score)], score = vector distance for head keys
               and the RRF keyword score for rescued keys (diagnostic only)
      form: the parsed form dict
      where_clause: the SQL WHERE fragment ("" if empty form)
    text_terms is NOT used as a filter (it only ranks via the vector query).
    """
    form = parse_question(q)
    where, params = to_sql(form)
    where_prefix = " AND " if where else ""

    qvec = embed_query(q)

    kw_tsquery, kept_words = _rare_kw_tsquery(q)
    if not kw_tsquery:
        logger.warning("no rare question lexemes for %r (kept %r); "
                       "keyword list skipped", q, kept_words)
        run_kw = False
    else:
        run_kw = True
        logger.info("keyword lexemes for %r: kept %s (dropped the rest)",
                    q[:40], kept_words)

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
            + "\nORDER BY distance, ic.issue_key\nLIMIT 100"
        )
        cur.execute(vec_sql, [str(qvec), strategy] + list(params))
        vec_rows = cur.fetchall()  # [(issue_key, distance)]

        # --- Keyword list: top 100 by ts_rank_cd over rare lexemes ---
        kw_rows = []
        if run_kw:
            kw_sql = KEYWORD_SQL.format(where_prefix=where_prefix, where=where)
            kw_params = [kw_tsquery, strategy] + list(params) + [kw_tsquery]
            cur.execute(kw_sql, kw_params)
            kw_rows = cur.fetchall()  # [(issue_key, rank)]
    finally:
        conn.close()

    vec_keys = [key for key, _ in vec_rows]  # deterministic order
    dist = {key: d for key, d in vec_rows}
    rank_kw = {key: i for i, (key, _) in enumerate(kw_rows)}

    head = vec_keys[:k - m]
    seen = set(head)
    tail = []
    for key in (k2 for k2, _ in kw_rows):
        if len(tail) == m:
            break
        if key not in seen:
            tail.append(key)
            seen.add(key)
    # fill from the vector list if the keyword list runs out
    if len(tail) < m:
        for key in vec_keys[k - m:]:
            if len(tail) == m:
                break
            if key not in seen:
                tail.append(key)
                seen.add(key)

    results = [(key, dist[key]) for key in head]
    results += [(key, 1.0 / (60 + rank_kw[key])) for key in tail]
    return results, form, where


def main():
    for key, dist in search("Upgrade ZooKeeper", "summary_only", 5):
        print(key, round(float(dist), 6))


if __name__ == "__main__":
    main()
