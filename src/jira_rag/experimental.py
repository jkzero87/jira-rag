#!/usr/bin/env python3
"""Rejected retrieval strategies, kept as evidence of the process.

Both were measured against summary_desc_filtered and lost (see README,
"Decisions and rejected options"); eval/run_eval.py still scores them.
Nothing in the chosen pipeline (retrieve.search_rerank) imports this file.

  summary_desc_hybrid  vector list + keyword (FTS) list, weighted RRF
  summary_desc_rescue  vector order + keyword hits in the last m slots

The parser and query embedding are called through the retrieve module
(retrieve.parse_question, retrieve.embed_query), so the eval's parse cache
and timing patches apply here too.
"""
import logging

import psycopg2

import retrieve
from fusion import rrf_term
from query_parse import to_sql

logger = logging.getLogger(__name__)
DSN = retrieve.DSN
CANDIDATES = 100  # vector and keyword list depth for both strategies

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
    Rank Fusion:  score = rrf_term(i_vec) + KW_WEIGHT * rrf_term(i_kw), where
    rrf_term(i) = 1/(60 + i + 1) is the 1-based term every RRF in the repo uses
    (results recorded before this file existed used 1/(60 + i), 0-based).

    KW_WEIGHT is set to 0.25 (Part E: no weight > 0 kept lookup MRR within
    0.05 of the w=0 baseline, so the default of 0.25 is retained).

    Returns (results, form, where_clause) where
      results: [(issue_key, rrf_score)], highest score first (top k)
      form: the parsed form dict
      where_clause: the SQL WHERE fragment ("" if empty form)
    text_terms is NOT used as a filter (it only ranks via the vector query).
    """
    form = retrieve.parse_question(q)
    where, params = to_sql(form)
    where_prefix = " AND " if where else ""

    qvec = retrieve.embed_query(q)

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

        # --- Vector list: top CANDIDATES (same WHERE as search_filtered) ---
        vec_rows = retrieve.vector_rows(cur, qvec, strategy, CANDIDATES,
                                        where, params)  # [(issue_key, distance)]

        # --- Keyword list: top 100 by ts_rank_cd over rare lexemes ---
        kw_rows = []
        if run_kw:
            kw_sql = KEYWORD_SQL.format(where_prefix=where_prefix, where=where)
            kw_params = [kw_tsquery, strategy] + list(params) + [kw_tsquery]
            cur.execute(kw_sql, kw_params)
            kw_rows = cur.fetchall()  # [(issue_key, rank)]
    finally:
        conn.close()

    # --- Weighted RRF fusion (deterministic: ties broken by issue_key) ---
    KW_WEIGHT = 0.25  # Part E: best weight that keeps lookup MRR within 0.05
    # of the w=0 baseline.  No w>0 satisfied that constraint, so the
    # default 0.25 is retained (keyword list contributes but is secondary).
    rank_vec = {key: i for i, (key, _) in enumerate(vec_rows)}  # 0-based
    rank_kw = {key: i for i, (key, _) in enumerate(kw_rows)}
    all_keys = set(rank_vec) | set(rank_kw)
    fused = []
    for key in all_keys:
        score = (rrf_term(rank_vec[key]) if key in rank_vec else 0.0) + \
                KW_WEIGHT * (rrf_term(rank_kw[key]) if key in rank_kw else 0.0)
        fused.append((key, score))
    fused.sort(key=lambda x: (-x[1], x[0]))
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
    form = retrieve.parse_question(q)
    where, params = to_sql(form)
    where_prefix = " AND " if where else ""

    qvec = retrieve.embed_query(q)

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

        # --- Vector list: top CANDIDATES (same WHERE as search_filtered) ---
        vec_rows = retrieve.vector_rows(cur, qvec, strategy, CANDIDATES,
                                        where, params)  # [(issue_key, distance)]

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
    # a tail key filled from the vector list has no keyword rank: keep its distance
    results += [(key, rrf_term(rank_kw[key]) if key in rank_kw else dist[key])
                for key in tail]
    return results, form, where
