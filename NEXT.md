# NEXT.md

## Last changes (this session)

1. **`response_format` now uses the `"schema"` key** — `query_parse.py` sends
   `{"type": "json_schema", "schema": _JSON_SCHEMA}`. Both shapes tested
   against llama-server enforce the grammar (green test returns
   `{"color": "..."}` that `json.loads` accepts, no fence).
2. **Keyword query now uses OR semantics** — `retrieve.py` builds the tsquery
   as `to_tsquery('english', replace(plainto_tsquery('english', q)::text,
   ' & ', ' | '))` and skips the keyword list when `plainto_tsquery` is empty.
   Keyword row counts for G04/G34/G27: 29371 / 16774 / 15742 (was ~0).

## Current best strategy

`summary_desc_filtered` (from `eval/results/2026-09-26_514b1c6_baseline.json`)
now beats `summary_desc_hybrid` on every metric; the OR keyword list dilutes
the vector list in RRF fusion.

| scope | filtered r@5 | filtered r@10 | filtered MRR | hybrid r@5 | hybrid r@10 | hybrid MRR |
|---|---|---|---|---|---|---|
| overall | 0.6394 | 0.7390 | 0.6911 | 0.5771 | 0.6350 | 0.5088 |
| lookup | 0.8000 | 0.8667 | 0.6984 | 0.8000 | 0.8000 | 0.3400 |
| topic | 0.3411 | 0.4900 | 0.5222 | 0.2167 | 0.3544 | 0.4167 |
| filtered | 0.8458 | 0.9208 | 0.9333 | 0.7833 | 0.8083 | 0.9000 |

Worse under hybrid (recall@10): G01, G02, G12, G25, G28, G30, G31, G35, G36, G37.
Better under hybrid: G03, G04 (the only two topic questions the keyword list
rescues from a zero).

Parse eval (`2026-09-26_514b1c6_parse.json`): 0 questions dropped gold —
no regression from either fix.

## Next task (open)

The OR keyword list is too broad (29k rows for G04) and hurts fusion. Ideas:
- cap the keyword list more tightly (top 100 by ts_rank_cd is already there;
  maybe the RRF constant k=60 is too small),
- weight the two lists differently,
- try `websearch_to_tsquery` (phrase + OR) instead of a pure OR rewrite,
- only fuse when the keyword list actually contains the gold (adaptive).
