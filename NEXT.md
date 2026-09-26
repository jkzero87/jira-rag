# NEXT.md

## Current best strategy

`summary_desc_hybrid` (from `eval/results/2026-09-26_2d22233_baseline.json`):

| scope | recall@5 | recall@10 | MRR |
|---|---|---|---|
| overall | 0.6394 | 0.7390 | 0.6973 |
| lookup | 0.8000 | 0.8667 | 0.7151 |
| topic | 0.3411 | 0.4900 | 0.5222 |
| filtered | 0.8458 | 0.9208 | 0.9333 |

`summary_desc_filtered` is identical on all recall@N; hybrid wins only on MRR (overall +0.0062, lookup +0.0167).

## Known bugs (pending)

1. **`response_format` missing the `"schema"` wrapper** — the dict uses
   `{"type": "json_schema", "json_schema": {...}}` but llama-server expects
   `{"type": "json_schema", "schema": {...}}`. The grammar is therefore NOT
   enforced: the green test (schema says enum `red`/`blue`, prompt says answer
   "green") returned the literal text `Green`, which fails `json.loads`.
   The fence strip in `_call_llm` is a safety net, not a fix.

2. **Keyword query uses `plainto_tsquery` (AND) → 0 rows for 38/40 questions.**
   `plainto_tsquery('english', q)` ANDs every token, so natural-language
   questions match nothing. Only G05 (4 rows) and G22 (2 rows) return rows.
   The keyword list needs OR semantics (e.g. `to_tsquery` with `|` or
   `websearch_to_tsquery`) so that hybrid retrieval is actually tested.
   Current hybrid = pure vector in practice.

## Next task

1. Fix (a): change `response_format` to use the `"schema"` key so the grammar
   is actually enforced by llama-server. Verify with the green test — it must
   now return a valid JSON object with `color` in `["red","blue"]`, not `Green`.
2. Fix (b): switch the keyword query to OR semantics so the keyword list is
   non-empty for natural-language questions. Verify: rerun the keyword check
   for G04, G34, G27 — expect non-zero row counts.
3. Rerun parse eval — dropped gold must stay empty (no regressions).
4. Rerun the full 40-question eval.
5. Compare `summary_desc_filtered` vs `summary_desc_hybrid` from the same
   results file: overall + by type, worse/better question lists.
