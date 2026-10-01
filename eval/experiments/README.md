# eval/experiments

One-off scripts from the tuning sessions, kept as evidence of the process.
They produced the comparisons behind the rejected options in the main
[README](../../README.md#6-decisions-and-rejected-options) (keyword hybrid,
tail rescue, reranker depth, int8 / ONNX / bge-reranker-base, token budget).

They are not part of the measured pipeline and are not maintained: several
read caches under `eval/cache/` that are not in the repo, and some timings are
hard-coded from the runs they report. The scripts that produce the published
numbers stay in [eval/](..): `run_eval.py`, `run_parse_eval.py`,
`build_contexts.py`, `generate_answers.py`, `grade_answers.py`.

| Script | What it measured |
|---|---|
| `build_lists.py`, `w0ref.py`, `eval_rrf.py` | cached vector + keyword lists; offline weighted-RRF sweep (hybrid) |
| `rerank_offline.py`, `rerank_d500.py`, `rerank_routed.py` | offline m3 rerank of cached lists; description length; routed rule |
| `rerank_speed.py`, `rerank_speed_b.py`, `clean_timing.py`, `final_table.py` | reranker speed vs accuracy: m3 vs base, int8, ONNX, N=10/20 |
| `token_check.py`, `token_check_run.py` | reranker token counts at 500 vs 1000 description chars |
| `compute_report.py` | baseline `summary_desc` vs `summary_desc_filtered` report |

`rerank_speed_b.py` was in `scratch/`; an identical copy of `rerank_routed.py`
that was also there was dropped.
