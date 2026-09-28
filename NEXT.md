# NEXT.md

## Last changes (this session)

1. **Routed m3 reranker in retrieval (`src/jira_rag/retrieve.py`, `search_rerank`)** —
   parse → WHERE → vector top-20 → bge-reranker-v2-m3 (fp32, 6 threads),
   RRF k=60 blend of vector rank + rerank rank (1-based), rerank only when
   the parser produced NO WHERE clause (filtered path returns vector order
   unchanged). Ties broken by vector rank.
2. **`summary_desc_rerank` strategy added to `eval/run_eval.py`** — uses the
   cached parse form (`2026-09-27_6410fb1_parse.json`) and cached candidate
   list (`eval/cache/lists.json`) to match the offline rerank eval exactly.
3. **Full run_eval pass with 6 strategies** —
   `2026-09-27_f30d1b2_baseline.json` (996.9 s total).

## Chosen configuration

`summary_desc_rerank` (m3 fp32, 6 threads, N=20, RRF k=60, routed,
blended, 1-based ranks, DESC_CHARS=500, max_length=512).

Results (`run_eval.py`, 40 questions):

```
strategy        scope      n   recall@5   recall@10      MRR
------------------------------------------------------------
summary_desc_filtered overall   40     0.6519      0.7390   0.7202
summary_desc_rerank   overall   40     0.7235      0.7790   0.7701
summary_desc_rerank   lookup    15     0.9333      0.9333   0.7856
summary_desc_rerank   topic     15     0.3989      0.5300   0.6013
summary_desc_rerank   filtered  10     0.8958      0.9208   1.0000
```

vs `summary_desc_filtered` (baseline): 5 questions worse, 10 better per MRR.

Clean timings (6 threads, fp32 m3):
- N=20: 19.97 s/q (vector 3.55 + rerank 16.42 + parse 0.00)
- N=10: 8.63 s/q
- d500 N=20: 7.96 s/q, N=10: 3.58 s/q

Offline reference (rerank_d500, m3 d500 N=20 routed): r@10 0.7790, MRR 0.7701,
lookup .7856, topic .6013, filtered 1.0000, N=10 r@10 0.7390 MRR 0.7640,
worse=5 better=10. run_eval.py matches exactly.

## Rejected configurations

- **Keyword hybrid (`summary_desc_hybrid`)**: RRF of vector + keyword
  `websearch_to_tsquery`. Overall MRR 0.6475–0.6496, worse than filtered
  baseline 0.7202. Keyword list adds noise on topic questions.
- **Tail rescue (`summary_desc_rescue`, m=1)**: fills top-10 tail with
  keyword hits. Matches baseline exactly (0 worse, 0 better); no gain.
- **Pure rerank (no RRF blend)**: rerank rank alone. Tested offline;
  worse than blended RRF on topic questions (rerank over-weights short
  summaries).
- **bge-reranker-base**: smaller model. Tested offline; lower quality than
  m3 (d500 scores diverge from m3 by ~0.05 avg, flipping several top-10
  slots).
- **int8 torch (quantized)**: `torch.int8` quantization of m3. Tested
  offline; scores diverge from fp32 by ~0.01–0.03, flipping 2–3 top-10
  slots per question. Not worth the accuracy loss.
- **ONNX runtime**: `reranker_v2m3.onnx` (fp32 + b8 variants). Tested
  offline; scores match fp32 torch to ~1e-4 but throughput is not faster
  at batch size 20 on CPU (torch 6 threads ≈ ONNX 6 threads). No gain.

## Current best strategy

`summary_desc_rerank` — the first strategy to beat the `summary_desc_filtered`
baseline on overall MRR (0.7701 vs 0.7202, +0.0499) and r@10 (0.7790 vs
0.7390, +0.0400). Filtered-type MRR stays perfect (1.0000) because the
routed path returns the vector order unchanged when a WHERE clause is
present.

## Next task (open)

- Consider whether the routed design (rerank only when no WHERE) is the
  right default, or whether always-rerank + post-filter is better.
- Explore larger N (40, 80) for the candidate pool — d500 timings suggest
  N=40 would be ~16 s/q, still feasible.
- Try cross-encoder distillation: train a smaller reranker on m3 scores
  to get d500-level latency at m3-level accuracy.
- Evaluate on a held-out question set to guard against overfitting to the
  40-question gold.
