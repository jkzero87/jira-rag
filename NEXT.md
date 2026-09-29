# NEXT.md

## Scope

- jira-rag is a portfolio project: the goal is a working, measured and
  well-documented RAG, not a service to run day to day.
- OUT of scope: a live `ask` command, and fitting the 27B + embedder +
  reranker together in 16 GB VRAM (they need ~25 GB). Hardware is fixed; no
  model swaps for that purpose.
- The system runs in phases, and that is the documented design: (1) parse
  with the 27B, (2) retrieval on GPU with the 27B stopped, (3) generation
  with the 27B.

## Next session

- [x] Phase 3 (commits `3d5a427`, `df5a31e`, `aead08c`): `eval/generate_answers.py`
  + `eval/grade_answers.py` (hit, invented citations, no_citation,
  seconds/tokens, overall + by type), run on all 40 with the 27B, using
  `eval/cache/contexts_57b160d.json`. Result: hit 39/40 (ceiling was
  39/40; miss: G19), 0 invented citations.
- [x] `examples.md` (`9f45c78`): 4-5 real cases (question -> retrieved issues
  -> answer with citations), taken from the phase-3 answers.
- [x] README.md: architecture diagram, results table (recall@10 0.55 -> 0.74 ->
  0.78, MRR), decisions and rejected options, generation results, hardware
  and the phased-run limitation, how to reproduce.

## Rule: retrieval experiments on GPU, 27B stopped

All retrieval experiments (run_eval.py, embedder + reranker) run on GPU
(conda `dl`, CUDA), with the 27B (llama-server, port 8092) STOPPED so the
GPU is free for the Qwen3-Embedding-4B embedder and bge-reranker-v2-m3.
The parser is an exception: it needs the 27B server live. Live parse runs
therefore need the 27B up (and then GPU experiments must stop it again);
reruns that only need search/rerank use a parse cache
(see the fingerprint rule below).

## GPU results (embedder + reranker on CUDA)

CPU vs GPU, `summary_desc_rerank`, 40 questions, identical settings
(fp32 m3, N=20, RRF k=60, d500):

| metric             | CPU (e4f9511) | GPU (788dc8b) |
|--------------------|---------------|---------------|
| rerank recall@10   | 0.7790        | 0.7790        |
| rerank MRR         | 0.7576        | 0.7567        |
| eval wall time     | 1253 s        | 55 s          |

Quality is the same (MRR delta 0.0009, within run-to-run noise); the win is
23x wall time. GPU run: `eval/results/2026-09-28_788dc8b_baseline.json`
(device cuda, 54.8 s, per-question avg vector 0.1483 s / rerank 0.2781 s).

## Parse cache is keyed by parser fingerprint

`eval/run_eval.py` no longer refuses a parse cache whenever the commit !=
HEAD. The cache file stores:
- `fingerprint` = sha256(sha256(src/jira_rag/query_parse.py) + "|" +
  model name from GET 127.0.0.1:8092/v1/models) — the validity key;
- `commit` — stored for information only.

`load_parse_cache` computes the current fingerprint (query_parse.py source +
live server model) and refuses on mismatch; `save_parse_cache` writes new
cache files with the current fingerprint. Tested: same fingerprint accepted,
fake fingerprint refused.

## Finding: parser changed G22 and G35 between server sessions

The 27B model is not deterministic across server sessions (KV cache /
server state differences): identical request, different server process,
different outputs.

- G22 ("The optimizer folds 1 + 2 + a into a constant, but not a + 1 + 2.
  Was that fixed?"): previous session `open: false`
  (cache `eval/cache/parse_788dc8b….json`); this session: all-null form.
- G35 ("Problems writing to S3 through the S3A committers (staging,
  magic)."): previous session `text_terms: ["S3", "S3A"]`; this session:
  `text_terms: ["S3"]`.

Stability WITHIN this server session (10 parses each, eval settings — see
the request body in the next section):

- G22: 10/10 one form — all fields null, `text_terms: []`
- G35: 10/10 one form — `text_terms: ["S3"]`, everything else null

So the drift is between server sessions, not within one session. Consequence:
parse caches are only valid for the server session that produced them —
which is why the cache is keyed by a fingerprint that includes the server
model, and why a stale cache must be refused, not silently scored.

Suspect for the cross-session drift: cache_prompt (KV reuse changes batch
numerics) and/or MTP. Not pursued: G22 changed route with identical metrics.

Exact JSON request body the parser sends (`query_parse._call_llm`; eval
settings: temperature 0.0, max_tokens 1024, thinking off). Messages content
truncated to 120 chars:

```json
{
  "model": "default",
  "messages": [
    {
      "role": "system",
      "content": "You are a filter extractor for a Jira issue database (Apache Spark project).\nGiven a user's question about issues, extra..."
    },
    {
      "role": "user",
      "content": "Question: <the question>\nExtract the filter form."
    }
  ],
  "temperature": 0.0,
  "max_tokens": 1024,
  "stream": false,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "schema": {
        "type": "object",
        "properties": {
          "priority":     {"type": ["array", "null"], "items": {"type": "string", "enum": ["Blocker", "Critical", "Major", "Minor", "Trivial"]}},
          "issue_type":   {"type": ["array", "null"], "items": {"type": "string", "enum": ["Bug", "Improvement", "Sub-task", "New Feature", "Task", "Test", "Documentation", "Umbrella", "Question", "Wish", "Dependency upgrade", "Story", "Epic", "Brainstorming", "IT Help", "Request", "Planned Work", "Github Integration", "Technical task", "RTC", "New JIRA Project", "Blog - New Blog Request"]}},
          "open":         {"type": ["boolean", "null"]},
          "resolution":   {"type": ["array", "null"], "items": {"type": "string", "enum": ["Fixed", "Incomplete", "Duplicate", "Won't Fix", "Not A Problem", "Invalid", "Cannot Reproduce", "Done", "Resolved", "Later", "Won't Do", "Not A Bug", "Auto Closed", "Implemented", "Abandoned", "Workaround", "Information Provided", "Works for Me", "Feedback Received"]}},
          "created_from": {"type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
          "created_to":   {"type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
          "text_terms":   {"type": "array", "items": {"type": "string"}}
        },
        "required": ["priority", "issue_type", "open", "resolution", "created_from", "created_to", "text_terms"],
        "additionalProperties": false
      }
    }
  },
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Full untruncated dump: `scratch/parser_stability_out.txt`
(`scratch/parser_stability.py` reproduces it).

## Last changes (this session)

1. **Parse cache keyed by parser fingerprint (`eval/run_eval.py`)** — the
   commit check that invalidated the cache on every commit is replaced by
   `fingerprint` = sha256(query_parse.py source + "|" + server model name);
   `commit` kept only as information. `load_parse_cache` refuses on
   fingerprint mismatch; `save_parse_cache` writes new caches with the
   current fingerprint. Same-fingerprint accepted, fake-fingerprint
   refused (tested).
2. **Parser stability check (`scratch/parser_stability.py`)** — G22 and G35,
   10 parses each in this server session: both fully stable (10/10 one form),
   but both differ from the previous server session's forms (see
   "Finding: parser changed G22 and G35 between server sessions").

## Old last changes (previous session)

1. **Routed m3 reranker in retrieval (`src/jira_rag/retrieve.py`, `search_rerank`)** —

- search_rerank currently accepts cached candidates (cand_keys) from lists.json, which makes the live
  eval a replay of the cache. Remove that path and rerun run_eval.py live.
- G22 got a WHERE (resolution IN ('Fixed')) in lists.json but none in the live run: check whether the
  parser is non-deterministic (5 calls in a row) or lists.json was built from a stale parse file.
- Paste eval/token_check.py output (500 vs 1000 chars) and per-stage timing (parse, vector, rerank).

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
4. **Embedder + reranker on GPU** — `retrieve.init()`/`search_rerank` now
   run on CUDA (conda `dl`, 27B stopped): eval 1253 s → 55 s, same quality
   (see "GPU results" above). `2026-09-28_788dc8b_baseline.json`.

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

## Parked (not planned)

### Retrieval ideas (formerly "Next task (open)")

- Consider whether the routed design (rerank only when no WHERE) is the
  right default, or whether always-rerank + post-filter is better.
- Explore larger N (40, 80) for the candidate pool — d500 timings suggest
  N=40 would be ~16 s/q, still feasible.
- Try cross-encoder distillation: train a smaller reranker on m3 scores
  to get d500-level latency at m3-level accuracy.
- Evaluate on a held-out question set to guard against overfitting to the
  40-question gold.
