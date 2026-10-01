# NEXT.md

## Scope

- **Status: CLOSED.** jira-rag is a portfolio piece: the goal is a working,
  measured and well-documented RAG, not a service to run day to day. No new
  functionality is added; remaining work is honesty of the numbers, cleanup
  and presentation.
- OUT of scope: a live `ask` command, and fitting the 27B + embedder +
  reranker together in 16 GB VRAM (they need ~25 GB). Hardware is fixed; no
  model swaps for that purpose.
- The system runs in phases, and that is the documented design: (1) parse
  with the 27B, (2) retrieval on GPU with the 27B stopped, (3) generation
  with the 27B.

## Open: re-measure without the G07 leak

Gold question G07 was a verbatim few-shot example in the parser prompt from
`8d3b652` until it was replaced (`9b10d86`). Every published number was
measured with the leak, so the README marks them PENDING.

- [ ] With the GPU free: `bash eval/remeasure.sh` (parse, retrieval,
  generation; waits for llama-server to be started/stopped between phases).
- [ ] Replace the PENDING numbers in README.md (TL;DR, Sections 3–4, results
  table) and examples.md with the new `eval/results/*_<commit>*` files.
- [ ] Confirm `SHOW timezone;` is `UTC` on de_postgres. Date filters are now
  explicit UTC; on a UTC server the results are unchanged.
- [ ] Optional: 15–20 new held-out questions nobody tuned on, scored once.

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

A parse cache is written by `eval/run_eval.py --write-parse-cache PATH`
(27B up). A cache or live parse holding any `parse_error` form is refused,
and `query_parse.parse()` raises `ParserUnavailable` when the server is
down instead of returning the all-null form.

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

The full untruncated dump and the script that produced it
(`scratch/parser_stability.py`) were local scratch files and are not in the
repo. The prompt has changed since (G07 example replaced), so the request
body above is historical.

## Chosen configuration

`summary_desc_rerank` (m3 fp32, 6 threads, N=20, RRF k=60 via
`fusion.blend_vector_rerank`, routed, blended, 1-based ranks,
DESC_CHARS=500, max_length=512).

Live run (GPU, `2026-09-28_788dc8b_baseline.json`, PENDING re-measurement):
overall r@10 0.7790, MRR 0.7567 (filtered baseline: 0.7390 / 0.7202);
lookup MRR 0.7522, topic 0.5990, filtered 1.0000; 5 worse, 9 better per MRR.

An earlier offline replay that reused cached candidate lists
(`eval/cache/lists.json`, `eval/experiments/rerank_d500.py`) gave MRR
0.7701; the live number above replaces it.

Clean timings on CPU (6 threads, fp32 m3, `eval/experiments/clean_timing.py`):
N=20 19.97 s/q, N=10 8.63 s/q; with d500, N=20 7.96 s/q, N=10 3.58 s/q.

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

Code for the hybrid and rescue strategies: `src/jira_rag/experimental.py`;
the offline sweeps: `eval/experiments/`.

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
