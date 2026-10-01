# jira-rag

A retrieval-augmented generation system over the Apache Spark JIRA: 59,227 issues
ingested from the public Jira REST API into local PostgreSQL + pgvector, queried
end-to-end by a local 27B model on a single 16 GB GPU. Everything runs offline;
no external API is called at query time.

This is a portfolio project: the goal is a working, measured, well-documented
RAG — not a service to run day to day. Scope and limitations are documented in
[NEXT.md](NEXT.md).

## TL;DR

> **PENDING re-measurement.** The numbers below were measured before a
> leak was found and fixed: gold question G07 was a few-shot example in the
> parser prompt. They are kept here, marked, until
> [`eval/remeasure.sh`](eval/remeasure.sh) is re-run on the fixed parser.

| | Value | Status |
|---|---:|---|
| Retrieval recall@10 (40-question dev set) | 0.78 | PENDING |
| Answers citing a correct issue | 39 / 40 | PENDING |
| Citations invented (issue keys not in context) | 0 | PENDING |

- **Measured, not guessed.** Every component change was scored on the same 40
  questions; 6 options were tested and rejected with numbers
  ([Section 6](#6-decisions-and-rejected-options)).
- **On-prem on one 16 GB GPU.** 59,227 issues in PostgreSQL + pgvector, a
  4B embedder, a cross-encoder reranker and a 27B model (llama.cpp); no
  external API at query time.
- **A dev set, not a test set.** The 40 questions were used to tune the
  parser and the retrieval. The numbers show how well the system fits them,
  not how it generalizes ([Limitations](#7-limitations)).

---

## 1. Dataset and infrastructure

All components run locally:

| Component | Choice | Notes |
|---|---|---|
| Database | PostgreSQL 16.15 + pgvector | schema `jira`, see [sql/](sql/) |
| Corpus | 59,227 Apache Spark JIRA issues | `jira.issues`, ingested from the public Jira REST API |
| Embedder | Qwen3-Embedding-4B, 1024-dim, bf16 | stored vectors embedded in bf16 ([embed.py](src/jira_rag/embed.py), log in [notes/findings.md](notes/findings.md)); query embedding loads bf16 first, fp32 only as fallback; runs on CUDA during retrieval |
| Reranker | BAAI/bge-reranker-v2-m3, fp32, `max_length=512` | runs on CUDA during retrieval |
| Parser / generator | Qwen3.8-27B (GGUF: `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`) via llama.cpp `llama-server` on port 8092 | runs alone, in its own phase |
| GPU | 1× NVIDIA RTX 5060 Ti, 16 GB | the 27B + embedder + reranker need ~25 GB together, so they never run simultaneously |

Database layout ([sql/001_schema.sql](sql/001_schema.sql)):

- `jira.issues` — one row per issue: `issue_key`, `project`, `summary`,
  `description`, `issue_type`, `priority`, `status`, `resolution`, `created`,
  `updated`, `resolutiondate`, `raw` (full API JSON), `ingested_at`,
  `date_suspect` (unused: no code in this repo sets or reads it).
- `jira.issue_chunks` — chunked text with embeddings: `chunk_id`,
  `issue_key`, `strategy`, `chunk_index` (always 0: one chunk per issue,
  content cut at 8,000 characters), `content`, `token_count` (despite the
  name, the character count of the untruncated content),
  `embedding vector(1024)`, `embedded_at`; unique on
  `(issue_key, strategy, chunk_index)`; FK to `jira.issues` with
  `ON DELETE CASCADE`.

Supporting SQL:

- [sql/002_fts.sql](sql/002_fts.sql) — generated `fts tsvector` column
  (`summary=A || description=B`) plus a GIN index, for keyword ranking via
  `ts_rank_cd`.
- [sql/003_lexeme_df.sql](sql/003_lexeme_df.sql) — per-lexeme document
  frequencies from `ts_stat`; the hybrid search keeps only rare question
  lexemes (`ndoc < 2%` of issues) in the keyword query so common words don't
  dilute the fusion.

Python dependencies: see [requirements.txt](requirements.txt).

## 2. Architecture

<!-- source: NEXT.md ("Scope", "Rule: retrieval experiments on GPU, 27B stopped"), src/jira_rag/ module layout -->

The system runs in three phases; the phase separation is a deliberate design
constraint, not an accident:

```
            +------------------------------------------+
            |   Phase 1: parse (27B llama-server)      |
            |   question -> structured filter form    |
            +-------------------+----------------------+
                                |  form (JSON)
            +-------------------v----------------------+
            |   Phase 2: retrieval (embedder +         |
            |   reranker on GPU, 27B STOPPED)          |
            |   form -> SQL WHERE -> vector top-20     |
            |   -> bge-reranker-v2-m3 -> RRF blend     |
            +-------------------+----------------------+
                                |  top-10 issue keys
            +-------------------v----------------------+
            |   Phase 3: generation (27B llama-server) |
            |   fetches descriptions from Postgres,   |
            |   top-10 + question -> answer w/        |
            |   [SPARK-xxxxx] citations               |
            +------------------------------------------+
```

1. **Parse** (`src/jira_rag/query_parse.py`): the 27B extracts a structured
   filter form from the question using JSON-schema mode
   (`priority`, `issue_type`, `open`, `resolution`, `created_from`,
   `created_to`, `text_terms`).
2. **Retrieval** (`src/jira_rag/retrieve.py`): the form becomes a SQL `WHERE`
   clause; vector search (Qwen3-Embedding-4B) takes the top-20 candidates;
   the bge-reranker-v2-m3 reranks them; final ranking is an RRF fusion
   (k=60) of vector rank and rerank rank. The reranker is used **only when
   the parser produced no `WHERE` clause** — filtered questions return the
   vector order unchanged.
3. **Generation** (phase-3 eval scripts in [eval/](eval/)): the 27B answers
   from the top-10 issues with inline `[SPARK-xxxxx]` citations.

Because the 27B, embedder and reranker cannot co-fit in 16 GB, the phases run
in turns: retrieval experiments always run with the 27B server stopped, and
parse runs require it up. Reruns that only need search/rerank use a parse
cache keyed by a parser fingerprint (see Limitations).

Code modules in [src/jira_rag/](src/jira_rag/): `ingest.py`, `embed.py`,
`query_parse.py`, `retrieve.py` (one shared vector search, `vector_rows`),
`fusion.py` (the RRF blend), `metrics.py`, and `experimental.py` (the
rejected keyword strategies, still scored by `run_eval.py`).

Date filters are UTC days: `created` is `timestamptz`, and the parser's
`created_from` / `created_to` become `('YYYY-MM-DD'::timestamp AT TIME ZONE 'UTC')`
bounds, so results do not depend on the server's `TimeZone` setting.

## 3. Retrieval results

<!-- source: eval/results/2026-09-28_e4f9511_baseline.json (CPU), eval/results/2026-09-28_788dc8b_baseline.json (GPU) -->

Evaluation: 40 gold questions from [eval/gold.jsonl](eval/gold.jsonl)
(15 lookup, 15 topic, 10 filtered), each with one or more expected issues
(104 gold keys total). `k=10` candidates, embedder dim 1024.

> **Dev-set numbers, PENDING re-measurement.** The same 40 questions were
> used to choose every strategy and to tune the parser, and one of them (G07)
> was a few-shot example in the parser prompt when these runs were made. The
> tables below are kept as recorded; they are in-sample and will be replaced
> by the output of `eval/remeasure.sh`.

Strategy progression (recall@10 and MRR, overall and by type):

| Strategy (cumulative) | Scope | n | recall@5 | recall@10 | MRR |
|---|---|---:|---:|---:|---:|
| `summary_only` — vector on summary text | overall | 40 | 0.3854 | 0.5037 | 0.4709 |
| `summary_desc` — vector on summary + description | overall | 40 | 0.4571 | 0.5535 | 0.5147 |
| `summary_desc_filtered` — + parser-driven SQL `WHERE` | overall | 40 | 0.6519 | 0.7390 | 0.7202 |
| `summary_desc_rerank` — + bge-reranker-v2-m3 (routed) | overall | 40 | 0.7115 | 0.7790 | 0.7567 |

By type, `summary_desc_rerank` (chosen configuration):

| Type | n | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|
| lookup | 15 | 0.9333 | 0.9333 | 0.7522 |
| topic | 15 | 0.3667 | 0.5300 | 0.5990 |
| filtered | 10 | 0.8958 | 0.9208 | 1.0000 |

<!-- source: eval/results/2026-09-28_788dc8b_baseline.json, summary_desc_rerank (overall + by_type) -->

Key observations:

Note: an earlier offline replay that reused cached candidate lists scored MRR 0.7701; the numbers above are the live run.

- The parser-driven `WHERE` clause is the biggest single gain: recall@10
  0.5535 → 0.7390, MRR 0.5147 → 0.7202.
- The reranker adds recall@10 0.7390 → 0.7790 and MRR 0.7202 → 0.7567
  (+0.0400 / +0.0365). It is routed: only applied when there is no `WHERE`
  clause, which is why filtered-type MRR stays perfect (1.0000) — the routed
  path returns the vector order unchanged.
- Rerank vs. filtered (per question, same run): 5 worse, 9 better, 26 equal by MRR.
- Topic questions remain the weak spot (recall@10 0.5300): they have no
  structured filters, so they rely entirely on dense retrieval + rerank.

Chosen configuration (`summary_desc_rerank`): m3 fp32, 6 threads,
N=20 candidates, RRF k=60 blend of vector rank and rerank rank (1-based),
`RERANK_DESC_CHARS=500`, `max_length=512`
([src/jira_rag/retrieve.py](src/jira_rag/retrieve.py) `RERANK_*` constants,
[src/jira_rag/fusion.py](src/jira_rag/fusion.py)).

### CPU vs GPU

<!-- source: NEXT.md ("GPU results"), eval/results/2026-09-28_e4f9511_baseline.json, eval/results/2026-09-28_788dc8b_baseline.json -->

Same settings on CPU vs CUDA (`summary_desc_rerank`, 40 questions, fp32 m3,
N=20, RRF k=60):

| Metric | CPU (e4f9511) | GPU (788dc8b) |
|---|---:|---:|
| rerank recall@10 | 0.7790 | 0.7790 |
| rerank MRR | 0.7576 | 0.7567 |
| eval wall time | 1253 s | 55 s |

Quality is identical (MRR delta 0.0009, within run-to-run noise); the win is
~23× wall time. GPU per-question averages: vector 0.1483 s, rerank 0.2781 s.

## 4. Generation results

<!-- source: eval/results/answers_3d5a427.json (commit 3d5a427), eval/results/grade_3d5a427.json (graded at df5a31e) -->

> **Dev-set numbers, PENDING re-measurement** (same caveat as Section 3).

Generation settings: temperature 0.0, `max_tokens` 1024, thinking off,
`desc_chars` 1500. Each of the 40 questions received its top-10 issues with
summaries plus descriptions cut to 1500 chars (avg 6,212 chars of context per
question).

Overall:

| Metric | Value |
|---|---:|
| n | 40 |
| hit (answer cites a gold issue) | 39 / 40 (0.975) |
| gold-in-context ceiling | 39 / 40 (only G19 missed retrieval) |
| hit on ceiling | 39 / 39 (1.000) |
| invented citations (keys not in context) | 0 |
| `from_text` citations (key absent from context keys, found inside a context's issue text) | 1 |
| answers with no citation | 0 |
| avg citation ratio (cited / context keys) | 0.503 |
| questions citing every context | 6 / 40 |
| avg seconds / question | 7.48 |
| avg completion tokens | 199.6 |
| total generation time | 299.1 s (~5.0 min) |
| total prompt tokens | 106,165 |
| total completion tokens | 7,983 |

By type:

| Type | n | hit | cite ratio | cited all | avg s | avg completion tokens |
|---|---:|---:|---:|---:|---:|---:|
| filtered | 10 | 10/10 | 0.740 | 6/10 | 4.12 | 96.9 |
| lookup | 15 | 14/15 | 0.300 | 0/15 | 7.02 | 154.9 |
| topic | 15 | 15/15 | 0.547 | 0/15 | 10.17 | 312.7 |

Interpretation:

- **Zero invented citations.** The model never cites an issue key that was
  not among the provided contexts; the single exception is a `from_text`
  citation (see below).
- **The only failure is a retrieval miss, not a generation failure.** G19
  (missing R package for SparkR CRAN checks) is the only question whose gold
  `SPARK-17191` never reached the contexts, so it was unreachable — the model
  honestly said so and cited 5 legitimate in-context issues instead. On the
  39 questions where the gold was in context, every one hit.
- **Citation density tracks question type.** Filtered (list) questions cite
  nearly all contexts (avg 0.74) and are the fastest (4.12 s) and shortest
  (96.9 completion tokens). Lookups cite ~30% of contexts — a single key is
  usually enough (e.g. G09 answers "3.9.6 [SPARK-59652]" from one context out
  of ten). Topic answers are the longest (312.7 tokens) and the slowest
  (10.17 s) because they synthesize several causes.
- The one `from_text` citation: G36 cited `SPARK-31475`, which was not one of
  the context issue keys but appears inside the description of context issue
  `SPARK-33933` ("introduced in SPARK-31475"). That is why the citation
  grader counts `from_text` separately: the key is grounded in the provided
  text, not invented.

## 5. Examples

Four real cases, reproduced verbatim (questions and model answers unedited),
with the retrieved issue lists and grading verdicts — see
[examples.md](examples.md):

| Case | Type | Question (short) | Outcome |
|---|---|---|---|
| [G01](examples.md) | topic | OOM during shuffle — known causes? | hit; 7 causes, cites 8/10 contexts; one hedged sentence goes beyond the source (see Limitations) |
| [G09](examples.md) | lookup | Newest ZooKeeper version Spark is moving to? | hit; answer is one sentence citing one context (cite ratio 0.1) |
| [G23](examples.md) | filtered | Blocker bugs filed since June 2026, still open? | hit; lists exactly the 2 matching issues, cites 2/2 |
| [G19](examples.md) | lookup | SparkR CRAN checks fail: missing R package? | miss in retrieval (gold never in context); model honestly says the provided issues don't identify it |

## 6. Decisions and rejected options

<!-- source: NEXT.md ("Chosen configuration", "Rejected configurations", "Rule: retrieval experiments on GPU, 27B stopped") -->

### Chosen

| Decision | Rationale |
|---|---|
| Rerank only when there is no `WHERE` clause (routed) | Filtered questions already have a strong deterministic filter; reranking them gains nothing and filtered MRR stays 1.0000 |
| RRF blend of vector rank + rerank rank (k=60) | Pure rerank over-weights short summaries and loses on topic questions |
| bge-reranker-v2-m3, fp32, max_length=512 | Best tested quality; N=20 candidates |
| Run 27B / embedder / reranker in turns | They need ~25 GB together; 16 GB is fixed |
| Parse cache keyed by parser fingerprint | The 27B is not deterministic across server sessions; see Limitations |

### Rejected (tested)

| Option | Result |
|---|---|
| Keyword hybrid (`summary_desc_hybrid`, RRF of vector + FTS `websearch_to_tsquery`) | Overall MRR 0.6475–0.6496, worse than the filtered baseline 0.7202; the keyword list adds noise on topic questions |
| Tail rescue (`summary_desc_rescue`, m=1): fill top-10 tail with keyword hits | Matches baseline exactly (0 worse, 0 better); no gain |
| Pure rerank (no RRF blend) | Worse than blended RRF on topic questions |
| bge-reranker-base (smaller model) | Lower quality than m3 (d500 scores diverge by ~0.05 on average, flipping several top-10 slots) |
| int8 torch quantization of m3 | Scores diverge from fp32 by ~0.01–0.03, flipping 2–3 top-10 slots per question; not worth the accuracy loss |
| ONNX runtime for the reranker | Scores match fp32 torch to ~1e-4, but no throughput gain at batch 20 on CPU (torch 6 threads ≈ ONNX 6 threads) |

Other parked retrieval ideas (larger candidate pool N=40/80, cross-encoder
distillation, held-out question set) are listed under "Parked" in
[NEXT.md](NEXT.md).

The rejected strategies stay in [src/jira_rag/experimental.py](src/jira_rag/experimental.py),
and the one-off scripts behind these comparisons are in
[eval/experiments/](eval/experiments/), as evidence of the process; they are
not part of the measured pipeline. Git history keeps everything else.

## 7. Limitations

<!-- source: NEXT.md, eval/gold.jsonl, eval/results/grade_3d5a427.json, examples.md (G01) -->

- **Small gold set, and it was used during tuning (a dev set).** All 40
  questions in [eval/gold.jsonl](eval/gold.jsonl) were also the set used to
  pick strategies and configurations, and the parser prompt was written
  while looking at them: one gold question (G07) was a verbatim few-shot
  example until it was replaced, and other examples follow the same
  templates as gold questions ("Which blocker bugs filed in … are still
  open?"). The reported numbers are in-sample and overfitting risk is real.
  There is no held-out set; `tests/test_prompt.py` now fails if a gold
  question appears in the prompt.
- **`hit` measures citation, not fidelity.** The grader checks whether the
  answer cites a gold issue key and whether all cited keys are in context. It
  does not check that the prose is true to the source: in G01 one sentence
  ("potentially due to inefficient execution plans..." on SPARK-22438) is the
  model's own hedged speculation, and the citation check does not catch it.
  (The `from_text` citation in G36 *is* grounded — the key appears in a
  context issue's description — which is why it is counted separately from
  invented citations.)
- **The parser is not deterministic across llama-server sessions.** With
  identical requests, a different server process produced different filter
  forms for G22, and one session gave `text_terms: ["S3", "S3A"]` while the other gave `["S3"]` for G35. Within one
  server session the parser is stable (10/10 identical for both). That is why
  the parse cache is keyed by a fingerprint
  (`sha256(query_parse.py source + "|" + server model name)`) instead of the
  git commit, and a stale cache is refused rather than silently scored.
  Suspect causes (KV cache reuse / MTP) are documented but not pursued.
- **No interactive `ask` command by design** — the project's scope is a
  measured RAG, not a day-to-day service (see [NEXT.md](NEXT.md)).
- **16 GB VRAM forces phased execution** — the 27B, embedder and reranker
  never run at the same time; retrieval experiments must stop the 27B server.

## 8. What I would do differently

- **Write the held-out set first.** 15–20 questions nobody tunes on, written
  before the first experiment, and a check that no eval question enters a
  prompt. Both came late here, so every number above is a dev-set number.
- **Make the eval fail loudly from day one.** A stopped 27B server used to
  turn into an all-null parse that was scored as "no filters"; a stale
  parse cache was once scored silently (`lists.json`). Both now stop the run.
- **Measure faithfulness, not only citations.** The grader checks that a
  cited key is real and in context, not that the sentence is true to it
  (G01). A small manual audit or an LLM judge would close that gap.
- **One retrieval code path from the start.** The vector SQL was copied into
  four functions before it was shared, and the rejected strategies drifted
  (0-based RRF ranks, a sort that was not deterministic, a `KeyError` path).
- **Pin what makes runs comparable.** The parser drifts between llama-server
  sessions; recording the server build and settings with every result, and
  pinning Python dependencies, would have explained that sooner.
- **Tests and CI in the first commit**, not after the results.

## 9. How to reproduce

<!-- source: requirements.txt, sql/, NEXT.md ("Rule: retrieval experiments on GPU, 27B stopped") -->

1. **Prerequisites:** PostgreSQL with pgvector; NVIDIA GPU with CUDA (tested:
   RTX 5060 Ti, 16 GB, `torch==2.14.0+cu130`); llama.cpp `llama-server`
   serving `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` on `127.0.0.1:8092`;
   Qwen3-Embedding-4B and bge-reranker-v2-m3 (Hugging Face).
2. **Dependencies:** `pip install -r requirements.txt` (includes the CUDA
   torch wheel).
3. **Database:** `CREATE EXTENSION vector;` (the schema dump does not create
   it), then create the schema from [sql/001_schema.sql](sql/001_schema.sql),
   then apply [sql/002_fts.sql](sql/002_fts.sql) (FTS column + GIN index) and
   [sql/003_lexeme_df.sql](sql/003_lexeme_df.sql) (lexeme document
   frequencies). The corpus is 59,227 Apache Spark JIRA issues, ingested via
   [src/jira_rag/ingest.py](src/jira_rag/ingest.py).
4. **Embeddings:** chunk and embed with [src/jira_rag/embed.py](src/jira_rag/embed.py)
   (Qwen3-Embedding-4B, 1024-dim).
5. **All evals, one command:** `bash eval/remeasure.sh` runs the three
   phases on the current commit and waits for llama-server to be started or
   stopped between them (or runs `LLAMA_START_CMD` / `LLAMA_STOP_CMD`):
    1. 27B up: `eval/run_parse_eval.py` (parser filters vs gold) and
       `eval/run_eval.py --write-parse-cache` (one fingerprinted parse of
       all 40 questions);
    2. 27B **stopped**, embedder + reranker on GPU: `eval/run_eval.py
       --parse-cache` (every retrieval strategy) and `eval/build_contexts.py`
       (top-10 per question, descriptions cut to 1500 chars);
    3. 27B up: `eval/generate_answers.py` (temperature 0.0, `max_tokens`
       1024) and `eval/grade_answers.py` (hit, invented citations,
       `from_text`, no citation, seconds/tokens — overall and by type).

   Any parse error stops the run. Outputs are named after the commit in
   `eval/results/` and `eval/cache/` and are never overwritten.
6. **Tests:** `pip install -r requirements-dev.txt && python -m pytest tests`
   (no GPU, model or database; also run by GitHub Actions).

Results referenced in this README:

| Artifact | File |
|---|---|
| Retrieval metrics (CPU / GPU), PENDING re-measurement | [eval/results/2026-09-28_e4f9511_baseline.json](eval/results/2026-09-28_e4f9511_baseline.json), [eval/results/2026-09-28_788dc8b_baseline.json](eval/results/2026-09-28_788dc8b_baseline.json) |
| Generated answers (commit `3d5a427`), PENDING re-measurement | [eval/results/answers_3d5a427.json](eval/results/answers_3d5a427.json) |
| Graded generation results | [eval/results/grade_3d5a427.json](eval/results/grade_3d5a427.json) |
| Real cases | [examples.md](examples.md) |
| Gold set (40 questions) | [eval/gold.jsonl](eval/gold.jsonl) |
| Open next steps | [NEXT.md](NEXT.md) |
