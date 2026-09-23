# Findings — measured facts from the two logs

Sources (read before relocation): `embed_summary_only.log`, `ingest_spark.log` (repo root).
One bullet per fact; the exact log line each fact comes from is quoted. No inference is added.

## embed_summary_only.log

- 232 rows for `summary_only` already existed and were skipped at start:
  `already embedded for 'summary_only': 232 (will skip)`
- The model is loaded in bf16:
  `loading Qwen/Qwen3-Embedding-4B (bf16) ...`
- Unauthenticated Hugging Face Hub warning:
  `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.`
- Weights finished loading (398/398) at 9503.31 it/s:
  `Loading weights: 100%|██████████| 398/398 [00:00<00:00, 9503.31it/s]`
- Model load confirmed:
  `model loaded`
- First batch: 32 docs, 36.48 docs/s in 0.9s:
  `[batch 1] 32 embedded, 0 truncated, 0.9s elapsed, 36.48 docs/s`
- Last batch was batch 1844 with 58995 embedded at 169.26 docs/s (348.6s elapsed):
  `[batch 1844] 58995 embedded, 0 truncated, 348.6s elapsed, 169.26 docs/s`
- Rows written: 58995:
  `rows written:   58995`
- Rows truncated: 0:
  `rows truncated: 0 (content > 8000 chars; untruncated length in token_count)`
- Wall time: 348.6s:
  `wall time:      348.6s`
- Mean rate: 169.26 docs/s:
  `mean rate:      169.26 docs/s`
- Peak VRAM: 8.24 GiB:
  `peak VRAM:      8.24 GiB (torch.cuda.max_memory_allocated)`

## ingest_spark.log

- Progress is printed every 100 upserted rows, from 100 up to the full 59227:
  `100/59227 upserted (startAt=100)` … `59227/59227 upserted (startAt=59227)`
- A `503 Service Unavailable` HTTP error occurred at `startAt=21100`:
  `HTTP 503 at startAt=21100: <!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">`
  - `503 Service Unavailable`:
    `<title>503 Service Unavailable</title>`
  - `Service Unavailable`:
    `<h1>Service Unavailable</h1>`
- Progress lines continue past the 503 up to completion:
  `59227/59227 upserted (startAt=59227)`
- Final row count for SPARK: 59227:
  `rows in db for SPARK: 59227`

## Notes on the files themselves

- No timestamps or calendar dates appear in either log (grep for date/time patterns returns nothing).
- `ingest_spark.log` contains 6654 NUL bytes (carried in the captured 503 response body), which is why `grep` reports it as a binary file.

## summary_desc embedding (2026-09-23)

- 59,227 rows; final run 54,941 texts in 3201.2 s, mean 17.16 docs/s, peak VRAM 11.04 GiB
- 913 texts over 8,000 chars (truncated), about 1.5% of the corpus
- fixed-size batches of 8 ran at 4.8 docs/s and ran out of memory at 3,696 rows
- bug found: sorting by characters but budgeting by tokens let a batch reach ~31k padded tokens against an 8,192 budget; fixed by sorting by token length and budgeting on the longest text in the batch
- with the fix: 300-doc test at 10.47 docs/s, 9.25 GiB peak with an 8,192 budget
- most of the run time went to the longest texts at the end of the queue
