#!/usr/bin/env python3
"""Embed jira.issues into jira.issue_chunks using Qwen/Qwen3-Embedding-4B (bf16, local).

Strategies:
  summary_only  content = summary
  summary_desc  content = summary + "\n\n" + description  (description omitted when null)

Matryoshka: the model emits the full 2560-dim vector; it is truncated to --dim
and L2-normalized AFTER truncation, then stored in the vector column.

Content longer than 8000 chars is truncated (no chunking in this script);
the UNTRUNCATED length is recorded in token_count.

Resumable: issues already present in jira.issue_chunks for the given --strategy
are skipped. Commits every batch.
"""
import argparse, sys, time
import psycopg2
from psycopg2.extras import execute_batch
import torch
from transformers import AutoModel, AutoTokenizer

DSN = "host=localhost port=5432 dbname=dedb user=deuser password=depassword"
MODEL = "Qwen/Qwen3-Embedding-4B"
FULL_DIM = 2560        # native output dimension of Qwen3-Embedding-4B
MAX_CHARS = 8000       # content truncated above this (untruncated length kept in token_count)
MAX_TOKENS = 32768     # tokenizer max_length (model's native context)
STRATEGIES = ("summary_only", "summary_desc")

UPSERT = """
INSERT INTO jira.issue_chunks (issue_key, strategy, chunk_index, content, token_count,
    embedding, embedded_at)
VALUES (%(key)s, %(strategy)s, 0, %(content)s, %(token_count)s, %(embedding)s::vector, now())
ON CONFLICT (issue_key, strategy, chunk_index) DO UPDATE SET
    content=EXCLUDED.content,
    token_count=EXCLUDED.token_count,
    embedding=EXCLUDED.embedding,
    embedded_at=now();
"""

SELECT_REMAINING = """
SELECT issue_key, summary, description
FROM jira.issues
WHERE issue_key NOT IN (
    SELECT issue_key FROM jira.issue_chunks WHERE strategy = %s)
ORDER BY issue_id
"""

def last_token_pool(last_hidden, attention_mask):
    """Pool the last non-padding token (right padding for left-padded tokenizer)."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0]), seq_len]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = all remaining issues")
    ap.add_argument("--dim", type=int, default=1024,
        help="stored (truncated) dimension; must match jira.issue_chunks.embedding")
    args = ap.parse_args()

    if args.dim > FULL_DIM:
        print(f"error: --dim={args.dim} exceeds native {FULL_DIM} dims", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("error: no CUDA device available (the GPU must be free)", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(DSN)
    cur = conn.cursor()

    # Validate --dim against the actual column dimension before loading the model.
    cur.execute("""
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = 'jira.issue_chunks'::regclass
          AND attname = 'embedding' AND NOT attisdropped""")
    col_dim = cur.fetchone()[0]
    if col_dim != args.dim:
        print(f"error: jira.issue_chunks.embedding is vector({col_dim}) but --dim={args.dim}; "
              f"use --dim {col_dim} or alter the column", file=sys.stderr)
        conn.close()
        sys.exit(1)

    cur.execute("SELECT count(*) FROM jira.issue_chunks WHERE strategy=%s", (args.strategy,))
    already = cur.fetchone()[0]
    print(f"already embedded for '{args.strategy}': {already} (will skip)", flush=True)

    # Full read of the work list BEFORE loading the model: the per-batch commit
    # would destroy a pending SELECT result, and an empty list must exit clean.
    sql = SELECT_REMAINING
    params = [args.strategy]
    if args.limit > 0:
        sql += "LIMIT %s"
        params.append(args.limit)
    cur.execute(sql, params)
    worklist = cur.fetchall()
    cur.close()

    if not worklist:
        print(f"nothing to embed for strategy '{args.strategy}': all rows already present",
              flush=True)
        conn.close()
        sys.exit(0)

    print(f"loading {MODEL} (bf16) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16,
                                      trust_remote_code=True).cuda()
    model.eval()
    print("model loaded", flush=True)

    @torch.no_grad()
    def embed(texts):
        b = tokenizer(texts, return_tensors="pt", padding=True,
                      max_length=MAX_TOKENS, truncation=True)
        b = {k: v.cuda() for k, v in b.items()}
        out = model(**b)
        e = last_token_pool(out.last_hidden_state, b["attention_mask"])
        del b, out
        e = e.float()[:, :args.dim]                    # Matryoshka truncation
        e = e / torch.linalg.norm(e, dim=1, keepdim=True)  # normalize AFTER truncation
        return e.cpu().tolist()

    ins = conn.cursor()                # separate cursor for the batch inserts

    rows = truncated = 0
    t0 = time.monotonic()
    batch_no = 0
    for i in range(0, len(worklist), args.batch_size):
        chunk = worklist[i:i + args.batch_size]
        contents, token_counts, keys = [], [], []
        for key, summary, description in chunk:
            summary = summary or ""
            if args.strategy == "summary_only":
                content = summary
            else:
                content = summary if description is None else summary + "\n\n" + description
            token_counts.append(len(content))          # untruncated length
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS]
                truncated += 1
            contents.append(content)
            keys.append(key)
        embs = embed(contents)
        execute_batch(ins, UPSERT, [
            dict(key=k, strategy=args.strategy, content=c,
                 token_count=t, embedding=str(e))
            for k, c, t, e in zip(keys, contents, token_counts, embs)
        ], page_size=100)
        conn.commit()
        rows += len(chunk)
        batch_no += 1
        el = time.monotonic() - t0
        print(f"[batch {batch_no}] {rows} embedded, {truncated} truncated, "
              f"{el:.1f}s elapsed, {rows / el:.2f} docs/s", flush=True)

    total = time.monotonic() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print(f"rows written:   {rows}")
    print(f"rows truncated: {truncated} (content > {MAX_CHARS} chars; untruncated length in token_count)")
    print(f"wall time:      {total:.1f}s")
    print(f"mean rate:      {rows / total:.2f} docs/s")
    print(f"peak VRAM:      {peak_gib:.2f} GiB (torch.cuda.max_memory_allocated)")
    conn.close()

if __name__ == "__main__":
    main()
