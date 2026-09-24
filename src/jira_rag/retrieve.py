#!/usr/bin/env python3
"""Retrieve jira.issue_chunks with Qwen3-Embedding-4B on CPU.

Queries are embedded with the Qwen3-Embedding retrieval instruction
prefix, then the SAME tokenizer settings (padding_side="left"),
last-token pooling, Matryoshka truncation and L2 normalization as
embed.py (constants and the pooling helper are imported from embed.py,
not copied). Search is plain pgvector cosine distance; no index, no filters.
"""
import sys
import time
import psycopg2
import torch
from transformers import AutoModel, AutoTokenizer

from embed import DSN, MODEL, MAX_TOKENS, last_token_pool

QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
DIM = 1024  # stored Matryoshka dimension; must match jira.issue_chunks.embedding

_tokenizer = None
_model = None
_dtype = None


def init():
    """Load tokenizer + model on CPU (lazy). Tries bfloat16, falls back to float32."""
    global _tokenizer, _model, _dtype
    if _model is not None:
        return _tokenizer, _model
    t0 = time.monotonic()
    print("loading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", trust_remote_code=True)
    model = None
    for dtype, label in ((torch.bfloat16, "bfloat16"), (torch.float32, "float32")):
        try:
            print(f"loading {MODEL} on CPU ({label}) ...", flush=True)
            cand = AutoModel.from_pretrained(MODEL, dtype=dtype, trust_remote_code=True,
                                             attn_implementation="sdpa")
            with torch.no_grad():
                cand(torch.zeros(1, 4, dtype=torch.long))  # CPU smoke test
            model = cand
            break
        except Exception as exc:
            print(f"{label} on CPU failed ({type(exc).__name__}: {exc}); "
                  "trying next dtype", file=sys.stderr)
    if model is None:
        sys.exit("error: could not load model on CPU")
    print(f"model loaded on CPU ({label}) in {time.monotonic() - t0:.1f}s", flush=True)
    _tokenizer, _model, _dtype = tokenizer, model, label
    return tokenizer, model


def embed_query(q, dim=DIM):
    """Embed one query -> list of `dim` floats.

    Prefixes QUERY_PREFIX, tokenizes (truncated at MAX_TOKENS), left-pads,
    last-token pools, truncates to `dim` and L2-normalizes after truncation
    (identical pipeline to embed.py).
    """
    tokenizer, model = init()
    enc = tokenizer(QUERY_PREFIX + q, max_length=MAX_TOKENS, truncation=True)
    b = tokenizer.pad({"input_ids": [enc["input_ids"]]}, return_tensors="pt")
    with torch.no_grad():
        out = model(**b)
    e = last_token_pool(out.last_hidden_state, b["attention_mask"])
    del out
    e = e.float()[:, :dim]
    e = e / torch.linalg.norm(e, dim=1, keepdim=True)
    return e[0].tolist()


SEARCH_SQL = """
SELECT issue_key, embedding <=> %s::vector AS distance
FROM jira.issue_chunks
WHERE strategy = %s
ORDER BY distance
LIMIT %s
"""


def search_vec(qvec, strategy, k):
    """Cosine search over an already-embedded query (list of floats)."""
    conn = psycopg2.connect(DSN)
    try:
        cur = conn.cursor()
        cur.execute(SEARCH_SQL, (str(qvec), strategy, k))
        return cur.fetchall()  # [(issue_key, distance)], nearest first
    finally:
        conn.close()


def search(q, strategy, k):
    """Embed the query, then cosine-search jira.issue_chunks (top k)."""
    return search_vec(embed_query(q), strategy, k)


def main():
    for key, dist in search("Upgrade ZooKeeper", "summary_only", 5):
        print(key, round(float(dist), 6))


if __name__ == "__main__":
    main()
