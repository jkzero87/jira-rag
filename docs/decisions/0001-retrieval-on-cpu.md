# 0001 — Retrieval runs on CPU inside the API container

Status: accepted (2026-11-18)

## Context
One 16 GB GPU. The Qwen3.8-27B llama-server needs almost all of it, and it must
stay up: the API's parser and answer generation use it, and so does the local
coding agent. The embedder (Qwen3-Embedding-4B) and the reranker
(bge-reranker-v2-m3) need several GB more. They cannot share the card.

## Decision
The API container runs embedder + reranker on CPU (`JIRA_RAG_DEVICE=cpu`).
The device is an environment variable, so the same code runs on CUDA when the
27B is stopped.

## Alternatives considered
- **Phased execution (GPU retrieval, 27B off):** fast search, but `/ask` cannot
  work while retrieval runs. Rejected: the system would not be usable end to end.
- **GPU inside the container (NVIDIA toolkit + CUDA image):** does not create
  VRAM, and adds several GB to the image. Rejected.
- **Smaller models:** would change the measured retrieval quality. Rejected.

## Consequences
- `/search` takes ~16–23 s and `/ask` ~40 s per question on this machine
  (README section 9). Retrieval on CUDA is ~23x faster (eval wall time
  1253 s CPU vs 55 s GPU, README section 3).
- Quality is unchanged: the API returned the same 10 keys with the same scores
  as the CPU eval run `e4f9511` for gold question G06.
- The image installs CPU-only torch; the CUDA pin in requirements.txt is for the host venv.
