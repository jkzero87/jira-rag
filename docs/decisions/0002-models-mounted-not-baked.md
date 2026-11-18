# 0002 — Model weights are mounted, not baked into the image

Status: accepted (2026-11-18)

## Context
The embedder and reranker weights are ~9.7 GB. The API image without them is
2.03 GB on disk (437 MB compressed).

## Decision
- Weights live in `./.hf` on the host (git-ignored) and are mounted at
  `/models` (`HF_HOME=/models`).
- `HF_HUB_OFFLINE=1`: the container never calls the Hugging Face Hub at runtime.
- Runtime-only ENV values are set after `pip install` in the Dockerfile, so
  changing them does not rebuild the dependency layer (no-change rebuild: 0.46 s).
- The container runs as a non-root user (uid 1000, same as the host user, so the
  mounted cache stays readable); the code is copied with `--chown`.

## Alternatives considered
- **Bake weights into the image:** ~12 GB image, slow to build, push and pull. Rejected.
- **Download on first start:** needs network at runtime and makes the first start
  slow and fragile. Rejected.

## Consequences
- A fresh clone needs a one-time download into `./.hf` (README section 9).
- Changing permissions of a copied file also invalidates Docker's cache
  (observed: a chmod of requirements.txt re-ran pip install, 84 s).
