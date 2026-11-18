# 0004 — Testing: fake retrieval in CI, real quality in the eval harness

Status: accepted (2026-11-18)

## Context
Real retrieval needs ~10 GB of weights, a populated database and the 27B.
GitHub's CI runners have none of that.

## Decision
- `tests/test_api.py` replaces the `retrieve` module with a fake before
  importing the API, so the tests check only the HTTP layer: routing,
  validation (422), response shape, invented-citation split and the 503 when
  the model is down.
- `tests/test_generate_equivalence.py` requires the API's prompt to be
  byte-identical to the phase-3 eval's, so the eval numbers keep describing `/ask`.
- Real quality is measured by the eval harness (40 gold questions, README
  sections 3 and 4), not by unit tests.
- CI (`.github/workflows/ci.yml`) runs the tests and also builds the Docker
  image from scratch and validates the compose file.

## Alternatives considered
- **Integration tests with real models in CI:** needs GPU or ~10 GB of weights
  and a database; slow and costly. Rejected.

## Consequences
- 18 tests run in under a second, with no GPU or database.
- The fake can drift from the real `retrieve.search_rerank` signature; the eval
  harness is the guard for that.
