# 0003 — Private compose network; the 27B stays on the host behind an API key

Status: accepted (2026-11-18)

## Context
The API needs PostgreSQL and the 27B. The 27B runs on the host GPU as
llama-server and is shared with other tools, so it is not moved into compose.

## Decision
- Compose runs `api` and `db` on their own network; the API reaches the
  database by name (`PGHOST=db`).
- The API reaches the 27B as `host.docker.internal:8092`
  (`extra_hosts: host-gateway`). llama-server listens on 0.0.0.0 and requires a
  key (`--api-key-file`, file mode 0600); every client sends
  `Authorization: Bearer $LLAMA_API_KEY` through one shared helper,
  `query_parse.llm_headers()`.
- Host ports are bound to 127.0.0.1 only: API 8000, database 5433.

## Alternatives considered
- **`network_mode: host`:** simplest on Linux, but it behaves differently on
  Docker Desktop (macOS/Windows), the container sees every port on the host,
  and ports cannot be remapped. Rejected.
- **llama-server on 0.0.0.0 without a key:** any device on the LAN could use
  the model. Rejected.
- **27B in its own container:** the cleanest long-term option; needs a GPU
  container and moving the other tools to it. Deferred.

## Consequences
- The 27B is usable from other devices on the LAN, but only with the key.
- The database is a separate copy from the development database; data is loaded
  once (README section 9).
