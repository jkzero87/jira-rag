#!/usr/bin/env bash
# Smoke test for the jira-rag API stack: health, auth, search, ask.
# Reads JIRA_RAG_API_KEY from .env and never prints it.
set -euo pipefail

BASE="http://127.0.0.1:8000"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JIRA_RAG_API_KEY="$(sed -n 's/^JIRA_RAG_API_KEY=//p' "$ROOT/.env" | head -n 1)"
: "${JIRA_RAG_API_KEY:?JIRA_RAG_API_KEY not set in $ROOT/.env}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

failures=0
report() { # $1=description  $2=0/1 (ok/fail)
  if [[ "$2" -eq 0 ]]; then
    echo "PASS - $1"
  else
    echo "FAIL - $1"
    failures=$((failures + 1))
  fi
}
json_len() { # $1=file  $2=list field -> length (0 if unparseable)
  python3 -c '
import json, sys
try:
    print(len(json.load(open(sys.argv[1]))[sys.argv[2]]))
except Exception:
    print(0)
' "$1" "$2" || echo 0
}

# a) GET /v1/health -> 200
code=$(curl -s -o "$tmp/a" -w '%{http_code}' --max-time 15 "$BASE/v1/health" || true)
if [[ "$code" == "200" ]]; then report "GET /v1/health -> 200" 0
else report "GET /v1/health -> 200 (got ${code:-no-response})" 1; fi

# b) POST /v1/search without key -> 401
code=$(curl -s -o "$tmp/b" -w '%{http_code}' --max-time 15 -X POST "$BASE/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"question": "shuffle spill to disk", "k": 5}' || true)
if [[ "$code" == "401" ]]; then report "POST /v1/search without key -> 401" 0
else report "POST /v1/search without key -> 401 (got ${code:-no-response})" 1; fi

# c) POST /v1/search with key -> 200 and at least 1 hit
code=$(curl -s -o "$tmp/c" -w '%{http_code}' --max-time 180 -X POST "$BASE/v1/search" \
  -H 'Content-Type: application/json' -H "X-API-Key: $JIRA_RAG_API_KEY" \
  -d '{"question": "shuffle spill to disk", "k": 5}' || true)
hits=$(json_len "$tmp/c" hits)
if [[ "$code" == "200" && "$hits" -ge 1 ]]; then report "POST /v1/search with key -> 200, $hits hit(s)" 0
else report "POST /v1/search with key -> 200 (got ${code:-no-response}, $hits hit(s))" 1; fi

# d) POST /v1/ask with key -> 200 and at least 1 citation
code=$(curl -s -o "$tmp/d" -w '%{http_code}' --max-time 180 -X POST "$BASE/v1/ask" \
  -H 'Content-Type: application/json' -H "X-API-Key: $JIRA_RAG_API_KEY" \
  -d '{"question": "Why does Spark shuffle spill to disk and how can I reduce shuffle spill?", "k": 5}' || true)
cits=$(json_len "$tmp/d" citations)
if [[ "$code" == "200" && "$cits" -ge 1 ]]; then report "POST /v1/ask with key -> 200, $cits citation(s)" 0
else report "POST /v1/ask with key -> 200 (got ${code:-no-response}, $cits citation(s))" 1; fi

if [[ "$failures" -gt 0 ]]; then
  echo "SMOKE: $failures check(s) failed"
  exit 1
fi
echo "SMOKE: all checks passed"
