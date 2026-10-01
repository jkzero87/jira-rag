#!/usr/bin/env bash
# Re-measure every published number on the current commit, in the three
# phases the 16 GB GPU forces (README, "Architecture"):
#
#   1. 27B UP:      parser accuracy (run_parse_eval) + parse cache for all 40
#   2. 27B STOPPED: retrieval eval (run_eval) + generation contexts (build_contexts)
#   3. 27B UP:      answers (generate_answers) + grading (grade_answers)
#
# Usage (repo root, PG* variables set, CUDA env active):
#   bash eval/remeasure.sh
#
# Between phases the script waits until llama-server is up or down. Set
# LLAMA_START_CMD / LLAMA_STOP_CMD to have it start/stop the server itself;
# otherwise start/stop it by hand when the script asks.
# Every output is named after the commit and is never overwritten.
set -euo pipefail

cd "$(dirname "$0")/.."
LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://127.0.0.1:8092/v1/chat/completions}"
MODELS_URL="${LLAMA_SERVER_URL%/chat/completions}/models"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "error: uncommitted changes; results are keyed by commit, commit first" >&2
    exit 1
fi
COMMIT="$(git rev-parse --short HEAD)"
PARSE_CACHE="eval/cache/parse_${COMMIT}.json"
CONTEXTS="eval/cache/contexts_${COMMIT}.json"
ANSWERS="eval/results/answers_${COMMIT}.json"
GRADE="eval/results/grade_${COMMIT}.json"

server_up() { curl -sf -o /dev/null --max-time 5 "$MODELS_URL"; }

wait_for() {  # wait_for up|down
    local want="$1" cmd
    if [[ "$want" == up ]]; then cmd="${LLAMA_START_CMD:-}"; else cmd="${LLAMA_STOP_CMD:-}"; fi
    if [[ "$want" == up ]] && server_up; then return; fi
    if [[ "$want" == down ]] && ! server_up; then return; fi
    if [[ -n "$cmd" ]]; then
        echo ">> running: $cmd"
        bash -c "$cmd"
    else
        echo ">> $( [[ "$want" == up ]] && echo START || echo STOP ) llama-server ($MODELS_URL) now; waiting ..."
    fi
    until { [[ "$want" == up ]] && server_up; } || { [[ "$want" == down ]] && ! server_up; }; do
        sleep 5
    done
    echo ">> llama-server is $want"
}

echo "== phase 1/3: parse (27B up) — commit $COMMIT"
wait_for up
MODEL="$(python3 -c 'import json, sys, urllib.request
d = json.load(urllib.request.urlopen(sys.argv[1], timeout=10))
print(d["data"][0]["id"] if "data" in d else d["models"][0]["name"])' "$MODELS_URL")"
echo ">> server model: $MODEL"
python3 eval/run_parse_eval.py
python3 eval/run_eval.py --write-parse-cache "$PARSE_CACHE"

echo "== phase 2/3: retrieval (27B stopped, embedder + reranker on GPU)"
wait_for down
python3 eval/run_eval.py --parse-cache "$PARSE_CACHE" --model "$MODEL"
python3 eval/build_contexts.py --parse-cache "$PARSE_CACHE" --out "$CONTEXTS" --model "$MODEL"

echo "== phase 3/3: generation (27B up)"
wait_for up
python3 eval/generate_answers.py --contexts "$CONTEXTS" --out "$ANSWERS"
python3 eval/grade_answers.py --answers "$ANSWERS" --out "$GRADE"

echo
echo "done. new results for $COMMIT:"
ls -1 eval/results/*"${COMMIT}"* "$PARSE_CACHE" "$CONTEXTS"
