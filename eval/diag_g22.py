"""Read-only diagnostic: call the server with the exact parse request for G22
and report finish_reason / token counts / content+reasoning lengths / max_tokens.

This script is diagnostic only — it does not write anything.
"""

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jira_rag.query_parse import (
    LLAMA_URL, MODEL, SYSTEM_PROMPT, _USER_TEMPLATE, _JSON_SCHEMA, parse,
)

G22 = "The optimizer folds 1 + 2 + a into a constant, but not a + 1 + 2. Was that fixed?"

MAX_TOKENS = 1024  # default in _call_llm

def main():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(question=G22)},
    ]
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": _JSON_SCHEMA},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(LLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    print(f"max_tokens sent: {MAX_TOKENS}")
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    choice = result["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    print("finish_reason:", choice.get("finish_reason"))
    print("completion_tokens:", result.get("usage", {}).get("completion_tokens"))
    print("prompt_tokens:", result.get("usage", {}).get("prompt_tokens"))
    print("content length:", len(content))
    print("reasoning_content length:", len(reasoning))
    print("content preview:", repr(content[:200]))
    print("reasoning preview:", repr(reasoning[:200]))
    # Also try parsing to confirm behavior
    try:
        form = parse(G22)
        print("parse OK, form:", json.dumps(form))
    except Exception as exc:
        print("parse FAILED:", exc)

if __name__ == "__main__":
    main()
