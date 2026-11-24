# eval/

Offline evaluation scripts for retrieval, reranking and answers. Most of them run with the project venv (`.venv`).

## `rerank_speed.py` and `.venv-eval`

`eval/rerank_speed.py` (reranker speed and accuracy: `m3`, `torch-quant`, `onnx`) has its own venv, `.venv-eval`, pinned to the September 2026 stack in [requirements-eval.txt](requirements-eval.txt).

Why a separate venv: `optimum-onnx` has no release compatible with transformers 5.x. Its latest release (0.1.0) requires `transformers<4.58`, and the `optimum[onnx]` / `optimum[onnxruntime]` extras install `optimum-onnx`, so they carry the same requirement. The project venv uses transformers 5.24.0 (see `requirements-gpu.txt`), so optimum cannot be installed there.

Note: the script itself does not import `optimum`. The `onnx` mode exports the model with `torch.onnx.export` and then stops, because int8 quantization failed with a shape-inference error. Only the docstring and comments mention `optimum-cli`. `.venv-eval` reproduces the September environment, in which those numbers were measured.

Create it once (CPU torch, from the repo root):

```sh
python3.12 -m venv .venv-eval
.venv-eval/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv-eval/bin/pip install -r eval/requirements-eval.txt
.venv-eval/bin/pip check
```

Run it (from the repo root). The script reads the `PG*` variables (libpq), the cached lists in `eval/cache/`, and the model weights from the repo HF cache:

```sh
set -a; . ./.env; set +a
HF_HOME=.hf HF_HUB_OFFLINE=1 .venv-eval/bin/python eval/rerank_speed.py m3
HF_HOME=.hf HF_HUB_OFFLINE=1 .venv-eval/bin/python eval/rerank_speed.py onnx
```

`.venv-eval/` is git-ignored. Never change `requirements-api.txt` or `requirements-gpu.txt` for this script.
