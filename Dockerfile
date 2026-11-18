# jira-rag API image. Models are NOT baked in: mount the HF cache at /models.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    JIRA_RAG_DEVICE=cpu

WORKDIR /app

# 1) Dependencies first: this layer is reused until requirements.txt changes.
#    CPU-only torch here; the CUDA pin in requirements.txt is for the host venv.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && grep -v '^torch' requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt

# 2) Code last: editing src/ rebuilds only this layer.
COPY src/ src/

# Run as a non-root user with the same uid as the host user, so the mounted
# /models cache stays readable and writable.
RUN useradd --uid 1000 --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# 0.0.0.0, not 127.0.0.1: inside a container 127.0.0.1 means "only myself".
CMD ["uvicorn", "api:app", "--app-dir", "src/jira_rag", "--host", "0.0.0.0", "--port", "8000"]
