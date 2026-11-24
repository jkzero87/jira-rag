"""The image's Docker HEALTHCHECK must report the API as healthy.

Builds the image from the repo Dockerfile, starts it without Postgres, models
or the 27B (/v1/health needs none of them), and polls `docker inspect` until
the health status is "healthy". Needs a local Docker daemon, so it is marked
`local` and excluded from CI (`pytest -m "not local"`).
Run:  .venv/bin/python -m pytest -q -m local tests/test_docker_health.py
"""
import subprocess
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "jira-rag-api:healthcheck-test"
TIMEOUT_S = 180  # start-period 30s + interval 30s, with margin for a cold start


def docker(*args, check=True):
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True)


@pytest.mark.local
def test_container_becomes_healthy():
    docker("build", "-q", "-t", IMAGE, str(ROOT))
    name = f"jira-rag-health-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, IMAGE)
    try:
        status = ""
        deadline = time.monotonic() + TIMEOUT_S
        while time.monotonic() < deadline:
            status = docker("inspect", "--format", "{{.State.Health.Status}}", name).stdout.strip()
            if status in ("healthy", "unhealthy"):
                break
            time.sleep(3)
        assert status == "healthy", docker("inspect", "--format", "{{json .State.Health}}", name).stdout
    finally:
        docker("rm", "-f", name, check=False)
