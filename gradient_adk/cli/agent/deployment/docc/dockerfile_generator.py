from __future__ import annotations

from pathlib import Path


AGENT_DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8080

ENV PORT=8080

CMD ["python", "-m", "uvicorn", "gradient_adk.runtime.server:app", "--host", "0.0.0.0", "--port", "8080"]
"""


def generate_dockerfile(source_dir: Path) -> Path:
    """Generate a Dockerfile for the ADK agent in the source directory.

    If a Dockerfile already exists, it is left untouched and its path is returned.

    Args:
        source_dir: The agent project root containing main.py / requirements.txt.

    Returns:
        Path to the Dockerfile.
    """
    dockerfile_path = source_dir / "Dockerfile"
    if dockerfile_path.exists():
        return dockerfile_path

    dockerfile_path.write_text(AGENT_DOCKERFILE_TEMPLATE)
    return dockerfile_path
