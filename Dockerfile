# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PETSAFE_RUNS_DIR=/models \
    PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /cache/huggingface /cache/torch /models \
    && chown -R app:app /cache /models

COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra api --no-dev --no-install-project

COPY src ./src
COPY web ./web
COPY data/toxicity/plant_toxicity.json ./data/toxicity/plant_toxicity.json

USER app
EXPOSE 8000

CMD ["uvicorn", "petsafe.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
