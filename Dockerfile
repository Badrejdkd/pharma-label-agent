# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Étape 1 : installer les dépendances figées par uv.lock dans un environnement virtuel
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Étape 2 : image finale, sans uv ni outils de build
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CORPUS_DIR=/app/corpus \
    CACHE_DIR=/app/.cache

RUN useradd --create-home --uid 10001 agent
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY corpus ./corpus
RUN mkdir -p /app/.cache && chown agent:agent /app/.cache

USER agent
EXPOSE 8000

# L'index (embeddings du corpus) est construit au démarrage : on laisse 60 s avant le premier contrôle.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]