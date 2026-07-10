# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy lockfile first to cache dependency layer.
COPY pyproject.toml README.md uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project \
    && uv sync --no-dev --frozen

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /app

# Copy only the virtual environment from builder.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code for editable install reference and any runtime data paths.
COPY src ./src
COPY scripts ./scripts
COPY README.md ./

# Run as non-root.
RUN groupadd -r app && useradd -r -g app app
RUN chown -R app:app /app
USER app

# Runtime env defaults.
ENV RAG_PORT=8000
ENV RAG_HOST=0.0.0.0
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${RAG_PORT}/health').read()" || exit 1

CMD ["sh", "-c", "uvicorn rag_zero.serving.app:create_app --factory --host ${RAG_HOST} --port ${RAG_PORT}"]
