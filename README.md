# RAG Zero (Production)

Production-grade, near-zero-hallucination agentic RAG pipeline.

Built from the research notebook [rag-zero-hallucinations](https://github.com/FareedKhan-dev/rag-zero-hallucinations) and refactored into modular services.

## Quick start

```bash
cp .env.example .env
# Edit .env to point the endpoints at your vLLM instance.
uv sync --extra dev
uv run rag-zero ingest --source hotpotqa --output artifacts/ --slice 20000
uv run uvicorn rag_zero.serving.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker-compose up --build
```

## Tests

```bash
uv run pytest
```
