# RAG Zero (Production)

Production-grade, near-zero-hallucination agentic RAG pipeline. Saw a lot of sloppy articles on this title, so thought why not write something that really works instead of clickbait slop
 
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
