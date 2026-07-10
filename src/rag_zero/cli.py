"""Command-line interface for rag-zero."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from rag_zero.clients.embedder_client import OpenAIEmbedderClient
from rag_zero.clients.llm_client import LLMClient
from rag_zero.config import Settings
from rag_zero.ingestion.chunker import SentenceChunker
from rag_zero.ingestion.clean import clean_passages
from rag_zero.ingestion.contextualizer import Contextualizer
from rag_zero.ingestion.dedup import deduplicate_passages
from rag_zero.ingestion.indexer import IndexStore
from rag_zero.ingestion.loader import load_corpus

app = typer.Typer(name="rag-zero")
console = Console()


def _settings_from_env(artifact_dir: Path | None = None) -> Settings:
    settings = Settings()
    if artifact_dir is not None:
        settings.artifact_dir = artifact_dir
        settings.lancedb_uri = artifact_dir / "lancedb"
        settings.bm25_path = artifact_dir / "bm25s"
        settings.checkpoint_dir = artifact_dir / "checkpoints"
    return settings


@app.command()
def ingest(
    source: str = typer.Option("hotpotqa", help="Corpus source: hotpotqa, squad_v2, jsonl"),
    output: Path = typer.Option(Path("./artifacts"), help="Artifact output directory"),
    jsonl_path: Path | None = typer.Option(None, help="Path to JSONL corpus if source=jsonl"),
    slice: int | None = typer.Option(None, help="Limit number of HotpotQA examples"),
    min_words: int = typer.Option(20, help="Minimum words per passage"),
    overwrite: bool = typer.Option(False, help="Overwrite existing artifacts"),
) -> None:
    """Run the full ingestion pipeline and write LanceDB + bm25s artifacts."""
    settings = _settings_from_env(output)
    settings.ensure_dirs()

    llm_client = LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )
    embedder = OpenAIEmbedderClient(
        base_url=settings.embedder_base_url,
        model=settings.embedder_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        max_retries=settings.embedder_max_retries,
        expected_dim=settings.embedder_dim,
        query_instruction=settings.embedder_query_instruction,
    )

    async def _run() -> None:
        console.print(f"[bold green]Loading corpus:[/bold green] {source}")
        passages, _ = load_corpus(source, slice_n=slice, jsonl_path=jsonl_path)
        console.print(f"  raw passages: {len(passages)}")

        passages = clean_passages(passages, min_words=min_words)
        console.print(f"  after quality filter: {len(passages)}")

        passages = deduplicate_passages(passages)
        console.print(f"  after deduplication: {len(passages)}")

        chunker = SentenceChunker(
            tokenizer_name=settings.embedder_model,
            target_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap,
        )
        chunks = chunker.chunk_passages(passages)
        console.print(f"  chunks: {len(chunks)}")

        contextualizer = Contextualizer(
            llm_client,
            checkpoint_path=settings.normalized_checkpoint_dir / "contextualizer.jsonl",
            concurrency=8,
        )
        chunks = await contextualizer.contextualize(chunks)
        console.print("  contextualization complete")

        store = IndexStore(settings, embedder)
        await store.build_or_load(chunks, overwrite=overwrite)
        console.print(
            f"[bold green]Indices built at[/bold green] {settings.normalized_lancedb_uri} "
            f"and {settings.normalized_bm25_path}"
        )

        await llm_client.close()
        await embedder.close()

    asyncio.run(_run())


def main() -> None:
    app()
