"""Add one-line context prefixes to chunks via parallel LLM calls."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rag_zero.clients.base import BaseLLMClient, ChatMessage
from rag_zero.models.domain import Chunk
from rag_zero.models.prompts import CONTEXUALIZER_PROMPT


class Contextualizer:
    """Generates one-sentence context prefixes for chunks."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        checkpoint_path: Path | str = Path("./checkpoints/contextualizer.jsonl"),
        concurrency: int = 8,
        max_tokens: int = 64,
    ) -> None:
        self.llm_client = llm_client
        self.checkpoint_path = Path(checkpoint_path)
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self._semaphore = asyncio.Semaphore(concurrency)

    def _load_checkpoint(self) -> dict[str, str]:
        prefixes: dict[str, str] = {}
        if not self.checkpoint_path.exists():
            return prefixes
        with self.checkpoint_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prefixes[str(obj["chunk_id"])] = str(obj["context_prefix"])
        return prefixes

    async def _contextualize_one(self, chunk: Chunk) -> str:
        prompt = CONTEXUALIZER_PROMPT.format(text=chunk.text)
        async with self._semaphore:
            response = await self.llm_client.chat(
                [ChatMessage(role="user", content=prompt)],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
        return response.strip().split("\n")[0].strip()

    async def contextualize(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Attach context_prefix to each chunk, resuming from checkpoint."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_map = self._load_checkpoint()

        pending = [c for c in chunks if c.chunk_id not in prefix_map]

        async def worker(chunk: Chunk) -> tuple[str, str]:
            prefix = await self._contextualize_one(chunk)
            return chunk.chunk_id, prefix

        tasks = [asyncio.create_task(worker(c)) for c in pending]
        if tasks:
            with self.checkpoint_path.open("a", encoding="utf-8") as fh:
                for coro in asyncio.as_completed(tasks):
                    chunk_id, prefix = await coro
                    fh.write(json.dumps({"chunk_id": chunk_id, "context_prefix": prefix}) + "\n")
                    prefix_map[chunk_id] = prefix

        return [
            Chunk(
                chunk_id=c.chunk_id,
                passage_id=c.passage_id,
                title=c.title,
                text=c.text,
                context_prefix=prefix_map.get(c.chunk_id, ""),
                token_len=c.token_len,
                metadata=c.metadata,
            )
            for c in chunks
        ]
