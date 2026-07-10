"""Document loaders for HotpotQA, SQuAD v2, and JSONL corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rag_zero.models.domain import Passage


class SQuADV2Loader:
    """Load SQuAD v2 examples (used for impossible questions evaluation)."""

    def __init__(self, split: str = "validation") -> None:
        self.split = split

    def load(self) -> tuple[list[Passage], list[dict[str, Any]]]:
        ds = load_dataset("rajpurkar/squad_v2", split=self.split)
        passages: list[Passage] = []
        questions: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for example in ds:
            title = str(example["title"])
            context = str(example["context"])
            if title not in seen_titles:
                passages.append(
                    Passage(
                        id=f"squad_{title}",
                        title=title,
                        text=context,
                        metadata={"source": "squad_v2"},
                    )
                )
                seen_titles.add(title)
            answers = example["answers"]
            questions.append(
                {
                    "question": str(example["question"]),
                    "gold_titles": [title],
                    "gold_answer": answers["text"][0] if answers["text"] else None,
                    "answerable": bool(answers["text"]),
                    "passage_id": f"squad_{title}",
                }
            )
        return passages, questions


class HotpotQALoader:
    """Load HotpotQA distractor validation split."""

    def __init__(self, split: str = "validation", slice_n: int | None = None) -> None:
        self.split = split
        self.slice_n = slice_n

    def load(self) -> tuple[list[Passage], list[dict[str, Any]]]:
        ds = load_dataset("hotpot_qa", "distractor", split=self.split)
        if self.slice_n:
            ds = ds.select(range(min(self.slice_n, len(ds))))

        passages: list[Passage] = []
        questions: list[dict[str, Any]] = []
        seen_titles: set[str] = set()

        for _i, example in enumerate(ds):
            titles = list(example["context"]["title"])
            sentences = list(example["context"]["sentences"])
            for title, sents in zip(titles, sentences, strict=True):
                if title in seen_titles:
                    continue
                text = " ".join(str(s) for s in sents)
                passages.append(
                    Passage(
                        id=f"hotpot_{title}",
                        title=str(title),
                        text=text,
                        metadata={"source": "hotpotqa", "split": self.split},
                    )
                )
                seen_titles.add(title)

            questions.append(
                {
                    "question": str(example["question"]),
                    "gold_titles": list(example["supporting_facts"]["title"]),
                    "gold_answer": str(example["answer"]),
                    "answerable": True,
                    "example_id": str(example["_id"]),
                }
            )

        return passages, questions


class JSONLLoader:
    """Load a corpus from JSONL where each line is {"id", "title", "text"}."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def load(self) -> list[Passage]:
        passages: list[Passage] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                passages.append(
                    Passage(
                        id=str(obj.get("id", obj.get("title"))),
                        title=str(obj.get("title", "")),
                        text=str(obj.get("text", "")),
                        metadata=obj.get("metadata", {}),
                    )
                )
        return passages


def load_corpus(
    source: str,
    *,
    slice_n: int | None = None,
    jsonl_path: Path | str | None = None,
) -> tuple[list[Passage], list[dict[str, Any]]]:
    """Convenience dispatcher for known corpora."""
    if source == "hotpotqa":
        return HotpotQALoader(slice_n=slice_n).load()
    if source == "squad_v2":
        return SQuADV2Loader().load()
    if source == "jsonl":
        if jsonl_path is None:
            raise ValueError("jsonl_path is required")
        return JSONLLoader(jsonl_path).load(), []
    raise ValueError(f"Unknown source: {source}")
