"""Golden set builder using HotpotQA + SQuAD v2 + false-premise questions."""

from __future__ import annotations

from typing import Any

from rag_zero.ingestion.loader import HotpotQALoader, SQuADV2Loader
from rag_zero.models.domain import EvalItem


class GoldenSetBuilder:
    """Build a balanced golden set of answerable/unanswerable examples."""

    def __init__(
        self,
        answerable_n: int = 100,
        unanswerable_n: int = 100,
        seed: int = 42,
    ) -> None:
        self.answerable_n = answerable_n
        self.unanswerable_n = unanswerable_n
        self.seed = seed

    @staticmethod
    def _false_premise_questions() -> list[EvalItem]:
        templates = [
            "What year did Napoleon invade the moon?",
            "Who wrote the novel 'The Great Gatsby Returns'?",
            "What is the capital of the United Kingdom of Great Britain and Mars?",
            "Which NASA mission discovered water on the sun?",
            "Who invented the perpetual motion machine in 1999?",
        ]
        return [
            EvalItem(
                question=q,
                gold_titles=[],
                gold_answer=None,
                answerable=False,
                is_false_premise=True,
            )
            for q in templates
        ]

    def build(self) -> list[EvalItem]:
        import random

        rng = random.Random(self.seed)

        # Answerable from HotpotQA.
        _hotpot_passages, hotpot_qs = HotpotQALoader(split="validation").load()
        answerable_items: list[EvalItem] = []
        limit = min(self.answerable_n, len(hotpot_qs))
        for q in hotpot_qs[:limit]:
            answerable_items.append(
                EvalItem(
                    question=q["question"],
                    gold_titles=list(set(q["gold_titles"])),
                    gold_answer=q.get("gold_answer"),
                    answerable=True,
                )
            )

        # Unanswerable from SQuAD v2 impossible questions.
        _squad_passages, squad_qs = SQuADV2Loader(split="validation").load()
        impossible = [q for q in squad_qs if not q["answerable"]]
        impossible_items: list[EvalItem] = []
        limit = min(self.unanswerable_n - 20, len(impossible))
        for q in impossible[:limit]:
            impossible_items.append(
                EvalItem(
                    question=q["question"],
                    gold_titles=list(set(q["gold_titles"])),
                    gold_answer=None,
                    answerable=False,
                )
            )

        false_premise_items = self._false_premise_questions()
        combined = impossible_items + false_premise_items
        rng.shuffle(combined)
        # Ensure exactly unanswerable_n unanswerable items.
        combined = combined[: self.unanswerable_n]

        return rng.sample(answerable_items, len(answerable_items)) + combined

    def save(self, items: list[EvalItem], path: Any) -> None:
        import json
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(
                    json.dumps(
                        {
                            "id": item.id,
                            "question": item.question,
                            "gold_titles": item.gold_titles,
                            "gold_answer": item.gold_answer,
                            "answerable": item.answerable,
                            "is_false_premise": item.is_false_premise,
                        }
                    )
                    + "\n"
                )
