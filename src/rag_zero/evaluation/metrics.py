"""Evaluation metrics: confusion matrix, coverage, hallucination, RAGAS proxies."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from rag_zero.evaluation.golden import GoldenSetBuilder

if TYPE_CHECKING:
    from rag_zero.config import Settings


class ConfusionMatrix:
    """2x2 confusion matrix over answerable x answered/abstained."""

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def from_predictions(
        self,
        predictions: list[tuple[bool, str]],
    ) -> ConfusionMatrix:
        for answerable, status in predictions:
            answered = status == "answered"
            if answerable and answered:
                self.tp += 1
            elif answerable and not answered:
                self.fn += 1
            elif not answerable and answered:
                self.fp += 1
            else:
                self.tn += 1
        return self

    @property
    def hallucination_rate(self) -> float:
        total_unanswerable = self.fp + self.tn
        if total_unanswerable == 0:
            return 0.0
        return self.fp / total_unanswerable

    @property
    def coverage(self) -> float:
        total_answerable = self.tp + self.fn
        if total_answerable == 0:
            return 0.0
        return self.tp / total_answerable

    def dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn}


class Metrics:
    """Compute hallucination, coverage, and RAGAS-style metrics over a golden set."""

    def __init__(
        self,
        settings: Settings,
        query_service: Any | None = None,
    ) -> None:
        self.settings = settings
        self.query_service = query_service
        self._corpus: dict[str, str] | None = None

    async def _run_query(self, question: str) -> Any:
        if self.query_service is None:
            raise RuntimeError("query_service is required for evaluation")
        return await self.query_service.query(question)

    def _load_corpus(self) -> dict[str, str]:
        """Load a chunk_id -> text mapping from the persisted BM25 corpus file."""
        if self._corpus is not None:
            return self._corpus
        corpus: dict[str, str] = {}
        if self.query_service is None:
            self._corpus = corpus
            return corpus
        try:
            bm25_path = self.query_service.index_store.bm25_path
            corpus_file = bm25_path / "corpus.jsonl"
            if corpus_file.exists():
                with corpus_file.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        corpus[str(obj["id"])] = str(obj.get("text", ""))
        except Exception:
            corpus = {}
        self._corpus = corpus
        return corpus

    def _chunk_texts(self, chunk_ids: list[str]) -> list[str]:
        """Resolve chunk IDs to their text using the persisted corpus."""
        corpus = self._load_corpus()
        return [corpus[cid] for cid in chunk_ids if cid in corpus]

    async def run_golden_set(
        self,
        answerable_n: int = 20,
        unanswerable_n: int = 20,
    ) -> dict[str, Any]:
        builder = GoldenSetBuilder(
            answerable_n=answerable_n,
            unanswerable_n=unanswerable_n,
            seed=self.settings.seed,
        )
        items = builder.build()

        predictions: list[tuple[bool, str]] = []
        details: list[dict[str, Any]] = []
        recall_scores: list[float] = []
        faithfulness_scores: list[float] = []
        relevancy_scores: list[float] = []

        for item in items:
            try:
                response = await self._run_query(item.question)
            except Exception as exc:
                details.append(
                    {
                        "question": item.question,
                        "answerable": item.answerable,
                        "predicted": "error",
                        "error": str(exc),
                    }
                )
                continue

            status = response.status
            predictions.append((item.answerable, status))

            # Context recall at retrieval cutoff.
            if item.gold_titles and self.query_service is not None:
                try:
                    retrieved = await self.query_service.retriever.retrieve(item.question)
                    k = self.settings.rerank_top_n
                    titles = [c.title for c in retrieved[:k]]
                    found = sum(1 for t in item.gold_titles if t in titles)
                    recall = found / len(set(item.gold_titles))
                except Exception:
                    recall = 0.0
                recall_scores.append(recall)

            # Faithfulness proxy: fraction of claims supported by cited evidence text.
            if status == "answered":
                evidence_texts = self._chunk_texts(response.citations)
                if evidence_texts:
                    faith_scores = await self._faithfulness_from_answer(
                        response.answer, evidence_texts
                    )
                    faithfulness_scores.append(
                        sum(faith_scores) / len(faith_scores) if faith_scores else 0.0
                    )

            # Answer relevancy proxy: keyword overlap.
            relevancy_scores.append(
                self._answer_relevancy(item.question, response.answer)
            )

            details.append(
                {
                    "question": item.question,
                    "answerable": item.answerable,
                    "predicted": status,
                    "answer": response.answer,
                    "citations": response.citations,
                    "hops": response.hops,
                }
            )

        matrix = ConfusionMatrix().from_predictions(predictions)

        def _avg(scores: list[float]) -> float:
            return sum(scores) / len(scores) if scores else 0.0

        return {
            "metrics": {
                "hallucination_rate": matrix.hallucination_rate,
                "coverage": matrix.coverage,
                "context_recall_at_k": _avg(recall_scores),
                "faithfulness": _avg(faithfulness_scores),
                "answer_relevancy": _avg(relevancy_scores),
            },
            "confusion": matrix.dict(),
            "details": details,
        }

    async def _faithfulness_from_answer(
        self, answer: str, evidence: list[str]
    ) -> list[float]:
        """Return support scores for each sentence claim against evidence passages."""
        if self.query_service is None or not evidence:
            return [0.0]
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not sentences or answer.upper() == "INSUFFICIENT_EVIDENCE":
            return [0.0]
        scores: list[float] = []
        for sentence in sentences:
            # Use only the most relevant evidence passages per claim.
            context = evidence[:3]
            score = await self.query_service.verifier.verifier.verify(
                sentence, context
            )
            scores.append(score)
        return scores

    def _answer_relevancy(self, question: str, answer: str) -> float:
        """Simple keyword-overlap relevancy proxy."""
        q_tokens = set(question.lower().split())
        a_tokens = set(answer.lower().split())
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens & a_tokens)
        return min(1.0, overlap / math.sqrt(len(q_tokens)))

    @staticmethod
    def risk_coverage_curve(
        tau_values: list[float],
        scores: list[tuple[bool, float]],
    ) -> list[tuple[float, float, float]]:
        """Return (tau, coverage, hallucination_rate) for each threshold."""
        curve: list[tuple[float, float, float]] = []
        for tau in tau_values:
            predictions = [
                (answerable, score >= tau) for answerable, score in scores
            ]
            matrix = ConfusionMatrix().from_predictions(
                [(a, "answered" if p else "abstained") for a, p in predictions]
            )
            curve.append((tau, matrix.coverage, matrix.hallucination_rate))
        return curve
