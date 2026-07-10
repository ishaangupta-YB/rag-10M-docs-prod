"""Unit tests for evaluation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rag_zero.evaluation.golden import GoldenSetBuilder
from rag_zero.evaluation.metrics import ConfusionMatrix

if TYPE_CHECKING:
    from pathlib import Path


def test_confusion_matrix_hallucination_and_coverage() -> None:
    preds = [
        (True, "answered"),   # TP
        (True, "abstained"),  # FN
        (False, "answered"),  # FP
        (False, "abstained"), # TN
    ]
    matrix = ConfusionMatrix().from_predictions(preds)
    assert matrix.dict() == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert matrix.hallucination_rate == 0.5
    assert matrix.coverage == 0.5


@pytest.mark.slow
@pytest.mark.integration
def test_golden_set_has_answerable_and_unanswerable() -> None:
    builder = GoldenSetBuilder(answerable_n=5, unanswerable_n=5, seed=42)
    items = builder.build()
    assert len(items) == 10
    answerable = [i for i in items if i.answerable]
    unanswerable = [i for i in items if not i.answerable]
    assert len(answerable) == 5
    assert len(unanswerable) == 5


@pytest.mark.slow
def test_scale_lab_10k_no_oom(tmp_path: Path) -> None:
    from rag_zero.evaluation.scale_lab import ScaleLab

    lab = ScaleLab(uri=tmp_path / "scale", dim=128, num_queries=10, seed=42)
    result = lab.run_scale(sizes=[10_000])
    assert "10000" in result["scales"]
    assert result["scales"]["10000"]["p95_latency_sec"] >= 0.0
