"""Evaluation metrics for the RAPTOR reproduction harness.

Faithful to RAPTOR (arXiv:2401.18059v1) and QASPER/NarrativeQA/QuALITY
evaluation conventions.
"""

from experiments.metrics.qa_f1 import (
    normalize_answer,
    token_f1,
    qasper_answer_f1,
)
from experiments.metrics.evidence_f1 import (
    evidence_f1,
    evidence_f1_max,
)
from experiments.metrics.accuracy import (
    extract_choice,
    accuracy,
    accuracy_on_subset,
)
from experiments.metrics.generation import (
    rouge_l,
    bleu_1,
    bleu_4,
    meteor,
    narrativeqa_metrics,
)

__all__ = [
    # qa_f1
    "normalize_answer",
    "token_f1",
    "qasper_answer_f1",
    # evidence_f1
    "evidence_f1",
    "evidence_f1_max",
    # accuracy
    "extract_choice",
    "accuracy",
    "accuracy_on_subset",
    # generation
    "rouge_l",
    "bleu_1",
    "bleu_4",
    "meteor",
    "narrativeqa_metrics",
]
