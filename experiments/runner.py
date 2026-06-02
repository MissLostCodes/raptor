"""Orchestration: dataset -> QA prompt + scorer, and the per-(arm, dataset, doc,
question) evaluation loop.

The whole loop is offline-testable: ``build_answerer_fn`` and ``loaders`` are
injectable, so a fake loader (1 tiny Document) and a fake Answerer (canned
answer) exercise the orchestration, scoring, record assembly, and JSON write
without any network or model load.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

from experiments.datasets import get_loader
from experiments.datasets.base import Document, QAExample
from experiments.metrics import (
    extract_choice,
    narrativeqa_metrics,
    qasper_answer_f1,
)


# ---------------------------------------------------------------------------
# Dataset -> QA prompt mapping
# ---------------------------------------------------------------------------
_QA_PROMPTS: Dict[str, Tuple[str, str]] = {
    "qasper": (
        "You are Question Answering Portal",
        "Given Context: {context}\n\nAnswer the following question concisely "
        "based only on the context: {question}",
    ),
    "narrativeqa": (
        "You are Question Answering Portal",
        "Given Context: {context}\n\nAnswer the following question with a short "
        "phrase: {question}",
    ),
    "quality": (
        "You are Question Answering Portal",
        "Given Context: {context}\n\nQuestion: {question}\n\nRespond with the "
        "letter (A, B, C, or D) of the best option.",
    ),
}


def qa_prompt_for(dataset_name: str) -> Tuple[str, str]:
    """Return (system_prompt, user_template) for a dataset.

    The user template uses only ``{context}`` and ``{question}`` (the two fields
    ``CachedOpenRouterQAModel.answer_question`` fills). For QuALITY, the answer
    options are folded into the question text by :func:`compose_query` so they
    reach the model without an unfillable ``{options}`` placeholder.
    """
    try:
        return _QA_PROMPTS[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown dataset for QA prompt: {dataset_name!r}")


def format_options(options: List[str]) -> str:
    """Format options as 'A) ...\\nB) ...\\n...'."""
    return "\n".join(
        f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
    )


def compose_query(dataset_name: str, example: QAExample) -> str:
    """The query string passed to ``Answerer.answer`` (used for retrieval AND as
    ``{question}`` in the QA prompt).

    For QuALITY the multiple-choice options are appended so the model can pick a
    letter; for other datasets it is just the question text.
    """
    if dataset_name == "quality" and example.options:
        return f"{example.question}\n\nOptions:\n{format_options(example.options)}"
    return example.question


# ---------------------------------------------------------------------------
# Dataset -> scorer mapping
# ---------------------------------------------------------------------------
def score_question(dataset_name: str, example: QAExample, prediction: str) -> dict:
    """Return a per-question metric record for the dataset.

    - qasper: {answer_f1} (+ evidence_f1 only if both gold evidence sets and a
      retrieved-evidence list are available; here we have neither at scoring
      time, so evidence is omitted and noted).
    - quality: {choice, correct, is_hard}
    - narrativeqa: {rouge_l, bleu_1, bleu_4, meteor}
    """
    if dataset_name == "qasper":
        rec = {
            "answer_f1": qasper_answer_f1(
                prediction, example.gold_answers, example.is_unanswerable
            )
        }
        # Evidence F1 needs both gold evidence sets AND a retrieved-evidence
        # list. We do not thread retrieved evidence through scoring, so we omit
        # evidence_f1 here rather than report a misleading 0.
        rec["evidence_f1"] = None
        return rec

    if dataset_name == "quality":
        choice = extract_choice(prediction, example.options or [])
        return {
            "choice": choice,
            "correct": int(choice == example.gold_index),
            "is_hard": bool(example.is_hard),
        }

    if dataset_name == "narrativeqa":
        return narrativeqa_metrics(prediction, example.gold_answers)

    raise ValueError(f"Unknown dataset for scoring: {dataset_name!r}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    cfg,
    build_answerer_fn: Optional[Callable] = None,
    loaders: Optional[Dict] = None,
    seed: int = 0,
    progress: Callable = print,
    summarization_model=None,
    qa_models: Optional[Dict] = None,
) -> dict:
    """Run the full experiment grid and write results to disk.

    Args:
        cfg: ExperimentConfig.
        build_answerer_fn: injectable; defaults to ``pipeline.build_answerer``.
            Signature: (arm, doc, cfg, summ_model, qa_model, ...) -> Answerer.
        loaders: optional {dataset_name: DatasetLoader} override (fakes in tests).
        seed: seed used for subset selection and the results filename.
        progress: callable for progress messages (default print).
        summarization_model / qa_models: optional injected models. When
            ``build_answerer_fn`` is provided (test path), real OpenRouter models
            are NOT constructed unless explicitly injected.

    Returns ``{"config": cfg.to_dict(), "records": [...]}`` and writes it to
    ``cfg.results_dir/results_<seed>.json``.
    """
    real_build = build_answerer_fn is None
    if real_build:
        from experiments import pipeline
        from experiments.models import build_models, make_qa_model

        build_answerer_fn = pipeline.build_answerer
        if summarization_model is None:
            summarization_model, _ = build_models(cfg)

    records: List[dict] = []

    for dataset in cfg.datasets:
        progress(f"[dataset] {dataset}")
        loader = (loaders or {}).get(dataset) if loaders else None
        if loader is None:
            loader = get_loader(dataset)
        n = cfg.subset_sizes.get(dataset)
        documents = loader.load(limit=n)

        # QA model carries the dataset-specific prompt.
        if qa_models is not None and dataset in qa_models:
            qa_model = qa_models[dataset]
        elif real_build:
            qa_model = make_qa_model(cfg, *qa_prompt_for(dataset))
        else:
            qa_model = None  # fake build_answerer_fn won't use it

        for arm in cfg.arms:
            progress(f"  [arm] {arm}")
            for doc in documents:
                answerer = build_answerer_fn(
                    arm, doc, cfg, summarization_model, qa_model
                )
                for q in doc.questions:
                    pred, _context = answerer.answer(compose_query(dataset, q))
                    rec = score_question(dataset, q, pred)
                    routing = {}
                    if hasattr(answerer, "routing_info"):
                        routing = answerer.routing_info() or {}
                    records.append(
                        {
                            "arm": arm,
                            "dataset": dataset,
                            "doc_id": doc.doc_id,
                            "question_id": q.question_id,
                            **rec,
                            "routing": routing,
                        }
                    )

    result = {"config": cfg.to_dict(), "records": records}

    os.makedirs(cfg.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.results_dir, f"results_{seed}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    progress(f"[done] wrote {len(records)} records -> {out_path}")

    return result
