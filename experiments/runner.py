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
from experiments.datasets.base import Document, QAExample, load_subset_ids
from experiments.metrics import (
    extract_choice,
    gold_evidence_coverage,
    narrativeqa_metrics,
    qasper_answer_f1,
)
from experiments.models import ModerationBlocked


def _maybe_tqdm(iterable, show, **kw):
    """Wrap ``iterable`` in a tqdm bar when requested and available.

    Degrades gracefully to the bare iterable when ``show`` is False or tqdm is
    not installed, so the loop runs identically in headless/test environments.
    """
    if not show:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kw)
    except Exception:
        return iterable


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
def score_question(
    dataset_name: str,
    example: QAExample,
    prediction: str,
    retrieved_context: str = "",
) -> dict:
    """Return a per-question metric record for the dataset.

    - qasper: {answer_f1, evidence_f1, evidence_coverage}.
    - quality: {choice, correct, is_hard}
    - narrativeqa: {rouge_l, bleu_1, bleu_4, meteor}

    ``retrieved_context`` is the concatenated context the answerer retrieved; it
    is used to compute the gold-evidence token-coverage proxy for QASPER.
    """
    if dataset_name == "qasper":
        rec = {
            "answer_f1": qasper_answer_f1(
                prediction, example.gold_answers, example.is_unanswerable
            )
        }
        # Paragraph-set evidence-F1 needs discrete retrieved evidence paragraphs
        # that string-match the gold paragraphs; a hierarchical-RAG answerer only
        # exposes a concatenated chunk context, so set-overlap F1 is inapplicable
        # (kept None). The computable, meaningful signal is token-coverage: what
        # fraction of gold-evidence tokens the retrieval surfaced.
        rec["evidence_f1"] = None
        rec["evidence_coverage"] = gold_evidence_coverage(
            retrieved_context, example.evidence
        )
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
    resume: bool = True,
    show_progress: bool = True,
    subsets_dir: Optional[str] = None,
    allow_unpinned: bool = False,
) -> dict:
    """Run the full experiment grid and write results to disk.

    The loop is hardened against the failure modes that killed a long Colab run:

    - **Per-question isolation:** a moderation block or any other answer-time
      error is recorded (as a ``blocked``/``error`` record with no score fields)
      and the grid keeps going instead of crashing.
    - **Per-doc isolation:** a doc whose answerer BUILD fails is skipped (no
      records) and is NOT marked done, so a later resume retries it.
    - **Checkpoint + resume:** results are written after every doc; on restart
      with ``resume=True`` already-completed ``(dataset, arm, doc)`` are skipped
      and their records are loaded from disk.

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
        resume: when True, seed records from an existing results file and skip
            ``(dataset, arm, doc)`` that are already complete.
        show_progress: when True, wrap the per-doc loop in a tqdm bar.
        subsets_dir: directory of pinned-subset JSONs (defaults to the
            version-controlled ``experiments/datasets/subsets``).
        allow_unpinned: when False (default), a dataset with no pinned subset is
            a hard error -- a real run can never silently evaluate an
            unreproducible document set. Set True to fall back to the first
            ``subset_sizes[dataset]`` docs by load order (NOT reproducible).

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

    out_path = os.path.join(cfg.results_dir, f"results_{seed}.json")

    def _write():
        os.makedirs(cfg.results_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"config": cfg.to_dict(), "records": records}, fh, indent=2)

    records: List[dict] = []
    done_docs = set()
    if resume and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                prev = json.load(fh)
            records = prev.get("records", [])
            done_docs = {
                (r.get("dataset"), r.get("arm"), r.get("doc_id")) for r in records
            }
        except Exception:
            records, done_docs = [], set()

    n_ok = n_blocked = n_errored = 0

    for dataset in cfg.datasets:
        progress(f"[dataset] {dataset}")
        loader = (loaders or {}).get(dataset) if loaders else None
        if loader is None:
            loader = get_loader(dataset)
        n = cfg.subset_sizes.get(dataset)

        # Reproducibility guard: load the version-controlled pinned subset. An
        # unpinned dataset is a hard error unless explicitly opted out of.
        pinned = load_subset_ids(dataset, subsets_dir)
        if pinned:
            documents = loader.load(subset_ids=pinned)
        elif allow_unpinned:
            progress(
                f"[WARN] {dataset}: no pinned subset; using the first {n} docs "
                f"by load order (NOT reproducible)."
            )
            documents = loader.load(limit=n)
        else:
            raise RuntimeError(
                f"No pinned subset for {dataset!r}: "
                f"{os.path.join(subsets_dir or 'experiments/datasets/subsets', dataset + '_ids.json')} "
                f"is missing or empty. Pin it with experiments/select_subsets.py "
                f"(reproducible), or pass allow_unpinned=True to evaluate the "
                f"first {n} docs by load order (NOT reproducible)."
            )

        # QA model carries the dataset-specific prompt.
        if qa_models is not None and dataset in qa_models:
            qa_model = qa_models[dataset]
        elif real_build:
            qa_model = make_qa_model(cfg, *qa_prompt_for(dataset))
        else:
            qa_model = None  # fake build_answerer_fn won't use it

        for arm in cfg.arms:
            progress(f"  [arm] {arm}")
            for doc in _maybe_tqdm(
                documents, show_progress, desc=f"{dataset}/{arm}", leave=False
            ):
                if (dataset, arm, doc.doc_id) in done_docs:
                    continue

                try:
                    answerer = build_answerer_fn(
                        arm, doc, cfg, summarization_model, qa_model
                    )
                except Exception as exc:
                    # Build failure: skip the doc entirely, leave it NOT done so
                    # a later resume retries it.
                    progress(
                        f"[skip-doc] {dataset}/{arm}/{doc.doc_id} "
                        f"build failed: {exc!r}"
                    )
                    continue

                doc_records: List[dict] = []
                for q in doc.questions:
                    base = {
                        "arm": arm,
                        "dataset": dataset,
                        "doc_id": doc.doc_id,
                        "question_id": q.question_id,
                    }
                    try:
                        pred, retrieved_context = answerer.answer(
                            compose_query(dataset, q)
                        )
                    except ModerationBlocked as exc:
                        doc_records.append(
                            {**base, "blocked": True, "error": "moderation",
                             "routing": {}}
                        )
                        n_blocked += 1
                        continue
                    except Exception as exc:
                        doc_records.append(
                            {**base, "blocked": False,
                             "error": type(exc).__name__, "routing": {}}
                        )
                        n_errored += 1
                        continue

                    rec = score_question(dataset, q, pred, retrieved_context)
                    routing = {}
                    if hasattr(answerer, "routing_info"):
                        routing = answerer.routing_info() or {}
                    doc_records.append(
                        {**base, **rec, "blocked": False, "error": None,
                         "routing": routing}
                    )
                    n_ok += 1

                records.extend(doc_records)
                done_docs.add((dataset, arm, doc.doc_id))
                _write()  # checkpoint after each doc

    _write()
    progress(
        f"[done] {n_ok} answered, {n_blocked} moderation-blocked, "
        f"{n_errored} errored -> {out_path}"
    )

    return {"config": cfg.to_dict(), "records": records}
