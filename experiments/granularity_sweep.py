"""Leaf-granularity sweep — a cheap GO/NO-GO de-risk experiment.

Question this answers: **does leaf chunk size materially affect retrieval of
QASPER gold-evidence, and is the optimal size document-dependent?** If size
barely moves coverage, or every document peaks at the same size, there is no
"adaptive granularity" story to tell and we should just pick a fixed size. If
the best size varies *across documents*, that heterogeneity is the green light
to build a density-driven adaptive chunker.

Deliberately cheap: **NO LLM**. We chunk each document at several token sizes,
build a flat FAISS index over the chunks using SBERT embeddings (reusing the
project's existing flat-retrieval machinery so the result is authentic), run the
real questions through ``FaissRetriever.retrieve``, and measure how much of the
gold evidence ends up inside the retrieved context.

Two layers:

* PURE metric functions (``normalize`` / ``paragraph_covered`` /
  ``evidence_coverage`` / ``best_size_per_doc``) — no SBERT, no FAISS, fully
  unit-tested and deterministic.
* HEAVY sweep driver (``build_flat_retriever`` / ``sweep_document`` /
  ``run_sweep``) — loads SBERT + FAISS; guarded imports live *inside* the
  functions (mirroring ``experiments.pipeline``) so importing this module stays
  cheap and offline-safe.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

# Punctuation we strip when deciding whether a "token" is content-bearing.
import string as _string

_PUNCT = set(_string.punctuation)


# ---------------------------------------------------------------------------
# PURE metric functions (unit-tested; no SBERT / FAISS / network).
# ---------------------------------------------------------------------------
def normalize(s: str) -> List[str]:
    """Lowercase, strip, split on whitespace; drop punctuation-only tokens.

    Returns the list of content-bearing tokens. A token that is *entirely*
    punctuation (e.g. ``"-"``, ``"()"``) is dropped; punctuation attached to a
    word (e.g. ``"model,"``) is kept as-is so the comparison stays simple and
    symmetric between gold and context (both go through this same function).
    """
    out: List[str] = []
    for tok in (s or "").lower().strip().split():
        if all(ch in _PUNCT for ch in tok):
            continue
        out.append(tok)
    return out


def _token_recall(gold: str, context: str) -> float:
    """Multiset token recall of ``gold`` within ``context``, in [0, 1].

    Token recall = (count of gold tokens also present in the context, bounded by
    their multiplicity in the context) / (number of gold tokens), via the multiset
    intersection ``Counter(gold) & Counter(context)``. An empty gold string (no
    content tokens) recalls trivially (``1.0``). Shared by ``paragraph_covered``
    (gold-evidence proxy) and ``answer_coverage`` (answer-recall proxy).
    """
    from collections import Counter

    gold_tokens = normalize(gold)
    if not gold_tokens:
        return 1.0  # nothing to recall
    ctx_counts = Counter(normalize(context))
    gold_counts = Counter(gold_tokens)
    overlap = sum((gold_counts & ctx_counts).values())
    return overlap / len(gold_tokens)


def paragraph_covered(gold_para: str, context: str, threshold: float = 0.8) -> bool:
    """Is ``gold_para`` "covered" by ``context`` at >= ``threshold`` token recall?

    Thin wrapper over :func:`_token_recall`. Why token-recall and not
    ``evidence_f1``: ``evidence_f1`` is a *set overlap of whole paragraph strings*
    — it only scores 1.0 when the retriever returns the gold paragraph as a
    byte-identical element. Here the retriever returns *token-chunked* text
    (paragraph boundaries are gone; a gold paragraph may be split across chunks or
    padded with neighbours), so exact-string set overlap would be ~0 even when the
    gold content is fully present. An empty gold paragraph is trivially covered.
    """
    return _token_recall(gold_para, context) >= threshold


def answer_coverage(
    gold_answers: List[str], context: str, threshold: float = 0.8
) -> Optional[float]:
    """Answer-recall proxy: best token-recall of any gold answer within ``context``.

    Generalizes :func:`evidence_coverage` to corpora that ship gold *answers*
    rather than gold *evidence paragraphs* (QuALITY's correct option text,
    NarrativeQA's reference answers). Returns the maximum :func:`_token_recall`
    over the (non-empty) reference answers — any reference counts — as a
    continuous score in [0, 1], or ``None`` when there is no usable gold answer
    (so the caller skips the question, mirroring ``evidence_coverage``). The
    ``threshold`` argument is accepted for signature parity with the evidence
    proxy but is not applied (the score is continuous, giving a smoother
    optimal-size target).
    """
    answers = [a for a in (gold_answers or []) if a and a.strip()]
    if not answers:
        return None
    return max(_token_recall(a, context) for a in answers)


def evidence_coverage(
    gold_paras: List[str], context: str, threshold: float = 0.8
) -> Optional[float]:
    """Fraction of gold evidence paragraphs covered by ``context``.

    Returns ``None`` when ``gold_paras`` is empty (an unanswerable / no-evidence
    question) so the caller can skip it from the coverage metric. Otherwise
    returns the fraction of paragraphs for which
    ``paragraph_covered(..., threshold)`` is True.
    """
    paras = [p for p in (gold_paras or []) if p and p.strip()]
    if not paras:
        return None
    covered = sum(1 for p in paras if paragraph_covered(p, context, threshold))
    return covered / len(paras)


def _evidence_coverage_q(q, context: str, threshold: float) -> Optional[float]:
    """Per-question adapter: gold-evidence coverage (the QASPER default path)."""
    return evidence_coverage(getattr(q, "evidence", []) or [], context, threshold)


def _answer_coverage_q(q, context: str, threshold: float) -> Optional[float]:
    """Per-question adapter: answer-recall coverage (QuALITY / NarrativeQA path)."""
    return answer_coverage(getattr(q, "gold_answers", []) or [], context, threshold)


def best_size_per_doc(records: List[dict]) -> Dict[str, int]:
    """Argmax ``mean_evidence_coverage`` per ``doc_id`` — the heterogeneity signal.

    ``records`` is the flat list returned by :func:`run_sweep` / one dict per
    ``(doc_id, size)``. For each document we pick the size with the highest mean
    coverage (ties broken by the smaller size, since a smaller leaf that ties on
    coverage is the more precise/parsimonious choice). Records whose
    ``mean_evidence_coverage`` is ``None`` (no answerable questions) are ignored;
    a document with only such records is omitted from the result.
    """
    by_doc: Dict[str, List[dict]] = {}
    for r in records:
        if r.get("mean_evidence_coverage") is None:
            continue
        by_doc.setdefault(r["doc_id"], []).append(r)

    best: Dict[str, int] = {}
    for doc_id, recs in by_doc.items():
        # Highest coverage first; on a tie prefer the smaller size.
        winner = max(recs, key=lambda r: (r["mean_evidence_coverage"], -r["size"]))
        best[doc_id] = winner["size"]
    return best


# ---------------------------------------------------------------------------
# HEAVY sweep driver (SBERT + FAISS; NOT unit-tested). Heavy imports are guarded
# inside the functions, exactly like experiments.pipeline.build_answerer.
# ---------------------------------------------------------------------------
def build_flat_retriever(chunks: List[str], budget: int):
    """Build a flat ``FaissRetriever`` over ``chunks`` (HEAVY: loads SBERT + FAISS).

    Reuses ``experiments.pipeline._build_faiss_from_chunks`` (the project's
    canonical "populate a FaissRetriever from pre-computed chunks" helper) when
    importable, else replicates it. ``max_tokens`` in the config is the chunk
    size proxy used by ``retrieve`` to size its candidate pool; we set it from
    the chunks themselves. ``max_context_tokens`` is the retrieval ``budget``.
    """
    from raptor.FaissRetriever import FaissRetriever, FaissRetrieverConfig
    from raptor.EmbeddingModels import SBertEmbeddingModel

    try:
        from experiments.pipeline import _build_faiss_from_chunks
    except Exception:  # pragma: no cover - defensive fallback
        def _build_faiss_from_chunks(retriever, chunks):
            import numpy as np
            import faiss

            retriever.context_chunks = np.array(chunks)
            embeddings = [
                retriever.embedding_model.create_embedding(c) for c in chunks
            ]
            retriever.embeddings = np.array(embeddings, dtype=np.float32)
            retriever.index = faiss.IndexFlatIP(retriever.embeddings.shape[1])
            retriever.index.add(retriever.embeddings)

    # FaissRetrieverConfig requires max_tokens >= 1. ``retrieve`` computes its
    # candidate pool size as max_context_tokens / max_tokens, so max_tokens must
    # be a per-chunk token estimate. Use the configured chunk size if available,
    # otherwise a safe floor of 1.
    config = FaissRetrieverConfig(
        max_tokens=max(1, _estimate_chunk_tokens(chunks)),
        max_context_tokens=budget,
        embedding_model=SBertEmbeddingModel(),
    )
    retriever = FaissRetriever(config)
    _build_faiss_from_chunks(retriever, chunks)
    return retriever


def _estimate_chunk_tokens(chunks: List[str]) -> int:
    """Cheap upper-ish estimate of tokens-per-chunk for FAISS pool sizing.

    We use the tokenizer-free heuristic of the max whitespace-word count over
    the chunks. Slightly conservative (real BPE tokens >= words), which only
    makes ``retrieve`` pull a *larger* candidate pool — never smaller — so we
    never under-retrieve relative to the budget.
    """
    if not chunks:
        return 1
    return max(1, max(len(c.split()) for c in chunks))


def sweep_document(
    doc,
    sizes: List[int],
    budget: int = 2000,
    cover_threshold: float = 0.8,
    coverage_fn=_evidence_coverage_q,
) -> List[dict]:
    """Sweep one document across leaf ``sizes`` (HEAVY).

    For each size S: token-chunk ``doc.text`` at ``max_tokens=S``, build a flat
    SBERT+FAISS retriever, and for every *scored* question retrieve the context
    and compute ``coverage_fn(q, ctx, cover_threshold)``. Returns one dict per
    size::

        {doc_id, size, n_chunks, n_questions, mean_evidence_coverage}

    where ``n_questions`` counts only the questions ``coverage_fn`` scored (those
    for which it returned non-``None``) and ``mean_evidence_coverage`` is the mean
    over them (``None`` if none, or no chunks).

    ``coverage_fn(question, context, threshold) -> Optional[float]`` selects the
    retrieval-quality proxy. The default :func:`_evidence_coverage_q` is the
    QASPER gold-evidence proxy (unchanged behavior); pass
    :func:`_answer_coverage_q` for the answer-recall proxy used on QuALITY /
    NarrativeQA. The record key stays ``mean_evidence_coverage`` regardless of
    proxy so all downstream calibration is unchanged.
    """
    from raptor.chunking.token_chunker import TokenChunker

    records: List[dict] = []
    for size in sizes:
        chunks = TokenChunker(max_tokens=size).chunk(doc.text)
        n_chunks = len(chunks)

        if n_chunks == 0:
            records.append(
                {
                    "doc_id": doc.doc_id,
                    "size": size,
                    "n_chunks": 0,
                    "n_questions": 0,
                    "mean_evidence_coverage": None,
                }
            )
            continue

        retriever = build_flat_retriever(chunks, budget)

        coverages: List[float] = []
        for q in doc.questions:
            ctx = retriever.retrieve(q.question)
            cov = coverage_fn(q, ctx, cover_threshold)
            if cov is None:  # unscored (no gold evidence / no gold answer) -> skip
                continue
            coverages.append(cov)

        mean_cov = (sum(coverages) / len(coverages)) if coverages else None
        records.append(
            {
                "doc_id": doc.doc_id,
                "size": size,
                "n_chunks": n_chunks,
                "n_questions": len(coverages),
                "mean_evidence_coverage": mean_cov,
            }
        )
    return records


def _load_existing(out_path: Optional[str]) -> List[dict]:
    """Load already-written records from ``out_path`` (resume support)."""
    if not out_path or not os.path.exists(out_path):
        return []
    try:
        with open(out_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def run_sweep(
    docs,
    sizes: List[int],
    budget: int = 2000,
    cover_threshold: float = 0.8,
    out_path: Optional[str] = None,
    progress=print,
    coverage_fn=_evidence_coverage_q,
) -> List[dict]:
    """Run the granularity sweep over ``docs`` (HEAVY).

    Loops documents and sizes, collecting one record per ``(doc_id, size)``.
    Resumable: if ``out_path`` is given, already-present ``(doc_id, size)`` pairs
    are skipped and the JSON file is rewritten after each document so a crash
    costs at most one document. ``progress`` (default ``print``) is called with a
    short status string per document — pass ``tqdm.write`` or a custom logger to
    integrate with a progress bar.
    """
    records: List[dict] = _load_existing(out_path)
    done = {(r["doc_id"], r["size"]) for r in records}

    sizes = list(sizes)
    for di, doc in enumerate(docs):
        pending = [s for s in sizes if (doc.doc_id, s) not in done]
        if not pending:
            if progress:
                progress(
                    f"[granularity_sweep] doc {di + 1}/{len(docs)} "
                    f"{doc.doc_id}: already complete, skipping"
                )
            continue

        if progress:
            progress(
                f"[granularity_sweep] doc {di + 1}/{len(docs)} {doc.doc_id}: "
                f"sweeping sizes {pending}"
            )

        new_recs = sweep_document(doc, pending, budget, cover_threshold, coverage_fn)
        for r in new_recs:
            done.add((r["doc_id"], r["size"]))
        records.extend(new_recs)

        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)

    return records
