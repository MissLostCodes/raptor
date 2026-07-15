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
# A single shared embedding model is reused across every (doc, size) so the
# 438 MB SBERT weights are loaded ONCE, not per retriever. Building a fresh
# SBertEmbeddingModel() per call leaks GPU memory (each lazily loads its own
# SentenceTransformer) and OOMs a free-tier T4 after a dozen documents.
_SHARED_EMBEDDER = None


def get_shared_embedder():
    """Return a process-wide singleton ``SBertEmbeddingModel`` (HEAVY, lazy).

    The first call constructs it (the model itself loads on its first embedding);
    later calls return the same instance, so the whole sweep shares one model on
    the GPU instead of one per (doc, size). Call :func:`reset_shared_embedder` to
    drop it (e.g. to free the GPU between corpora).
    """
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        from raptor.EmbeddingModels import SBertEmbeddingModel

        _SHARED_EMBEDDER = SBertEmbeddingModel()
    return _SHARED_EMBEDDER


def reset_shared_embedder() -> None:
    """Drop the shared embedder singleton (next call rebuilds it)."""
    global _SHARED_EMBEDDER
    _SHARED_EMBEDDER = None


def build_flat_retriever(chunks: List[str], budget: int, embedding_model=None):
    """Build a flat ``FaissRetriever`` over ``chunks`` (HEAVY: SBERT + FAISS).

    Reuses ``experiments.pipeline._build_faiss_from_chunks`` (the project's
    canonical "populate a FaissRetriever from pre-computed chunks" helper) when
    importable, else replicates it. ``max_tokens`` in the config is the chunk
    size proxy used by ``retrieve`` to size its candidate pool; we set it from
    the chunks themselves. ``max_context_tokens`` is the retrieval ``budget``.

    ``embedding_model`` defaults to the shared singleton
    (:func:`get_shared_embedder`) so the SBERT weights are loaded once for the
    whole sweep; pass an explicit model to override.
    """
    from raptor.FaissRetriever import FaissRetriever, FaissRetrieverConfig

    if embedding_model is None:
        embedding_model = get_shared_embedder()

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
        embedding_model=embedding_model,
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


def null_qa_model():
    """A ``BaseQAModel`` that must never be called (tiny object, HEAVY import).

    The sweep scores RETRIEVAL with a token-overlap proxy and never answers, but
    ``RetrievalAugmentationConfig`` defaults a missing ``qa_model`` to
    ``GPT3TurboQAModel()``, which would demand an OpenAI key. This satisfies the
    isinstance check and raises loudly if anything ever tries to answer -- which
    also proves the "no QA calls" cost claim rather than assuming it.
    """
    from raptor.QAModels import BaseQAModel

    class _NullQAModel(BaseQAModel):
        def answer_question(self, context, question, *args, **kwargs):
            raise AssertionError("QA model called during a retrieval-only sweep")

    return _NullQAModel()


def build_tree_retriever(
    doc, size: int, budget: int, summarization_model, embedding_model=None
):
    """Build a REAL RAPTOR tree over ``doc`` at leaf ``size`` (HEAVY, LLM calls).

    Returns ``(retrieve, n_leaves, n_layers)`` where ``retrieve(question) -> str``
    runs collapsed-tree retrieval under ``budget`` tokens -- the same signature as
    ``build_flat_retriever(...).retrieve``, so :func:`sweep_document` can swap
    between the flat proxy and the real hierarchy with no other change.

    ``n_layers`` is the tree's depth AFTER construction: ``construct_tree`` breaks
    out once a layer is down to ``reduction_dimension + 1`` (11) nodes and writes
    the achieved depth back. It can be **0** -- a short document at a coarse leaf
    size never clusters at all, so "RAPTOR" degenerates to a flat pile of leaves.
    Recording it per cell is the point: leaf size co-determines tree depth, so the
    two cannot be varied independently.
    """
    import dataclasses

    from experiments.config import ExperimentConfig
    from experiments.pipeline import build_answerer

    cfg = dataclasses.replace(
        ExperimentConfig(), leaf_max_tokens=size, retrieval_max_tokens=budget
    )
    answerer = build_answerer(
        "token",
        doc,
        cfg,
        summarization_model,
        null_qa_model(),
        embedding_model=embedding_model or get_shared_embedder(),
    )
    ra = answerer._ra
    tree = ra.tree

    def retrieve(question: str) -> str:
        return ra.retrieve(
            question,
            max_tokens=budget,
            collapse_tree=True,
            return_layer_information=False,
        )

    return retrieve, len(getattr(tree, "leaf_nodes", None) or {}), tree.num_layers


def question_meta(q) -> dict:
    """Per-question identity + metadata for the headroom decomposition.

    Returns ``{"qid": question_id, "m": evidence_multiplicity}`` where ``m`` is the
    number of gold-evidence paragraphs (the SLIDERS "aggregation" axis: 1 = lookup,
    higher = multi-hop / aggregation). Corpora without gold evidence (answer-recall
    proxy on QuALITY) simply carry ``m = 0``. Tolerant of missing attributes so it
    works across loaders.
    """
    return {
        "qid": getattr(q, "question_id", None),
        "m": len(getattr(q, "evidence", []) or []),
    }


def sweep_document(
    doc,
    sizes: List[int],
    budget: int = 2000,
    cover_threshold: float = 0.8,
    coverage_fn=_evidence_coverage_q,
    embedding_model=None,
    summarization_model=None,
) -> List[dict]:
    """Sweep one document across leaf ``sizes`` (HEAVY).

    For each size S: token-chunk ``doc.text`` at ``max_tokens=S``, build a
    retriever, and for every *scored* question retrieve the context and compute
    ``coverage_fn(q, ctx, cover_threshold)``. Returns one dict per size::

        {doc_id, size, n_chunks, n_questions, mean_evidence_coverage, per_question,
         n_layers}

    ``summarization_model`` selects the retriever, and it is the whole
    flat-vs-hierarchical question:

    * ``None`` (default) -- a flat SBERT+FAISS index over the chunks. No LLM, no
      tree. Every result in the paper to date is this path.
    * set -- a REAL RAPTOR tree (clusters + LLM summaries + collapsed-tree
      retrieval). Costs summarization calls at build time; scoring stays LLM-free.

    ``n_layers`` is the achieved tree depth (``None`` in flat mode, and possibly
    ``0`` in tree mode when a document never clusters at the given leaf size)

    where ``n_questions`` counts only the questions ``coverage_fn`` scored (those
    for which it returned non-``None``) and ``mean_evidence_coverage`` is the mean
    over them (``None`` if none, or no chunks). ``per_question`` is a list of
    ``{qid, m, coverage}`` (one per scored question) feeding the nested-oracle
    headroom decomposition; it is additive metadata and leaves the existing
    ``mean_evidence_coverage`` calibration path unchanged.

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
        n_layers = None
        if summarization_model is None:
            chunks = TokenChunker(max_tokens=size).chunk(doc.text)
            n_chunks = len(chunks)
            retrieve = None
            if n_chunks:
                retrieve = build_flat_retriever(chunks, budget, embedding_model).retrieve
        else:
            retrieve, n_chunks, n_layers = build_tree_retriever(
                doc, size, budget, summarization_model, embedding_model
            )

        if n_chunks == 0:
            records.append(
                {
                    "doc_id": doc.doc_id,
                    "size": size,
                    "n_chunks": 0,
                    "n_questions": 0,
                    "mean_evidence_coverage": None,
                    "n_layers": n_layers,
                }
            )
            continue

        coverages: List[float] = []
        per_question: List[dict] = []
        for q in doc.questions:
            ctx = retrieve(q.question)
            cov = coverage_fn(q, ctx, cover_threshold)
            if cov is None:  # unscored (no gold evidence / no gold answer) -> skip
                continue
            coverages.append(cov)
            per_question.append({**question_meta(q), "coverage": cov})

        mean_cov = (sum(coverages) / len(coverages)) if coverages else None
        records.append(
            {
                "doc_id": doc.doc_id,
                "size": size,
                "n_chunks": n_chunks,
                "n_questions": len(coverages),
                "mean_evidence_coverage": mean_cov,
                # Per-question scores power the nested-oracle headroom decomposition
                # (experiments.headroom_decomposition); mean_evidence_coverage above
                # is unchanged, so all existing calibration keeps working verbatim.
                "per_question": per_question,
                # Achieved tree depth (None in flat mode). 0 means the document never
                # clustered at this leaf size -> not actually hierarchical.
                "n_layers": n_layers,
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
    embedding_model=None,
    summarization_model=None,
) -> List[dict]:
    """Run the granularity sweep over ``docs`` (HEAVY).

    Loops documents and sizes, collecting one record per ``(doc_id, size)``.
    Resumable: if ``out_path`` is given, already-present ``(doc_id, size)`` pairs
    are skipped and the JSON file is rewritten after each document so a crash
    costs at most one document. ``progress`` (default ``print``) is called with a
    short status string per document — pass ``tqdm.write`` or a custom logger to
    integrate with a progress bar.

    ``summarization_model=None`` keeps the flat proxy; set it to build REAL RAPTOR
    trees (see :func:`sweep_document`). Give each mode its OWN ``out_path``: the
    resume key is ``(doc_id, size)`` and carries no mode, so a tree run pointed at
    a flat cache would skip every document and hand back flat records labelled as
    tree results. Mixing is refused below rather than silently blended.
    """
    records: List[dict] = _load_existing(out_path)
    wants_tree = summarization_model is not None
    # Tree records carry an int n_layers (0 included); flat and legacy ones do not.
    cache_is_tree = any(r.get("n_layers") is not None for r in records)
    if records and cache_is_tree != wants_tree:
        raise ValueError(
            f"{out_path!r} holds {'tree' if cache_is_tree else 'flat'} records but this "
            f"is a {'tree' if wants_tree else 'flat'} sweep. Resuming would silently mix "
            f"retrieval modes. Use a separate out_path per mode."
        )
    done = {(r["doc_id"], r["size"]) for r in records}

    # Load the embedding model ONCE for the whole sweep (avoids per-(doc,size)
    # GPU leaks). Reused by every retriever built below, flat or tree.
    if embedding_model is None:
        embedding_model = get_shared_embedder()

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

        new_recs = sweep_document(
            doc,
            pending,
            budget,
            cover_threshold,
            coverage_fn,
            embedding_model,
            summarization_model,
        )
        for r in new_recs:
            done.add((r["doc_id"], r["size"]))
        records.extend(new_recs)

        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)

    return records
