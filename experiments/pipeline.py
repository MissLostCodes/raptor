"""Integration layer: arm -> RetrievalAugmentation/FAISS Answerer.

The OFFLINE-TESTABLE seam here is :func:`make_ra_config`, which maps a chunking
arm name to a fully-wired ``RetrievalAugmentationConfig`` (verifying the right
chunker type lands on ``tree_builder_config.chunker``).

:func:`build_answerer` is HEAVY (it builds the RAPTOR tree or a FAISS index and
loads SBERT) and therefore runs on Colab, NOT in unit tests. It is injectable
via ``runner.run(build_answerer_fn=...)`` so the orchestration loop stays
offline-testable with fakes.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

from raptor.chunking import get_chunker
from raptor.RetrievalAugmentation import (
    RetrievalAugmentation,
    RetrievalAugmentationConfig,
)

TREE_ARMS = ("token", "structure", "semantic", "ahc")


def make_ra_config(
    arm: str,
    cfg,
    summarization_model,
    qa_model,
    parse_fn=None,
    embed_fn=None,
) -> RetrievalAugmentationConfig:
    """Build a ``RetrievalAugmentationConfig`` wired with the arm's chunker.

    Offline-testable: pass a fake ``qa_model`` (subclassing ``BaseQAModel``) and
    fake ``parse_fn`` / ``embed_fn`` to avoid any network or model load.

    Raises ``ValueError`` for the 'flat' arm (handled by FAISS, not a tree) and
    for any unknown arm.
    """
    if arm == "flat":
        raise ValueError(
            "arm 'flat' is not a tree arm; build a FaissRetriever via build_answerer."
        )
    if arm not in TREE_ARMS:
        raise ValueError(
            f"Unknown arm: {arm!r} (expected one of {TREE_ARMS} or 'flat')"
        )

    chunker = get_chunker(
        arm,
        max_tokens=cfg.leaf_max_tokens,
        model=cfg.model,
        cache_dir=cfg.cache_dir,
        parse_fn=parse_fn,
        embed_fn=embed_fn,
        tau=cfg.tau,
    )
    return RetrievalAugmentationConfig(
        qa_model=qa_model,
        tb_summarization_model=summarization_model,
        tb_max_tokens=cfg.leaf_max_tokens,
        tb_chunker=chunker,
    )


class Answerer(Protocol):
    """Answers a question, returning (answer, retrieved_context)."""

    def answer(self, question: str) -> Tuple[str, str]:
        ...

    def routing_info(self) -> dict:
        ...


class _TreeAnswerer:
    """Answerer backed by a built RAPTOR ``RetrievalAugmentation`` tree."""

    def __init__(self, ra, cfg, chunker=None):
        self._ra = ra
        self._cfg = cfg
        self._chunker = chunker  # for AHC routing capture

    def answer(self, question: str) -> Tuple[str, str]:
        # retrieve() returns (context, layer_info); we answer from that context
        # so we can also surface the retrieved context for reporting.
        context, _layer_info = self._ra.retrieve(
            question,
            max_tokens=self._cfg.retrieval_max_tokens,
            collapse_tree=True,
            return_layer_information=True,
        )
        answer = self._ra.qa_model.answer_question(context, question)
        return answer, context

    def routing_info(self) -> dict:
        info: dict = {}
        ch = self._chunker
        if ch is not None and hasattr(ch, "last_route"):
            info["route"] = getattr(ch, "last_route", None)
            info["score"] = getattr(ch, "last_score", None)
        return info


class _FlatAnswerer:
    """Answerer backed by a FAISS index over token chunks (flat arm)."""

    def __init__(self, retriever, qa_model):
        self._retriever = retriever
        self._qa_model = qa_model

    def answer(self, question: str) -> Tuple[str, str]:
        context = self._retriever.retrieve(question)
        answer = self._qa_model.answer_question(context, question)
        return answer, context

    def routing_info(self) -> dict:
        return {}


# NOTE: NOT unit-tested. Heavy path: builds the tree / FAISS index and loads
# SBERT. Runs on Colab. faiss + SBERT are imported lazily inside this function.
def build_answerer(
    arm: str,
    doc,
    cfg,
    summarization_model,
    qa_model,
    parse_fn=None,
    embed_fn=None,
) -> Answerer:
    """Build the per-document Answerer for ``arm`` (HEAVY)."""
    if arm in TREE_ARMS:
        ra_config = make_ra_config(
            arm, cfg, summarization_model, qa_model, parse_fn=parse_fn, embed_fn=embed_fn
        )
        # Fresh RetrievalAugmentation each time => tree is None => no overwrite
        # prompt on add_documents.
        ra = RetrievalAugmentation(config=ra_config)
        ra.add_documents(doc.text)
        chunker = ra_config.tree_builder_config.chunker
        return _TreeAnswerer(ra, cfg, chunker=chunker)

    if arm == "flat":
        from raptor.FaissRetriever import FaissRetriever, FaissRetrieverConfig
        from raptor.EmbeddingModels import SBertEmbeddingModel

        token_chunks = get_chunker(
            "token", max_tokens=cfg.leaf_max_tokens, model=cfg.model, cache_dir=cfg.cache_dir
        ).chunk(doc.text)

        faiss_config = FaissRetrieverConfig(
            max_tokens=cfg.leaf_max_tokens,
            max_context_tokens=cfg.retrieval_max_tokens,
            embedding_model=SBertEmbeddingModel(),
        )
        retriever = FaissRetriever(faiss_config)
        _build_faiss_from_chunks(retriever, token_chunks)
        return _FlatAnswerer(retriever, qa_model)

    raise ValueError(f"Unknown arm: {arm!r}")


def _build_faiss_from_chunks(retriever, chunks):
    """Populate a FaissRetriever's index from pre-computed token chunks."""
    import numpy as np
    import faiss

    retriever.context_chunks = np.array(chunks)
    embeddings = [retriever.embedding_model.create_embedding(c) for c in chunks]
    retriever.embeddings = np.array(embeddings, dtype=np.float32)
    retriever.index = faiss.IndexFlatIP(retriever.embeddings.shape[1])
    retriever.index.add(retriever.embeddings)
