"""Nested-oracle decomposition of adaptive-chunking headroom.

Our prior result (``paper/main.tex``) showed the per-*document* optimal leaf size
is not an exploitable function of any document-intrinsic property. This module
answers the follow-up the paper only conjectured: **where does the oracle headroom
actually live?** It attributes the gap between a single tuned global leaf size and
a per-question oracle to named, separately-measurable components:

    total = document + query-type + interaction + within-document(ceiling residual)

The whole apparatus is one primitive, :func:`oracle_by_key`: an oracle that is
allowed to choose a leaf size per *group* (groups defined by a key function over
questions). Every rung of the nested-oracle ladder is that primitive with a
different key:

    key = const           -> O0  best global fixed size
    key = doc_id          -> O1  per-document oracle
    key = query type      -> O2  per-query-type oracle
    key = (doc_id, type)  -> O3  per (document x type) oracle
    key = question id     -> O4  per-question oracle (the absolute ceiling)

Everything here is **stdlib only** (no numpy / sklearn / network), so it imports
instantly and is fully unit-testable offline.

Cell schema (one dict per ``(doc_id, qid, size)``)::

    {"doc_id": str, "qid": str, "size": int, "coverage": float, ...metadata}

``qid`` need only be unique *within* a document; a question's identity is the pair
``(doc_id, qid)``. Metadata keys (e.g. evidence multiplicity ``m``, ``answer_type``)
are carried through so key functions can group on them.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, List, Tuple


# ---------------------------------------------------------------------------
# Adapter: enriched sweep records -> flat per-question cells.
# ---------------------------------------------------------------------------
def cells_from_records(records: List[dict]) -> List[dict]:
    """Explode enriched sweep records into per-question cells.

    An enriched ``sweep_document`` record carries a ``per_question`` list::

        {"doc_id", "size", ..., "per_question": [{"qid", "coverage", ...meta}, ...]}

    Each per-question entry becomes one cell ``{doc_id, qid, size, coverage,
    ...meta}``. Records lacking ``per_question`` (legacy, mean-only records) and
    entries whose ``coverage`` is ``None`` are dropped, mirroring how the pure
    calibration path ignores unscored cells.
    """
    # Keyed by (doc_id, qid, size); a resumed/merged sweep may repeat a
    # (doc, size) record, so the later occurrence overwrites the earlier one,
    # matching coverage_matrix's last-wins semantics.
    by_key: Dict[Tuple, dict] = {}
    for r in records:
        for pq in r.get("per_question") or []:
            if pq.get("coverage") is None:
                continue
            cell = {"doc_id": r["doc_id"], "size": r["size"]}
            cell.update(pq)  # qid, coverage, and any metadata (m, answer_type, ...)
            by_key[(cell["doc_id"], cell.get("qid"), cell["size"])] = cell
    return list(by_key.values())


# ---------------------------------------------------------------------------
# The one primitive: an oracle that picks a size per group.
# ---------------------------------------------------------------------------
def oracle_by_key(
    cells: List[dict],
    key_fn: Callable[[dict], Hashable],
) -> Tuple[float, Dict[Hashable, int]]:
    """Oracle that assigns one leaf size per group, scored over all questions.

    For each group (``key_fn(cell)``), choose the size maximizing the group's mean
    per-question coverage (ties broken toward the smaller size). Every question in
    the group is then scored at that chosen size, and the returned coverage is the
    mean over *all questions* of the coverage at its group's chosen size.

    Because every rung of the nested-oracle ladder is scored as ``mean over the
    same set of questions``, the rungs are directly comparable and their gaps are
    the headroom components. Returns ``(mean_coverage, {group_key: chosen_size})``;
    ``(0.0, {})`` for empty input.
    """
    # group -> size -> list of per-question coverages at that size
    by_group: Dict[Hashable, Dict[int, List[float]]] = {}
    # group -> chosen size (filled after we pick per group)
    for c in cells:
        g = key_fn(c)
        by_group.setdefault(g, {}).setdefault(c["size"], []).append(c["coverage"])

    chosen: Dict[Hashable, int] = {}
    for g, size_to_covs in by_group.items():
        # best size = argmax mean coverage, tie -> smaller size.
        best_size = min(
            size_to_covs,
            key=lambda s: (-(sum(size_to_covs[s]) / len(size_to_covs[s])), s),
        )
        chosen[g] = best_size

    # Score: mean over all questions of coverage at its group's chosen size.
    covs: List[float] = []
    for c in cells:
        g = key_fn(c)
        if c["size"] == chosen[g]:
            covs.append(c["coverage"])
    mean_cov = (sum(covs) / len(covs)) if covs else 0.0
    return mean_cov, chosen


# ---------------------------------------------------------------------------
# The ledger: attribute total headroom to named components.
# ---------------------------------------------------------------------------
def decompose(
    cells: List[dict],
    type_fn: Callable[[dict], Hashable],
) -> dict:
    """Attribute the oracle headroom to document / query-type / interaction /
    within-document components via the nested-oracle ladder.

    ``type_fn`` maps a question to its query-type label (e.g. evidence multiplicity
    bucket or answer type). Returns the five rung coverages (``O0_global`` ..
    ``O4_per_question``) and a ``components`` sub-dict whose entries sum to
    ``total`` (= ``O4 - O0``):

    * ``document``    = O1 - O0   (gain from knowing the document)
    * ``query_type``  = O2 - O0   (gain from knowing the query type)
    * ``interaction`` = O3 - O1 - O2 + O0   (ANOVA-style cross term)
    * ``within_doc``  = O4 - O3   (residual only the individual question explains)

    Note ``document + query_type + interaction + within_doc`` telescopes exactly to
    ``O4 - O0 = total`` (the O3 terms cancel), so the ledger is additive by
    construction. ``within_doc`` is the hard ceiling on any policy that conditions
    only on document and query type -- the headroom no such feature can capture.
    """
    o0, _ = oracle_by_key(cells, lambda c: "all")
    o1, _ = oracle_by_key(cells, lambda c: c["doc_id"])
    o2, _ = oracle_by_key(cells, type_fn)
    o3, _ = oracle_by_key(cells, lambda c: (c["doc_id"], type_fn(c)))
    o4, _ = oracle_by_key(cells, lambda c: (c["doc_id"], c["qid"]))

    return {
        "O0_global": o0,
        "O1_document": o1,
        "O2_query_type": o2,
        "O3_doc_x_type": o3,
        "O4_per_question": o4,
        "components": {
            "document": o1 - o0,
            "query_type": o2 - o0,
            "interaction": o3 - o1 - o2 + o0,
            "within_doc": o4 - o3,
            "total": o4 - o0,
        },
    }
