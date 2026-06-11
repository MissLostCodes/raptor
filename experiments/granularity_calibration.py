"""Training-free calibration of a density->leaf-size map, and adaptive headroom.

A prior de-risk sweep (:mod:`experiments.granularity_sweep`) **confirmed H1**: the
optimal QASPER leaf size is strongly document-dependent (best size spans 50-400
tokens across papers). That heterogeneity is only worth exploiting if a cheap,
deterministic signal can *predict* the right size per document, and if doing so
captures meaningful coverage over the single best fixed size. This module turns
the sweep's records into that analysis.

It is split into two responsibilities, both pure and dependency-light:

* **Headroom accounting** -- :func:`coverage_matrix`, :func:`mean_coverage_at_size`,
  :func:`best_global_fixed`, :func:`oracle_per_doc`, :func:`headroom`,
  :func:`coverage_under_sizes`. These quantify the gap between the best single
  fixed leaf size (a non-adaptive baseline) and a per-document oracle (the ceiling
  any sizing policy could reach), and let you score an *arbitrary* size-per-doc
  policy by snapping each doc's chosen size onto the nearest size actually swept.

* **Calibration** -- :func:`rank_corr`, :func:`fit_rho_to_size`,
  :func:`predict_size`, :func:`train_test_split_docs`. These correlate the
  deterministic density features against the per-doc optimum, fit a parsimonious
  two-parameter ``size = a + b * rho`` map on a training split, and predict held-out
  leaf sizes from density alone -- no SBERT, no LLM, no training labels at inference.

Everything here is **stdlib + math + random only** (NO numpy / scipy / sklearn).
The learned multi-feature regressor baseline lives in the companion notebook, where
sklearn is available; keeping this module lean means it imports instantly and is
fully unit-testable offline.

Record schema (one dict per ``(doc_id, size)``, as emitted by
:func:`experiments.granularity_sweep.run_sweep`)::

    {"doc_id": str, "size": int, "n_questions": int,
     "mean_evidence_coverage": float | None}

Records whose ``mean_evidence_coverage`` is ``None`` (a document with no answerable
questions at that size) are ignored throughout.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Headroom accounting (pure; no models / network).
# ---------------------------------------------------------------------------
def coverage_matrix(records: List[dict]) -> Dict[str, Dict[int, float]]:
    """Pivot flat sweep records into ``{doc_id: {size: mean_evidence_coverage}}``.

    Records with ``mean_evidence_coverage is None`` are dropped. If the same
    ``(doc_id, size)`` appears more than once (e.g. a resumed/merged run), the
    last occurrence wins, mirroring a dict overwrite.
    """
    matrix: Dict[str, Dict[int, float]] = {}
    for r in records:
        cov = r.get("mean_evidence_coverage")
        if cov is None:
            continue
        matrix.setdefault(r["doc_id"], {})[r["size"]] = cov
    return matrix


def mean_coverage_at_size(records: List[dict], size: int) -> float:
    """Mean coverage at ``size`` over the docs that were swept at that size.

    Only documents that actually have a (non-``None``) record at ``size`` are
    averaged; documents lacking that size do not contribute and are not counted in
    the denominator. Returns ``0.0`` if no document has that size.
    """
    matrix = coverage_matrix(records)
    vals = [sizes[size] for sizes in matrix.values() if size in sizes]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def best_global_fixed(records: List[dict]) -> Tuple[int, float]:
    """Single leaf size maximizing mean coverage across docs, and that coverage.

    This is the strongest *non-adaptive* baseline: one size applied to every
    document. Returns ``(best_size, best_mean_coverage)``. Ties on mean coverage
    are broken toward the smaller size (a smaller leaf that ties is the more
    precise/parsimonious choice -- consistent with
    :func:`granularity_sweep.best_size_per_doc`). Returns ``(0, 0.0)`` when there
    are no scored records.
    """
    matrix = coverage_matrix(records)
    sizes = sorted({s for d in matrix.values() for s in d})
    if not sizes:
        return 0, 0.0

    best_size = sizes[0]
    best_cov = -1.0
    for s in sizes:  # ascending, so first (smallest) winner is kept on ties
        cov = mean_coverage_at_size(records, s)
        if cov > best_cov:
            best_cov = cov
            best_size = s
    return best_size, best_cov


def oracle_per_doc(records: List[dict]) -> Tuple[float, Dict[str, int]]:
    """Per-document argmax sizing: the ceiling any sizing policy could reach.

    For each document, pick the size with the highest coverage (ties -> smaller
    size). Returns ``(mean_of_those_per_doc_best_coverages, {doc_id: best_size})``.
    The mean is over documents (each doc's own best), NOT over ``(doc, size)``
    pairs, so it is directly comparable to :func:`best_global_fixed`'s coverage.
    Returns ``(0.0, {})`` when there are no scored records.
    """
    matrix = coverage_matrix(records)
    best_size: Dict[str, int] = {}
    best_covs: List[float] = []
    for doc_id, sizes in matrix.items():
        if not sizes:
            continue
        # max coverage, tie -> smaller size (negate size in the key).
        size, cov = max(sizes.items(), key=lambda kv: (kv[1], -kv[0]))
        best_size[doc_id] = size
        best_covs.append(cov)
    mean_cov = (sum(best_covs) / len(best_covs)) if best_covs else 0.0
    return mean_cov, best_size


def headroom(records: List[dict]) -> dict:
    """Quantify the adaptive opportunity: oracle coverage minus best fixed coverage.

    Returns::

        {"best_fixed_size": int, "best_fixed_cov": float,
         "oracle_cov": float, "abs_gap": float, "rel_gain_pct": float}

    where ``abs_gap = oracle_cov - best_fixed_cov`` (always >= 0, since the oracle
    can always fall back to any single fixed size) and ``rel_gain_pct`` is that gap
    as a percentage of the best fixed coverage (``0.0`` if the best fixed coverage
    is ``0``). This is the single headroom number the paper reports: the most an
    adaptive policy could possibly gain over the best fixed size.
    """
    best_fixed_size, best_fixed_cov = best_global_fixed(records)
    oracle_cov, _ = oracle_per_doc(records)
    abs_gap = oracle_cov - best_fixed_cov
    rel_gain_pct = (abs_gap / best_fixed_cov * 100.0) if best_fixed_cov else 0.0
    return {
        "best_fixed_size": best_fixed_size,
        "best_fixed_cov": best_fixed_cov,
        "oracle_cov": oracle_cov,
        "abs_gap": abs_gap,
        "rel_gain_pct": rel_gain_pct,
    }


def _nearest_swept_size(available: List[int], target: int) -> int:
    """Return the swept size in ``available`` nearest to ``target`` (ties -> smaller)."""
    return min(available, key=lambda s: (abs(s - target), s))


def coverage_under_sizes(records: List[dict], size_by_doc: Dict[str, int]) -> float:
    """Mean coverage when each doc uses ``size_by_doc[doc_id]``, SNAPPED to the swept grid.

    An arbitrary policy may request a size that was never swept for a document
    (e.g. a density-predicted 173 tokens). We can only *measure* coverage at sizes
    that were actually run, so each requested size is snapped to the nearest size
    available for that document (ties broken toward the smaller size) and that swept
    coverage is used. This is the honest way to score a continuous-valued policy on
    a discrete sweep.

    Only documents present in BOTH ``size_by_doc`` and the coverage matrix
    contribute; the mean is over those documents. Returns ``0.0`` if no document
    qualifies.
    """
    matrix = coverage_matrix(records)
    covs: List[float] = []
    for doc_id, requested in size_by_doc.items():
        sizes = matrix.get(doc_id)
        if not sizes:
            continue
        snapped = _nearest_swept_size(list(sizes), requested)
        covs.append(sizes[snapped])
    if not covs:
        return 0.0
    return sum(covs) / len(covs)


# ---------------------------------------------------------------------------
# Calibration (pure; no models / network).
# ---------------------------------------------------------------------------
def _ranks(values: List[float]) -> List[float]:
    """Fractional (average) ranks of ``values``; ties share their mean rank."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Extend over a run of equal values.
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Average rank (1-based) for the tied group [i, j].
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation of two equal-length sequences (0.0 if either is constant)."""
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(vx * vy)
    if denom == 0.0:
        return 0.0
    return cov / denom


def rank_corr(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation: Pearson correlation of the fractional ranks.

    Implemented from scratch (ranks + Pearson) so the module stays scipy-free.
    Returns ``+1`` for a strictly monotone-increasing relationship, ``-1`` for
    strictly monotone-decreasing, and ``~0`` for no monotone association. Returns
    ``0.0`` for empty/length-1 inputs or when either side is constant (no ranking
    information). Raises ``ValueError`` if the inputs differ in length.
    """
    if len(xs) != len(ys):
        raise ValueError("rank_corr: xs and ys must have equal length")
    if len(xs) < 2:
        return 0.0
    return _pearson(_ranks(list(xs)), _ranks(list(ys)))


def fit_rho_to_size(
    rho_by_doc: Dict[str, float], opt_size_by_doc: Dict[str, int]
) -> dict:
    """Parsimonious ordinary least-squares fit of ``size = a + b * rho``.

    Two parameters only -- a deliberately under-parameterized, interpretable map
    from a single density scalar ``rho`` to a leaf-token budget. Fitted over the
    documents present in BOTH dicts. Returns::

        {"a": float, "b": float, "invert": bool,
         "rho_min": float, "rho_max": float, "n": int}

    ``invert`` is ``b < 0`` (denser text -> smaller leaves, the working
    hypothesis); ``rho_min`` / ``rho_max`` record the observed density range for
    reference. With fewer than two usable docs, or zero variance in ``rho``, the
    slope is ``0`` and the intercept is the mean target size (a sensible constant
    fallback). Use :func:`predict_size` to apply the fit.
    """
    keys = [k for k in rho_by_doc if k in opt_size_by_doc]
    xs = [float(rho_by_doc[k]) for k in keys]
    ys = [float(opt_size_by_doc[k]) for k in keys]
    n = len(keys)

    if n == 0:
        return {"a": 0.0, "b": 0.0, "invert": False,
                "rho_min": 0.0, "rho_max": 0.0, "n": 0}

    rho_min, rho_max = min(xs), max(xs)
    mean_y = sum(ys) / n

    mean_x = sum(xs) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if n < 2 or var_x == 0.0:
        # No slope is identifiable; fall back to the constant mean target size.
        return {"a": mean_y, "b": 0.0, "invert": False,
                "rho_min": rho_min, "rho_max": rho_max, "n": n}

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = cov_xy / var_x
    a = mean_y - b * mean_x
    return {"a": a, "b": b, "invert": b < 0,
            "rho_min": rho_min, "rho_max": rho_max, "n": n}


def predict_size(rho: float, fit: dict, l_min: int = 50, l_max: int = 400) -> int:
    """Predict a leaf-token size from density ``rho`` under a :func:`fit_rho_to_size` fit.

    Computes ``a + b * rho``, rounds to the nearest int, and clips to
    ``[l_min, l_max]`` so predictions never escape the swept range. ``l_min`` /
    ``l_max`` default to the QASPER sweep bounds (50, 400).
    """
    raw = fit.get("a", 0.0) + fit.get("b", 0.0) * float(rho)
    return int(max(l_min, min(l_max, round(raw))))


def feature_target_correlations(
    feat_by_doc: Dict[str, Dict[str, float]],
    target_by_doc: Dict[str, int],
    feature_names: List[str] | None = None,
) -> List[Tuple[str, float]]:
    """Spearman rank-corr of each density feature against the per-doc optimal size.

    ``feat_by_doc`` maps ``doc_id -> {feature_name: value}`` (the feature dict from
    :func:`raptor.chunking.density_score.density_score`); ``target_by_doc`` maps
    ``doc_id -> optimal_size``. Correlations are computed over the documents present
    in BOTH. Returns a list of ``(feature_name, signed_spearman)`` sorted by
    DESCENDING absolute correlation, so the most predictive feature (in either
    direction) is first.

    This is the diagnostic behind the key pilot finding: the equal-weight composite
    ``rho`` washes out because individual features correlate with optimal size in
    OPPOSITE directions. Selecting the single strongest feature (see
    :func:`select_and_fit`) recovers the signal the composite cancels. If
    ``feature_names`` is given, only those features are scored (and in that order
    before sorting); otherwise the union of keys across all feature dicts is used.
    """
    keys = [k for k in target_by_doc if k in feat_by_doc]
    if feature_names is None:
        names: List[str] = []
        for k in keys:
            for f in feat_by_doc[k]:
                if f not in names:
                    names.append(f)
    else:
        names = list(feature_names)

    targets = [float(target_by_doc[k]) for k in keys]
    out: List[Tuple[str, float]] = []
    for f in names:
        xs = [float(feat_by_doc[k].get(f, 0.0)) for k in keys]
        out.append((f, rank_corr(xs, targets)))
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return out


def select_and_fit(
    feat_by_doc: Dict[str, Dict[str, float]],
    target_by_doc: Dict[str, int],
    train_ids: List[str],
    feature_names: List[str] | None = None,
) -> dict:
    """Pick the most predictive single feature on TRAIN, then fit ``size = a + b*feature``.

    This is the parsimonious, training-free calibration arm. Rather than feeding the
    dead equal-weight composite ``rho`` into :func:`fit_rho_to_size`, it ranks the
    density features by |Spearman| against the optimal size **on the training docs
    only** (no test leakage), selects the top one, and fits the same two-parameter
    OLS line on that feature. Returns::

        {"feature": str, "corr": float, "fit": <fit_rho_to_size dict>}

    Apply it with ``predict_size(feat_by_doc[doc][result["feature"]], result["fit"])``.
    With no usable training docs the feature is ``None`` and ``fit`` is the empty/
    constant fallback from :func:`fit_rho_to_size`. Selecting among features on a tiny
    sample is itself a (mild) source of optimism — report it on a held-out TEST split
    and treat single-feature wins on small N as directional, not conclusive.
    """
    train = {k: feat_by_doc[k] for k in train_ids if k in feat_by_doc and k in target_by_doc}
    train_targets = {k: target_by_doc[k] for k in train}
    corrs = feature_target_correlations(train, train_targets, feature_names)
    if not corrs:
        return {"feature": None, "corr": 0.0, "fit": fit_rho_to_size({}, {})}
    feature, corr = corrs[0]
    feat_values = {k: float(train[k].get(feature, 0.0)) for k in train}
    fit = fit_rho_to_size(feat_values, train_targets)
    return {"feature": feature, "corr": corr, "fit": fit}


def train_test_split_docs(
    doc_ids: List[str], frac_train: float = 0.7, seed: int = 0
) -> Tuple[List[str], List[str]]:
    """Deterministically split ``doc_ids`` into ``(train_ids, test_ids)``.

    The ids are first sorted (so the result is independent of input order), then
    shuffled with a dedicated ``random.Random(seed)`` (never the global RNG) so the
    split is reproducible for a given ``(doc_ids, frac_train, seed)``. The first
    ``round(frac_train * n)`` go to train, the remainder to test. The split is
    always disjoint and jointly covers every id. ``frac_train`` is clamped to
    ``[0, 1]``.
    """
    frac_train = max(0.0, min(1.0, frac_train))
    ids = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = int(round(frac_train * len(ids)))
    return ids[:n_train], ids[n_train:]
