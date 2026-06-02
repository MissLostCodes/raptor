"""Pure, offline result aggregation + reporting for the chunking-arms harness.

No I/O, no model, no network: takes the records list produced by
``runner.run`` and computes per-(arm, dataset, metric) summary stats with
bootstrap CIs, plus the AHC structure-routing rate, and renders a Markdown
table.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Metric keys we know how to aggregate per dataset.
_NUMERIC_METRICS = (
    "answer_f1",
    "rouge_l",
    "bleu_1",
    "bleu_4",
    "meteor",
)


def bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean. Deterministic for a given seed.

    Returns ``(low, high)``. For an empty input returns ``(nan, nan)``; for a
    single value returns ``(v, v)``.
    """
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], clean[0]

    rng = random.Random(seed)
    n = len(clean)
    means = []
    for _ in range(n_boot):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    hi_idx = max(lo_idx, min(hi_idx, n_boot - 1))
    return means[lo_idx], means[hi_idx]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _summary(values: List[float], seed: int = 0) -> dict:
    clean = [float(v) for v in values if v is not None]
    lo, hi = bootstrap_ci(clean, seed=seed)
    return {
        "mean": _mean(clean),
        "std": _std(clean),
        "n": len(clean),
        "ci_low": lo,
        "ci_high": hi,
    }


def aggregate(records: List[dict], seed: int = 0) -> dict:
    """Group records by (arm, dataset, metric) -> summary stats.

    Numeric generative/F1 metrics are averaged directly. Quality 'correct'
    becomes both 'accuracy' (overall) and 'accuracy_hard' (is_hard subset).

    Returns a nested dict: ``{arm: {dataset: {metric: summary}}}``.
    """
    # Collect raw value lists.
    buckets: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in records:
        arm = r.get("arm")
        ds = r.get("dataset")
        for m in _NUMERIC_METRICS:
            if m in r and r[m] is not None:
                buckets[(arm, ds, m)].append(float(r[m]))
        # Quality: accuracy overall + hard subset.
        if "correct" in r and r["correct"] is not None:
            buckets[(arm, ds, "accuracy")].append(float(r["correct"]))
            if r.get("is_hard"):
                buckets[(arm, ds, "accuracy_hard")].append(float(r["correct"]))

    out: dict = defaultdict(lambda: defaultdict(dict))
    for (arm, ds, metric), values in buckets.items():
        out[arm][ds][metric] = _summary(values, seed=seed)

    # Convert to plain nested dicts for clean JSON/printing.
    return {arm: {ds: dict(metrics) for ds, metrics in dss.items()} for arm, dss in out.items()}


def routing_rate(records: List[dict]) -> dict:
    """For arm 'ahc', fraction of docs routed to 'structure' per dataset.

    Routing is per (dataset, doc_id); a doc is counted once. Returns
    ``{dataset: {"structure_rate": float, "n_docs": int}}``.
    """
    # Per (dataset, doc_id) -> route (last seen for that doc under ahc).
    doc_routes: Dict[Tuple[str, str], Optional[str]] = {}
    for r in records:
        if r.get("arm") != "ahc":
            continue
        routing = r.get("routing") or {}
        route = routing.get("route")
        if route is None:
            continue
        doc_routes[(r.get("dataset"), r.get("doc_id"))] = route

    per_ds_total: Dict[str, int] = defaultdict(int)
    per_ds_struct: Dict[str, int] = defaultdict(int)
    for (ds, _doc), route in doc_routes.items():
        per_ds_total[ds] += 1
        if route == "structure":
            per_ds_struct[ds] += 1

    return {
        ds: {
            "structure_rate": per_ds_struct[ds] / per_ds_total[ds]
            if per_ds_total[ds]
            else float("nan"),
            "n_docs": per_ds_total[ds],
        }
        for ds in per_ds_total
    }


def to_markdown(agg: dict, routing: Optional[dict] = None) -> str:
    """Render the aggregate (and optional routing rates) as Markdown."""
    lines: List[str] = []
    lines.append("# Chunking-arms results\n")
    lines.append("| arm | dataset | metric | mean | std | n | 95% CI |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")

    for arm in sorted(agg.keys()):
        for ds in sorted(agg[arm].keys()):
            for metric in sorted(agg[arm][ds].keys()):
                s = agg[arm][ds][metric]
                lines.append(
                    "| {arm} | {ds} | {metric} | {mean:.4f} | {std:.4f} | {n} | "
                    "[{lo:.4f}, {hi:.4f}] |".format(
                        arm=arm,
                        ds=ds,
                        metric=metric,
                        mean=s["mean"],
                        std=s["std"],
                        n=s["n"],
                        lo=s["ci_low"],
                        hi=s["ci_high"],
                    )
                )

    if routing:
        lines.append("\n## AHC structure-routing rate\n")
        lines.append("| dataset | structure_rate | n_docs |")
        lines.append("| --- | ---: | ---: |")
        for ds in sorted(routing.keys()):
            r = routing[ds]
            lines.append(
                "| {ds} | {rate:.4f} | {n} |".format(
                    ds=ds, rate=r["structure_rate"], n=r["n_docs"]
                )
            )

    return "\n".join(lines) + "\n"
