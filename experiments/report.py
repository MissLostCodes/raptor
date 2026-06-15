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
    "evidence_coverage",  # QASPER gold-evidence token-coverage proxy
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


def exclude_blocked_uniformly(records: List[dict]) -> List[dict]:
    """Drop a (dataset, doc_id, question_id) from ALL arms if blocked anywhere.

    Content-moderation blocks (and other errors) happen on the per-arm
    retrieved context, so the same question can be blocked under one arm and
    answered under another. Scoring only the arms that happened to answer it
    would make the arm-vs-arm comparison unfair. So if a question is
    blocked/errored under ANY arm, it is removed from EVERY arm at report time.

    Records lacking dataset/doc_id/question_id/blocked/error keys (e.g. the
    simpler fixtures in older tests) are never considered "blocked" and are
    passed through unchanged.
    """
    blocked_keys = {
        (r.get("dataset"), r.get("doc_id"), r.get("question_id"))
        for r in records
        if r.get("blocked") or r.get("error")
    }
    return [
        r
        for r in records
        if (r.get("dataset"), r.get("doc_id"), r.get("question_id")) not in blocked_keys
    ]


def blocked_report(records: List[dict]) -> dict:
    """Per-dataset count of distinct questions dropped due to a block/error.

    Returns ``{dataset: {"dropped_questions": int, "errors": {type: count}}}``
    including only datasets that actually had at least one block. An all-clean
    run yields ``{}``.
    """
    dropped: Dict[str, set] = defaultdict(set)
    errors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        if r.get("blocked") or r.get("error"):
            ds = r.get("dataset")
            dropped[ds].add((r.get("doc_id"), r.get("question_id")))
            err = r.get("error") or ("blocked" if r.get("blocked") else "unknown")
            errors[ds][str(err)] += 1

    return {
        ds: {
            "dropped_questions": len(pairs),
            "errors": dict(errors[ds]),
        }
        for ds, pairs in dropped.items()
    }


def aggregate(records: List[dict], seed: int = 0, drop_blocked: bool = True) -> dict:
    """Group records by (arm, dataset, metric) -> summary stats.

    Numeric generative/F1 metrics are averaged directly. Quality 'correct'
    becomes both 'accuracy' (overall) and 'accuracy_hard' (is_hard subset).

    When ``drop_blocked`` is True (the default) any question blocked/errored
    under any arm is removed from all arms first, so every arm is scored on the
    exact same question set.

    Returns a nested dict: ``{arm: {dataset: {metric: summary}}}``.
    """
    if drop_blocked:
        records = exclude_blocked_uniformly(records)

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


def paired_diffs(
    records: List[dict], arm_a: str, arm_b: str, dataset: str, metric: str
) -> List[float]:
    """Per-question ``arm_a - arm_b`` metric differences over their COMMON
    questions (paired by ``(doc_id, question_id)``).

    Questions only one arm scored (e.g. blocked under the other) are dropped so
    the comparison is genuinely paired. Ordered by ``(doc_id, question_id)`` for
    determinism.
    """

    def by_question(arm: str) -> Dict[Tuple, float]:
        out: Dict[Tuple, float] = {}
        for r in records:
            if r.get("arm") == arm and r.get("dataset") == dataset and r.get(metric) is not None:
                out[(r.get("doc_id"), r.get("question_id"))] = float(r[metric])
        return out

    a = by_question(arm_a)
    b = by_question(arm_b)
    common = sorted(set(a) & set(b))
    return [a[k] - b[k] for k in common]


def paired_bootstrap(
    diffs: List[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Two-sided paired bootstrap on per-question score differences.

    Resamples the paired differences with replacement to get a percentile CI of
    the mean difference, and a centered-bootstrap two-sided p-value for
    H0: mean difference = 0 (the bootstrap distribution shifted to mean 0 is the
    null; p is the share of null means at least as extreme as the observed,
    with add-one smoothing so p is never exactly 0). Deterministic for a seed.

    Returns ``{mean_diff, ci_low, ci_high, p_value, n, significant}``;
    ``significant`` is True iff the CI excludes 0.
    """
    clean = [float(d) for d in diffs if d is not None]
    n = len(clean)
    if n == 0:
        nan = float("nan")
        return {"mean_diff": nan, "ci_low": nan, "ci_high": nan,
                "p_value": nan, "n": 0, "significant": False}

    mean_diff = sum(clean) / n
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += clean[rng.randrange(n)]
        boot_means.append(total / n)
    boot_means.sort()

    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = max(lo_idx, min(int((1 - alpha / 2) * n_boot) - 1, n_boot - 1))
    ci_low, ci_high = boot_means[lo_idx], boot_means[hi_idx]

    # Centered-bootstrap two-sided p-value (H0: mean diff = 0).
    obs = abs(mean_diff)
    extreme = sum(1 for b in boot_means if abs(b - mean_diff) >= obs)
    p_value = min(1.0, (extreme + 1) / (n_boot + 1))

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n": n,
        "significant": not (ci_low <= 0.0 <= ci_high),
    }


def paired_arm_test(
    records: List[dict],
    arm_a: str,
    arm_b: str,
    dataset: str,
    metric: str,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Convenience: paired bootstrap of ``arm_a`` vs ``arm_b`` on ``metric``."""
    diffs = paired_diffs(records, arm_a, arm_b, dataset, metric)
    result = paired_bootstrap(diffs, n_boot=n_boot, seed=seed, alpha=alpha)
    result["arm_a"] = arm_a
    result["arm_b"] = arm_b
    result["dataset"] = dataset
    result["metric"] = metric
    return result


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


def cost_report(records: List[dict]) -> dict:
    """Per-(dataset, arm) compute accounting for the RQ2/H2 cost comparison.

    Aggregates the runner's per-record timings into:

    - ``n_docs`` / ``n_questions``
    - ``build_s`` : total answerer-build time (counted ONCE per doc, since a
      doc's questions share one build)
    - ``answer_s`` : total answer latency over all questions
    - ``wall_s`` : ``build_s + answer_s``
    - ``parse_docs`` : docs that triggered an LLM structure parse -- every doc
      for the ``structure`` arm, and only structure-routed docs for ``ahc``
      (``token``/``semantic``/``flat`` never parse). This is the headline
      "AHC parses fewer documents than always-Structure" number.

    Returns ``{dataset: {arm: {...}}}``.
    """
    agg: Dict[Tuple[str, str], dict] = defaultdict(
        lambda: {
            "docs": set(),
            "n_questions": 0,
            "build_s": 0.0,
            "answer_s": 0.0,
            "parse_docs": set(),
        }
    )
    counted_build: set = set()
    for r in records:
        ds, arm, doc = r.get("dataset"), r.get("arm"), r.get("doc_id")
        cell = agg[(ds, arm)]
        cell["docs"].add(doc)
        cell["n_questions"] += 1
        cell["answer_s"] += float(r.get("answer_s") or 0.0)
        if (ds, arm, doc) not in counted_build:
            counted_build.add((ds, arm, doc))
            cell["build_s"] += float(r.get("build_s") or 0.0)
            route = (r.get("routing") or {}).get("route")
            if arm == "structure" or (arm == "ahc" and route == "structure"):
                cell["parse_docs"].add(doc)

    out: dict = defaultdict(dict)
    for (ds, arm), cell in agg.items():
        out[ds][arm] = {
            "n_docs": len(cell["docs"]),
            "n_questions": cell["n_questions"],
            "build_s": cell["build_s"],
            "answer_s": cell["answer_s"],
            "wall_s": cell["build_s"] + cell["answer_s"],
            "parse_docs": len(cell["parse_docs"]),
        }
    return {ds: dict(arms) for ds, arms in out.items()}


def to_markdown(
    agg: dict, routing: Optional[dict] = None, dropped: Optional[dict] = None
) -> str:
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

    if dropped:
        lines.append("\n## Dropped (moderation-blocked) questions\n")
        lines.append("| dataset | dropped_questions |")
        lines.append("| --- | ---: |")
        for ds in sorted(dropped.keys()):
            lines.append(
                "| {ds} | {n} |".format(ds=ds, n=dropped[ds]["dropped_questions"])
            )

    return "\n".join(lines) + "\n"
