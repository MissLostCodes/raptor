"""Pool per-seed results into mean +/- std *across seeds*.

The study reports each metric as the mean +/- std over >=3 seeds -- the headline
seed-variance number. That is distinct from :func:`experiments.report.aggregate`,
which gives a within-seed bootstrap CI over per-question scores. This module
combines the per-seed aggregates into the across-seed statistic and can read the
``results_<seed>.json`` files the runner writes.

Pure functions (``pool_seed_aggregates`` / ``pool_results``) are unit-tested;
``load_per_seed_records`` / ``main`` are thin IO wrappers.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Dict, List

from experiments.report import _mean, _std, aggregate


def pool_seed_aggregates(per_seed_agg: Dict[int, dict]) -> dict:
    """Combine ``{seed: aggregate_output}`` into across-seed mean/std per metric.

    For each ``(arm, dataset, metric)`` collects that metric's per-seed mean
    (skipping seeds where it is absent or NaN) and reports the mean and sample
    std of those per-seed means. Returns
    ``{arm: {dataset: {metric: {mean, std, n_seeds, seed_means, seeds}}}}``.
    """
    collected: Dict[tuple, List[tuple]] = defaultdict(list)
    for seed in sorted(per_seed_agg.keys()):
        agg = per_seed_agg[seed]
        for arm, datasets in agg.items():
            for ds, metrics in datasets.items():
                for metric, summary in metrics.items():
                    m = summary.get("mean")
                    if m is None or (isinstance(m, float) and math.isnan(m)):
                        continue
                    collected[(arm, ds, metric)].append((seed, float(m)))

    out: dict = defaultdict(lambda: defaultdict(dict))
    for (arm, ds, metric), seed_means in collected.items():
        seeds = [s for s, _v in seed_means]
        vals = [v for _s, v in seed_means]
        out[arm][ds][metric] = {
            "mean": _mean(vals),
            "std": _std(vals),
            "n_seeds": len(vals),
            "seed_means": vals,
            "seeds": seeds,
        }
    return {
        arm: {ds: dict(metrics) for ds, metrics in datasets.items()}
        for arm, datasets in out.items()
    }


def pool_results(
    per_seed_records: Dict[int, List[dict]], drop_blocked: bool = True
) -> dict:
    """Aggregate each seed's raw records, then pool across seeds.

    Each seed is aggregated with its own seed value so the within-seed bootstrap
    is reproducible; only the per-seed means feed the across-seed statistic.
    """
    per_seed_agg = {
        seed: aggregate(records, seed=seed, drop_blocked=drop_blocked)
        for seed, records in per_seed_records.items()
    }
    return pool_seed_aggregates(per_seed_agg)


def load_per_seed_records(results_dir: str, seeds: List[int]) -> Dict[int, List[dict]]:
    """Read ``results_<seed>.json`` for each seed that exists on disk."""
    out: Dict[int, List[dict]] = {}
    for seed in seeds:
        path = os.path.join(results_dir, f"results_{seed}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            out[seed] = json.load(fh).get("records", [])
    return out


def main(cfg=None) -> dict:
    """Load every configured seed's results, pool them, and print as JSON."""
    from experiments.config import ExperimentConfig

    cfg = cfg or ExperimentConfig()
    per_seed = load_per_seed_records(cfg.results_dir, cfg.seeds)
    pooled = pool_seed_aggregates(
        {seed: aggregate(recs, seed=seed) for seed, recs in per_seed.items()}
    )
    print(json.dumps(pooled, indent=2, sort_keys=True))
    return pooled


if __name__ == "__main__":
    main()
