"""CLI entry point for the chunking-arms experiment harness.

Example::

    python -m experiments.run \
        --arms token,structure,ahc,semantic,flat \
        --datasets qasper,quality,narrativeqa \
        --model openai/gpt-oss-120b:free \
        --cache-dir .llm_cache --results-dir results --seed 0

``main(argv)`` builds an ExperimentConfig and (unless ``--dry-run``) runs the
grid and prints the Markdown report. ``--dry-run`` makes ``main`` return the
config without any network/run, which is how the CLI is unit-tested.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from experiments.config import ExperimentConfig


def _csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="experiments.run", description=__doc__)
    p.add_argument("--arms", type=str, default=None,
                   help="Comma-separated arms (token,structure,ahc,semantic,flat)")
    p.add_argument("--datasets", type=str, default=None,
                   help="Comma-separated datasets (qasper,quality,narrativeqa)")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--cache-dir", dest="cache_dir", type=str, default=None)
    p.add_argument("--results-dir", dest="results_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-qasper", dest="n_qasper", type=int, default=None)
    p.add_argument("--n-quality", dest="n_quality", type=int, default=None)
    p.add_argument("--n-narrativeqa", dest="n_narrativeqa", type=int, default=None)
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Build the config and return without running.")
    p.add_argument("--subsets-dir", dest="subsets_dir", type=str, default=None,
                   help="Directory of pinned-subset JSONs (default: "
                        "experiments/datasets/subsets).")
    p.add_argument("--allow-unpinned", dest="allow_unpinned", action="store_true",
                   help="Allow running a dataset with no pinned subset (uses the "
                        "first --n-<dataset> docs by load order; NOT reproducible). "
                        "Intended for pilot smoke runs only.")
    return p


def config_from_args(args) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if args.arms:
        cfg.arms = _csv(args.arms)
    if args.datasets:
        cfg.datasets = _csv(args.datasets)
    if args.model:
        cfg.model = args.model
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    if args.results_dir:
        cfg.results_dir = args.results_dir

    overrides = {
        "qasper": args.n_qasper,
        "quality": args.n_quality,
        "narrativeqa": args.n_narrativeqa,
    }
    sizes = dict(cfg.subset_sizes)
    for name, val in overrides.items():
        if val is not None:
            sizes[name] = val
    cfg.subset_sizes = sizes
    return cfg


def main(argv: Optional[List[str]] = None):
    """Parse args -> ExperimentConfig. Runs unless ``--dry-run``.

    Returns the ExperimentConfig (always), so the CLI is testable without
    executing the grid.
    """
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    if args.dry_run:
        import json

        print("[dry-run] resolved ExperimentConfig:")
        print(json.dumps(cfg.to_dict(), indent=2))
        return cfg

    # Heavy path: only imported when actually running.
    from experiments import report, runner

    # Pre-stage METEOR data up front so a NarrativeQA run reports it (or warns
    # loudly) rather than silently emitting meteor=None per question.
    if "narrativeqa" in cfg.datasets:
        from experiments.metrics import stage_meteor_data

        if stage_meteor_data():
            print("[meteor] NLTK data staged; METEOR enabled for narrativeqa.")
        else:
            print(
                "[meteor] WARNING: NLTK METEOR data unavailable offline; "
                "narrativeqa METEOR will be reported as None."
            )

    results = runner.run(
        cfg,
        seed=args.seed,
        subsets_dir=args.subsets_dir,
        allow_unpinned=args.allow_unpinned,
    )
    agg = report.aggregate(results["records"])
    routing = report.routing_rate(results["records"])
    print(report.to_markdown(agg, routing))
    return cfg


if __name__ == "__main__":
    main()
