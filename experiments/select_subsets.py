"""Pin reproducible per-dataset evaluation subsets (runs on Colab; NOT unit-tested).

For each configured dataset, loads all documents, deterministically selects
``cfg.subset_sizes[name]`` doc_ids stratified by document length, and writes
``experiments/datasets/subsets/<name>_ids.json`` as
``{"seed": seed, "ids": [...]}``.
"""

from __future__ import annotations

import json
import os

from experiments.config import ExperimentConfig
from experiments.datasets import get_loader
from experiments.datasets.base import select_subset

SUBSETS_DIR = os.path.join(os.path.dirname(__file__), "datasets", "subsets")


def _length_strata(doc) -> str:
    """Bucket documents into short/medium/long by token-ish word count."""
    n = len(doc.text.split())
    if n < 2000:
        return "short"
    if n < 8000:
        return "medium"
    return "long"


def main(cfg: ExperimentConfig = None, seed: int = 0) -> None:
    cfg = cfg or ExperimentConfig()
    os.makedirs(SUBSETS_DIR, exist_ok=True)

    for name in cfg.datasets:
        loader = get_loader(name)
        documents = loader.load()
        ids = select_subset(
            documents,
            cfg.subset_sizes.get(name, len(documents)),
            seed=seed,
            strata_fn=_length_strata,
        )
        out_path = os.path.join(SUBSETS_DIR, f"{name}_ids.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"seed": seed, "ids": ids}, fh, indent=2)
        print(f"[select_subsets] {name}: {len(ids)} ids -> {out_path}")


if __name__ == "__main__":
    main()
