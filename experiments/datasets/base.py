"""Core dataclasses and the loader base for the RAPTOR reproduction harness.

Provides a dataset-agnostic representation:

- ``QAExample``  : a single question with its gold answers / options / evidence.
- ``Document``   : a full source document (paper, article, or story) holding
                   all of its questions.
- ``DatasetLoader`` : abstract base; each dataset implements ``load``.
- ``select_subset`` : deterministic, optionally stratified, doc-id sampling
                      used to pin version-controlled evaluation subsets.
"""

from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

# Version-controlled pinned evaluation subsets live next to this module, in
# ``experiments/datasets/subsets/<name>_ids.json``. ``select_subset`` (below)
# selects the ids; ``select_subsets.py`` writes them; ``load_subset_ids`` reads
# them back at run time so every run uses the same documents.
SUBSETS_DIR = os.path.join(os.path.dirname(__file__), "subsets")


def subset_ids_path(name: str, subsets_dir: Optional[str] = None) -> str:
    """Path to the pinned-ids JSON for dataset ``name``."""
    return os.path.join(subsets_dir or SUBSETS_DIR, f"{name}_ids.json")


def load_subset_ids(name: str, subsets_dir: Optional[str] = None) -> List[str]:
    """Return the pinned doc_ids for ``name``, or ``[]`` if unpinned.

    "Unpinned" means the pin file is missing, malformed, or has an empty ``ids``
    list. A corrupt pin is treated as unpinned (never crashes a run); the runner
    decides whether an empty pin is a hard error.
    """
    path = subset_ids_path(name, subsets_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    return list(data.get("ids") or [])


@dataclass
class QAExample:
    question_id: str
    question: str
    gold_answers: List[str] = field(default_factory=list)
    options: Optional[List[str]] = None
    gold_index: Optional[int] = None
    evidence: List[str] = field(default_factory=list)
    is_unanswerable: bool = False
    is_hard: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    questions: List[QAExample] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class DatasetLoader(ABC):
    """Abstract base for dataset loaders.

    Subclasses set ``name`` and implement ``load``. The actual HuggingFace
    download lives entirely inside ``load`` so that the pure parsing helpers
    can be unit-tested offline.
    """

    name: str = ""

    @abstractmethod
    def load(
        self,
        subset_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Document]:
        raise NotImplementedError


def select_subset(
    documents: List[Document],
    n: int,
    seed: int = 0,
    strata_fn: Callable[[Document], Any] = None,
) -> List[str]:
    """Deterministically select up to ``n`` doc_ids from ``documents``.

    Reproducible for a given ``(documents, n, seed)``. Uses a dedicated
    ``random.Random(seed)`` instance (never the global RNG).

    If ``strata_fn`` is provided, samples proportionally across strata
    (largest-remainder allocation); otherwise samples uniformly.

    Returns the selected doc_ids in the order they were selected.
    """
    rng = random.Random(seed)
    n = min(n, len(documents))
    if n <= 0:
        return []

    if strata_fn is None:
        chosen = rng.sample(list(documents), n)
        return [d.doc_id for d in chosen]

    # Stratified: group, allocate proportionally, sample within each stratum.
    groups: dict = defaultdict(list)
    for doc in documents:
        groups[strata_fn(doc)].append(doc)

    # Deterministic, stable ordering of strata keys.
    keys = sorted(groups.keys(), key=lambda k: (str(type(k)), str(k)))
    total = len(documents)

    # Largest-remainder apportionment of n across strata by size.
    quotas = {}
    remainders = []
    allocated = 0
    for k in keys:
        exact = n * len(groups[k]) / total
        base = int(exact)
        # Cap by available docs in stratum.
        base = min(base, len(groups[k]))
        quotas[k] = base
        allocated += base
        remainders.append((exact - int(exact), k))

    # Distribute the leftover slots to the largest remainders (with capacity).
    leftover = n - allocated
    remainders.sort(key=lambda x: (-x[0], str(x[1])))
    i = 0
    while leftover > 0 and remainders:
        _, k = remainders[i % len(remainders)]
        if quotas[k] < len(groups[k]):
            quotas[k] += 1
            leftover -= 1
        i += 1
        # Safety: if no stratum has capacity, stop.
        if i > len(remainders) * (max(len(g) for g in groups.values()) + 1):
            break

    selected: List[str] = []
    for k in keys:
        take = quotas[k]
        if take <= 0:
            continue
        chosen = rng.sample(groups[k], take)
        selected.extend(d.doc_id for d in chosen)

    return selected


# ---------------------------------------------------------------------------
# HuggingFace loading (script-free; works on datasets >= 3.x / 4.x)
# ---------------------------------------------------------------------------
# datasets >= 3.0 dropped dataset *loading scripts* ("Dataset scripts are no
# longer supported"), which breaks former script datasets like allenai/qasper
# and deepmind/narrativeqa. The modern, version-proof path is to read the
# auto-converted Parquet that the Hub generates for every dataset on the
# ``refs/convert/parquet`` branch. We try a normal load first (covers
# Parquet/JSON-native repos) and fall back to that Parquet branch.

PARQUET_REVISION = "refs/convert/parquet"


def _parquet_data_files(repo: str, files: List[str], split: str) -> List[str]:
    """Build ``hf://`` Parquet URLs for ``split`` from a repo's converted-Parquet
    file listing.

    The conversion lays files out as ``<config>/<split>/<shard>.parquet`` (the
    config dir may be absent). Pure function so the path logic is unit-tested
    without any network.
    """
    out: List[str] = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        if f"/{split}/" in f or f.startswith(f"{split}/"):
            out.append(f"hf://datasets/{repo}@{PARQUET_REVISION}/{f}")
    return out


def load_hf_split(repo: str, split: str, config: Optional[str] = None):
    """Load one split of a Hub dataset, robust to the removal of loading scripts.

    1. Try ``load_dataset(repo, name=config, split=split)`` (Parquet/JSON-native).
    2. On failure, load the auto-converted Parquet from ``refs/convert/parquet``.
    """
    from datasets import load_dataset

    try:
        return load_dataset(repo, name=config, split=split)
    except Exception:
        pass  # former script dataset -> use the converted Parquet branch

    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        repo, repo_type="dataset", revision=PARQUET_REVISION
    )
    data_files = _parquet_data_files(repo, files, split)
    if not data_files:
        raise RuntimeError(
            f"Could not load {repo!r} split={split!r}: no loading script support "
            f"in datasets>=3.0 and no Parquet found on {PARQUET_REVISION}."
        )
    # The parquet builder places all rows in a single 'train' split.
    return load_dataset("parquet", data_files=data_files, split="train")
