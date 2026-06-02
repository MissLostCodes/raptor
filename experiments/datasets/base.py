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

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


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
