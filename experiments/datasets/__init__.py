"""Dataset loaders for the RAPTOR reproduction harness.

Exposes a common ``Document`` / ``QAExample`` representation, a deterministic
``select_subset`` sampler, and loaders for QASPER, QuALITY, and NarrativeQA.
The actual HuggingFace download lives in each loader's ``load`` method; the
pure ``parse_*`` / ``group_*`` helpers are unit-tested offline.
"""

from experiments.datasets.base import (
    QAExample,
    Document,
    DatasetLoader,
    select_subset,
)
from experiments.datasets.qasper import QASPERLoader, parse_qasper_row
from experiments.datasets.quality import QuALITYLoader, parse_quality_article
from experiments.datasets.narrativeqa import (
    NarrativeQALoader,
    group_narrativeqa_rows,
)

_LOADERS = {
    "qasper": QASPERLoader,
    "quality": QuALITYLoader,
    "narrativeqa": NarrativeQALoader,
}


def get_loader(name: str, **kwargs) -> DatasetLoader:
    """Factory mapping a dataset name to its loader instance."""
    try:
        cls = _LOADERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}; expected one of {sorted(_LOADERS)}"
        )
    return cls(**kwargs)


__all__ = [
    "QAExample",
    "Document",
    "DatasetLoader",
    "select_subset",
    "QASPERLoader",
    "QuALITYLoader",
    "NarrativeQALoader",
    "parse_qasper_row",
    "parse_quality_article",
    "group_narrativeqa_rows",
    "get_loader",
]
