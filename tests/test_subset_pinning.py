"""Offline tests for pinned-subset reading (reproducibility plumbing).

`select_subset` (writer side) is already tested; these cover the reader side
that the runner uses to load a version-controlled, pinned evaluation subset.
"""

import json

from experiments.datasets.base import load_subset_ids, subset_ids_path


def test_subset_ids_path_uses_name(tmp_path):
    p = subset_ids_path("qasper", subsets_dir=str(tmp_path))
    assert p.endswith("qasper_ids.json")
    assert str(tmp_path) in p


def test_missing_pin_file_returns_empty(tmp_path):
    assert load_subset_ids("qasper", subsets_dir=str(tmp_path)) == []


def test_empty_pin_returns_empty(tmp_path):
    (tmp_path / "qasper_ids.json").write_text(json.dumps({"seed": 0, "ids": []}))
    assert load_subset_ids("qasper", subsets_dir=str(tmp_path)) == []


def test_reads_pinned_ids_in_order(tmp_path):
    (tmp_path / "quality_ids.json").write_text(
        json.dumps({"seed": 0, "ids": ["d3", "d1", "d2"]})
    )
    assert load_subset_ids("quality", subsets_dir=str(tmp_path)) == ["d3", "d1", "d2"]


def test_malformed_pin_returns_empty(tmp_path):
    (tmp_path / "narrativeqa_ids.json").write_text("not json at all")
    # A corrupt pin must not crash a run; treated as "unpinned".
    assert load_subset_ids("narrativeqa", subsets_dir=str(tmp_path)) == []
