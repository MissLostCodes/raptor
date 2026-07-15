"""run_sweep tree-vs-flat mode: the resume cache must never blend the two.

The resume key is (doc_id, size) and carries no retrieval mode, so a tree sweep
pointed at a flat cache would skip every document and return flat coverage
labelled as tree results — a wrong number that looks exactly like a right one,
and one that would land straight in the paper.
"""

import json

import pytest

from experiments.granularity_sweep import run_sweep


def _write(tmp_path, name, records):
    p = tmp_path / name
    p.write_text(json.dumps(records))
    return str(p)


FLAT = [{"doc_id": "d1", "size": 50, "n_questions": 1, "mean_evidence_coverage": 0.5,
         "n_layers": None}]
LEGACY = [{"doc_id": "d1", "size": 50, "n_questions": 1, "mean_evidence_coverage": 0.5}]
TREE = [{"doc_id": "d1", "size": 50, "n_questions": 1, "mean_evidence_coverage": 0.9,
         "n_layers": 2}]
# A degenerate tree (never clustered) is still a TREE record: 0 is not None.
TREE_ZERO = [{"doc_id": "d1", "size": 400, "n_questions": 1,
              "mean_evidence_coverage": 0.9, "n_layers": 0}]


def test_tree_sweep_refuses_a_flat_cache(tmp_path):
    out = _write(tmp_path, "flat.json", FLAT)
    with pytest.raises(ValueError, match="silently mix"):
        run_sweep([], [50], out_path=out, summarization_model=object())


def test_tree_sweep_refuses_a_legacy_cache(tmp_path):
    """Pre-n_layers records look flat, because they are."""
    out = _write(tmp_path, "legacy.json", LEGACY)
    with pytest.raises(ValueError, match="silently mix"):
        run_sweep([], [50], out_path=out, summarization_model=object())


def test_flat_sweep_refuses_a_tree_cache(tmp_path):
    out = _write(tmp_path, "tree.json", TREE)
    with pytest.raises(ValueError, match="silently mix"):
        run_sweep([], [50], out_path=out)


def test_zero_layer_records_still_count_as_tree(tmp_path):
    """n_layers=0 means 'built no tree', NOT 'flat mode' — the guard must not confuse them."""
    out = _write(tmp_path, "tree0.json", TREE_ZERO)
    with pytest.raises(ValueError, match="silently mix"):
        run_sweep([], [400], out_path=out)


def test_matching_modes_resume_fine(tmp_path):
    flat_out = _write(tmp_path, "flat_ok.json", FLAT)
    assert run_sweep([], [50], out_path=flat_out) == FLAT

    tree_out = _write(tmp_path, "tree_ok.json", TREE)
    assert run_sweep([], [50], out_path=tree_out, summarization_model=object()) == TREE


def test_empty_cache_accepts_either_mode(tmp_path):
    out = _write(tmp_path, "empty.json", [])
    assert run_sweep([], [50], out_path=out) == []
    assert run_sweep([], [50], out_path=out, summarization_model=object()) == []
