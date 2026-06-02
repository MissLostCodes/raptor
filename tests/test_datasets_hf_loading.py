"""Offline tests for the script-free HF Parquet loading path (datasets >= 3.x)."""
from experiments.datasets.base import _parquet_data_files, PARQUET_REVISION


def test_parquet_data_files_with_config_dir():
    files = [
        "default/test/0000.parquet",
        "default/test/0001.parquet",
        "default/train/0000.parquet",
        "default/validation/0000.parquet",
        ".gitattributes",
        "README.md",
    ]
    out = _parquet_data_files("allenai/qasper", files, "test")
    assert out == [
        f"hf://datasets/allenai/qasper@{PARQUET_REVISION}/default/test/0000.parquet",
        f"hf://datasets/allenai/qasper@{PARQUET_REVISION}/default/test/0001.parquet",
    ]


def test_parquet_data_files_without_config_dir():
    files = ["test/0000.parquet", "train/0000.parquet"]
    out = _parquet_data_files("deepmind/narrativeqa", files, "test")
    assert out == [
        f"hf://datasets/deepmind/narrativeqa@{PARQUET_REVISION}/test/0000.parquet"
    ]


def test_parquet_data_files_split_isolation_and_non_parquet_skipped():
    files = ["default/validation/0000.parquet", "default/test/0000.parquet", "x.json"]
    out = _parquet_data_files("r/r", files, "validation")
    assert len(out) == 1 and "validation" in out[0]


def test_parquet_data_files_empty_when_no_match():
    assert _parquet_data_files("r/r", ["train/0.parquet"], "test") == []
