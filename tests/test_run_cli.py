"""Offline test for experiments.run CLI: --dry-run builds config, no network."""

from experiments.config import ExperimentConfig
from experiments import run as run_cli


def test_dry_run_builds_expected_config():
    cfg = run_cli.main(
        [
            "--arms", "token,structure,ahc,semantic,flat",
            "--datasets", "qasper,quality,narrativeqa",
            "--model", "openai/gpt-oss-120b:free",
            "--cache-dir", ".llm_cache",
            "--results-dir", "results",
            "--seed", "0",
            "--n-qasper", "10",
            "--n-quality", "8",
            "--n-narrativeqa", "5",
            "--dry-run",
        ]
    )
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.arms == ["token", "structure", "ahc", "semantic", "flat"]
    assert cfg.datasets == ["qasper", "quality", "narrativeqa"]
    assert cfg.model == "openai/gpt-oss-120b:free"
    assert cfg.cache_dir == ".llm_cache"
    assert cfg.results_dir == "results"
    assert cfg.subset_sizes == {"qasper": 10, "quality": 8, "narrativeqa": 5}


def test_dry_run_defaults_when_no_overrides():
    cfg = run_cli.main(["--dry-run"])
    # default subset sizes preserved.
    assert cfg.subset_sizes == {"qasper": 50, "quality": 50, "narrativeqa": 25}
    assert cfg.arms == ["token", "structure", "ahc", "semantic", "flat"]


def test_partial_size_override_keeps_other_defaults():
    cfg = run_cli.main(["--n-qasper", "3", "--dry-run"])
    assert cfg.subset_sizes["qasper"] == 3
    assert cfg.subset_sizes["quality"] == 50
    assert cfg.subset_sizes["narrativeqa"] == 25
