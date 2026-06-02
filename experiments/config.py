"""Experiment configuration for the RAPTOR chunking-arms reproduction harness.

The whole study compares chunking arms using the SAME gpt-oss model (via
OpenRouter) for BOTH summarization and QA, so results are comparable.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List


@dataclass
class ExperimentConfig:
    # --- model / OpenRouter ---
    model: str = "openai/gpt-oss-120b:free"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"

    # --- experiment grid ---
    arms: List[str] = field(
        default_factory=lambda: ["token", "structure", "ahc", "semantic", "flat"]
    )
    datasets: List[str] = field(
        default_factory=lambda: ["qasper", "quality", "narrativeqa"]
    )
    subset_sizes: Dict[str, int] = field(
        default_factory=lambda: {"qasper": 50, "quality": 50, "narrativeqa": 25}
    )
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # --- retrieval / chunking ---
    retrieval_max_tokens: int = 2000  # paper's collapsed-tree main setting
    leaf_max_tokens: int = 100  # RAPTOR leaf chunk size
    summarization_max_tokens: int = 150
    qa_max_tokens: int = 150
    tau: float = 0.5  # AHC routing threshold
    semantic_breakpoint_percentile: int = 85

    # --- io ---
    cache_dir: str = ".llm_cache"
    results_dir: str = "results"

    # --- generation ---
    temperature: float = 0.0

    def to_dict(self) -> dict:
        """Return a plain-dict representation (JSON-serializable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Build a config from a dict.

        Unknown keys are ignored gracefully so that configs saved by a newer
        or older version of the harness still load.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def save(self, path: str) -> None:
        """Write the config to ``path`` as JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load a config from a JSON file at ``path``."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
