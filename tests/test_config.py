from experiments.config import ExperimentConfig


def test_defaults_are_as_specified():
    cfg = ExperimentConfig()
    assert cfg.model == "openai/gpt-oss-120b:free"
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.arms == ["token", "structure", "ahc", "semantic", "flat"]
    assert cfg.datasets == ["qasper", "quality", "narrativeqa"]
    assert cfg.subset_sizes == {"qasper": 50, "quality": 50, "narrativeqa": 25}
    assert cfg.seeds == [0, 1, 2]
    assert cfg.retrieval_max_tokens == 2000
    assert cfg.leaf_max_tokens == 100
    assert cfg.summarization_max_tokens == 150
    assert cfg.qa_max_tokens == 150
    assert cfg.tau == 0.5
    assert cfg.semantic_breakpoint_percentile == 85
    assert cfg.cache_dir == ".llm_cache"
    assert cfg.results_dir == "results"
    assert cfg.temperature == 0.0


def test_to_dict_from_dict_round_trip():
    cfg = ExperimentConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.to_dict() == d
    assert cfg2 == cfg


def test_save_load_round_trip(tmp_path):
    cfg = ExperimentConfig(model="some/other-model", seeds=[7, 8])
    p = tmp_path / "config.json"
    cfg.save(str(p))
    assert p.exists()
    loaded = ExperimentConfig.load(str(p))
    assert loaded == cfg
    assert loaded.model == "some/other-model"
    assert loaded.seeds == [7, 8]


def test_overriding_fields_works():
    cfg = ExperimentConfig(
        model="x/y",
        subset_sizes={"qasper": 5},
        temperature=0.7,
    )
    assert cfg.model == "x/y"
    assert cfg.subset_sizes == {"qasper": 5}
    assert cfg.temperature == 0.7
    # untouched defaults remain
    assert cfg.arms == ["token", "structure", "ahc", "semantic", "flat"]


def test_mutable_defaults_not_shared():
    a = ExperimentConfig()
    b = ExperimentConfig()
    a.arms.append("zzz")
    a.subset_sizes["qasper"] = 999
    a.seeds.append(42)
    assert "zzz" not in b.arms
    assert b.subset_sizes["qasper"] == 50
    assert 42 not in b.seeds


def test_from_dict_ignores_unknown_keys():
    d = ExperimentConfig().to_dict()
    d["totally_unknown_key"] = "ignore me"
    cfg = ExperimentConfig.from_dict(d)
    assert not hasattr(cfg, "totally_unknown_key")
    assert cfg == ExperimentConfig()
