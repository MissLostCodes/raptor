"""Offline tests for experiments.pipeline.make_ra_config arm->chunker mapping."""

import pytest

from experiments.config import ExperimentConfig
from experiments.pipeline import make_ra_config
from raptor.QAModels import BaseQAModel
from raptor.chunking.token_chunker import TokenChunker
from raptor.chunking.structure_chunker import StructureChunker
from raptor.chunking.semantic_chunker import SemanticChunker
from raptor.chunking.ahc_chunker import AHCChunker
from raptor.chunking.density_adaptive_chunker import DensityAdaptiveChunker
from raptor.SummarizationModels import BaseSummarizationModel


class FakeQAModel(BaseQAModel):
    def answer_question(self, context, question):
        return "fake"


class FakeSummModel(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        return "fake summary"


def _fake_parse_fn(text):
    return ["A title"]


def _fake_embed_fn(sentences):
    return [[0.0, 1.0] for _ in sentences]


def _cfg():
    return ExperimentConfig(cache_dir="/tmp/_unused_cache")


@pytest.mark.parametrize(
    "arm, expected_cls",
    [
        ("token", TokenChunker),
        ("structure", StructureChunker),
        ("semantic", SemanticChunker),
        ("ahc", AHCChunker),
        ("density", DensityAdaptiveChunker),
    ],
)
def test_make_ra_config_maps_arm_to_chunker(arm, expected_cls):
    cfg = _cfg()
    ra_cfg = make_ra_config(
        arm,
        cfg,
        FakeSummModel(),
        FakeQAModel(),
        parse_fn=_fake_parse_fn,
        embed_fn=_fake_embed_fn,
    )
    chunker = ra_cfg.tree_builder_config.chunker
    assert isinstance(chunker, expected_cls)
    # leaf_max_tokens threads through.
    assert ra_cfg.tree_builder_config.max_tokens == cfg.leaf_max_tokens


def test_make_ra_config_ahc_uses_tau():
    cfg = _cfg()
    ra_cfg = make_ra_config(
        "ahc", cfg, FakeSummModel(), FakeQAModel(), parse_fn=_fake_parse_fn
    )
    assert ra_cfg.tree_builder_config.chunker.tau == cfg.tau


def test_make_ra_config_density_threads_calibrated_size_fn():
    # When density_feature is configured, the density arm sizes leaves per document
    # via the calibrated size = a + b*feature map (verified through a leaf-size record).
    cfg = ExperimentConfig(
        cache_dir="/tmp/_unused_cache",
        density_feature="mean_word_len_chars_norm",
        density_a=-2010.0,
        density_b=3385.0,
        density_l_min=50,
        density_l_max=400,
    )
    ra_cfg = make_ra_config(
        "density", cfg, FakeSummModel(), FakeQAModel(), parse_fn=_fake_parse_fn
    )
    chunker = ra_cfg.tree_builder_config.chunker
    assert isinstance(chunker, DensityAdaptiveChunker)
    long_words = "characterization optimization hierarchical representational analysis"
    short_words = "a it is on to be by of we an or a it is on"
    assert chunker._leaf_size(long_words) > chunker._leaf_size(short_words)


def test_make_ra_config_density_without_feature_uses_placeholder():
    # No density_feature -> uncalibrated composite placeholder, still a valid size.
    cfg = _cfg()
    ra_cfg = make_ra_config(
        "density", cfg, FakeSummModel(), FakeQAModel(), parse_fn=_fake_parse_fn
    )
    chunker = ra_cfg.tree_builder_config.chunker
    size = chunker._leaf_size("some ordinary prose with a few words in it here today")
    assert cfg.density_l_min <= size <= cfg.density_l_max


def test_make_ra_config_rejects_flat():
    with pytest.raises(ValueError):
        make_ra_config("flat", _cfg(), FakeSummModel(), FakeQAModel())


def test_make_ra_config_rejects_unknown():
    with pytest.raises(ValueError):
        make_ra_config("bogus", _cfg(), FakeSummModel(), FakeQAModel())
