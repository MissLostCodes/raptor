"""Offline tests for experiments.pipeline.make_ra_config arm->chunker mapping."""

import pytest

from experiments.config import ExperimentConfig
from experiments.pipeline import make_ra_config
from raptor.QAModels import BaseQAModel
from raptor.chunking.token_chunker import TokenChunker
from raptor.chunking.structure_chunker import StructureChunker
from raptor.chunking.semantic_chunker import SemanticChunker
from raptor.chunking.ahc_chunker import AHCChunker
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


def test_make_ra_config_rejects_flat():
    with pytest.raises(ValueError):
        make_ra_config("flat", _cfg(), FakeSummModel(), FakeQAModel())


def test_make_ra_config_rejects_unknown():
    with pytest.raises(ValueError):
        make_ra_config("bogus", _cfg(), FakeSummModel(), FakeQAModel())
