from raptor.chunking import get_chunker  # exercises full-package import
from raptor.chunking.token_chunker import TokenChunker
from raptor.chunking.structure_chunker import StructureChunker
from raptor.chunking.semantic_chunker import SemanticChunker
from raptor.chunking.ahc_chunker import AHCChunker


def test_get_token_chunker():
    c = get_chunker("token", max_tokens=50)
    assert isinstance(c, TokenChunker)
    assert c.max_tokens == 50


def test_get_structure_chunker_with_injected_parse_fn():
    c = get_chunker("structure", max_tokens=50, parse_fn=lambda t: ["A", "B"])
    assert isinstance(c, StructureChunker)


def test_get_semantic_chunker_with_injected_embed_fn():
    c = get_chunker("semantic", embed_fn=lambda xs: [[1.0, 0.0] for _ in xs])
    assert isinstance(c, SemanticChunker)


def test_get_ahc_chunker_with_injected_parse_fn():
    c = get_chunker("ahc", parse_fn=lambda t: ["A", "B"])
    assert isinstance(c, AHCChunker)


def test_unknown_chunker_raises():
    try:
        get_chunker("does-not-exist")
        assert False, "expected ValueError"
    except ValueError:
        pass
