"""Locks the lazy-import refactor: the library must import without the heavy ML stack.

These tests run in the *light* venv (no torch/sentence-transformers/faiss/umap). They
guarantee the chunking package and config construction stay importable on a CPU-only box
and on Colab before the GPU stack is installed.
"""
import sys


def test_import_raptor_does_not_pull_heavy_stack():
    import raptor  # noqa: F401

    # Lazy package must not have eagerly imported the heavy ML libraries.
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_sbert_constructs_without_loading_model():
    from raptor.EmbeddingModels import BaseEmbeddingModel, SBertEmbeddingModel

    m = SBertEmbeddingModel()
    assert isinstance(m, BaseEmbeddingModel)
    assert m.model is None  # model is loaded lazily on first create_embedding
    assert "sentence_transformers" not in sys.modules


def test_unknown_attribute_raises():
    import raptor

    try:
        raptor.DoesNotExist
        assert False, "expected AttributeError"
    except AttributeError:
        pass
