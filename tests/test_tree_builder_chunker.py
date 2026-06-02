"""TreeBuilder / RetrievalAugmentationConfig use a pluggable Chunker (offline)."""
from raptor.tree_builder import TreeBuilderConfig
from raptor.chunking.token_chunker import TokenChunker
from raptor.chunking.structure_chunker import StructureChunker


def test_config_defaults_to_token_chunker():
    cfg = TreeBuilderConfig()
    assert isinstance(cfg.chunker, TokenChunker)


def test_config_accepts_custom_chunker():
    sc = StructureChunker(parse_fn=lambda t: ["A", "B"], max_tokens=50)
    cfg = TreeBuilderConfig(chunker=sc)
    assert cfg.chunker is sc


def test_config_rejects_non_chunker():
    try:
        TreeBuilderConfig(chunker="not-a-chunker")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_log_config_mentions_chunker():
    cfg = TreeBuilderConfig()
    assert "Chunker:" in cfg.log_config()


def _stub_qa_model():
    # Avoid constructing a real OpenAI client (which needs an API key) in tests.
    from raptor.QAModels import BaseQAModel

    class _FakeQA(BaseQAModel):
        def answer_question(self, context, question):
            return ""

    return _FakeQA()


def test_retrieval_augmentation_config_passes_chunker_through():
    # RetrievalAugmentationConfig must forward tb_chunker into the (cluster) tree
    # builder config so the runner can inject an arm's chunker into the frozen pipeline.
    from raptor.RetrievalAugmentation import RetrievalAugmentationConfig

    sc = StructureChunker(parse_fn=lambda t: ["A", "B"], max_tokens=50)
    cfg = RetrievalAugmentationConfig(tb_chunker=sc, qa_model=_stub_qa_model())
    assert cfg.tree_builder_config.chunker is sc


def test_retrieval_augmentation_config_defaults_to_token_chunker():
    from raptor.RetrievalAugmentation import RetrievalAugmentationConfig

    cfg = RetrievalAugmentationConfig(qa_model=_stub_qa_model())
    assert isinstance(cfg.tree_builder_config.chunker, TokenChunker)
