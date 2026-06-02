# raptor/__init__.py
#
# Lazy public API (PEP 562). The heavy ML stack (torch, sentence-transformers, faiss,
# umap) is only imported when a name that needs it is actually accessed. This keeps
# `from raptor.chunking import ...` and `from raptor.tree_builder import ...` importable
# in a light (CPU-only) environment for the offline test suite, while
# `from raptor import RetrievalAugmentation` still works for real runs.

# Maps public name -> submodule it lives in. Resolved lazily on first access.
_LAZY_EXPORTS = {
    "ClusterTreeBuilder": "cluster_tree_builder",
    "ClusterTreeConfig": "cluster_tree_builder",
    "BaseEmbeddingModel": "EmbeddingModels",
    "OpenAIEmbeddingModel": "EmbeddingModels",
    "SBertEmbeddingModel": "EmbeddingModels",
    "FaissRetriever": "FaissRetriever",
    "FaissRetrieverConfig": "FaissRetriever",
    "BaseQAModel": "QAModels",
    "GPT3QAModel": "QAModels",
    "GPT3TurboQAModel": "QAModels",
    "GPT4QAModel": "QAModels",
    "UnifiedQAModel": "QAModels",
    "RetrievalAugmentation": "RetrievalAugmentation",
    "RetrievalAugmentationConfig": "RetrievalAugmentation",
    "BaseRetriever": "Retrievers",
    "BaseSummarizationModel": "SummarizationModels",
    "GPT3SummarizationModel": "SummarizationModels",
    "GPT3TurboSummarizationModel": "SummarizationModels",
    "TreeBuilder": "tree_builder",
    "TreeBuilderConfig": "tree_builder",
    "TreeRetriever": "tree_retriever",
    "TreeRetrieverConfig": "tree_retriever",
    "Node": "tree_structures",
    "Tree": "tree_structures",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):  # PEP 562
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'raptor' has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
