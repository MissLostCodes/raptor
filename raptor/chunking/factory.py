from typing import Callable, List, Optional

import tiktoken

from .base import Chunker
from .llm_cache import JsonFileCache, cached_parse
from .structure_chunker import StructureChunker
from .structure_parser import DEFAULT_MODEL, llm_parse_titles
from .token_chunker import TokenChunker


def _default_cached_parse_fn(model: str, cache_dir: str) -> Callable[[str], List[str]]:
    cache = JsonFileCache(cache_dir)

    def parse_fn(text: str) -> List[str]:
        return cached_parse(text, model, cache, lambda t: llm_parse_titles(t, model=model))

    return parse_fn


def get_chunker(
    name: str,
    max_tokens: int = 100,
    tokenizer=None,
    model: str = DEFAULT_MODEL,
    cache_dir: str = ".llm_cache",
    parse_fn: Optional[Callable[[str], List[str]]] = None,
    tau: float = 0.5,
    embed_fn=None,
) -> Chunker:
    """Return a Chunker by arm name. `parse_fn` overrides the default cached LLM parser
    (used in tests to avoid the network). `embed_fn` injects sentence embeddings for the
    semantic arm; `tau` is the routing threshold for the ahc arm."""
    tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
    if name == "token":
        return TokenChunker(max_tokens=max_tokens, tokenizer=tokenizer)
    if name == "structure":
        fn = parse_fn or _default_cached_parse_fn(model, cache_dir)
        return StructureChunker(parse_fn=fn, max_tokens=max_tokens, tokenizer=tokenizer)
    if name == "semantic":
        from .semantic_chunker import SemanticChunker

        return SemanticChunker(
            embed_fn=embed_fn, max_tokens=max_tokens, tokenizer=tokenizer
        )
    if name == "ahc":
        from .ahc_chunker import AHCChunker

        fn = parse_fn or _default_cached_parse_fn(model, cache_dir)
        return AHCChunker(
            parse_fn=fn, tau=tau, max_tokens=max_tokens, tokenizer=tokenizer
        )
    raise ValueError(
        f"Unknown chunker: {name!r} (expected 'token', 'structure', 'semantic', or 'ahc')"
    )
