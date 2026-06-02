from typing import Callable, Dict, List, Optional, Tuple

import tiktoken

from .base import Chunker
from .structure_chunker import StructureChunker
from .structure_score import structure_score
from .token_chunker import TokenChunker


class AHCChunker(Chunker):
    """Adaptive Hybrid Chunking.

    Stage 1 (score): compute a deterministic structure score for the document.
    Stage 2 (route): if score >= tau, route to the structure chunker; otherwise route to
        the token chunker.
    Stage 3 (repair): repair is delegated to the routed chunker (StructureChunker does
        per-section token repair; TokenChunker is already token-bounded).

    After each `chunk` call, `last_score`, `last_route`, and `last_features` are recorded
    for routing-rate logging.
    """

    def __init__(
        self,
        parse_fn: Callable[[str], List[str]],
        tau: float = 0.5,
        max_tokens: int = 100,
        tokenizer=None,
        score_fn: Optional[Callable[[str], Tuple[float, Dict]]] = None,
        structure_chunker: Optional[Chunker] = None,
        token_chunker: Optional[Chunker] = None,
    ):
        self.parse_fn = parse_fn
        self.tau = tau
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")
        self.score_fn = score_fn or structure_score
        self.structure_chunker = structure_chunker or StructureChunker(
            parse_fn=parse_fn, max_tokens=max_tokens, tokenizer=self.tokenizer
        )
        self.token_chunker = token_chunker or TokenChunker(
            max_tokens=max_tokens, tokenizer=self.tokenizer
        )
        self.last_score: Optional[float] = None
        self.last_route: Optional[str] = None
        self.last_features: Optional[Dict] = None

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        score, features = self.score_fn(text)
        if score >= self.tau:
            route = "structure"
            chunks = self.structure_chunker.chunk(text)
        else:
            route = "token"
            chunks = self.token_chunker.chunk(text)

        self.last_score = score
        self.last_route = route
        self.last_features = features
        return [c for c in chunks if c.strip()]
