from typing import List, Optional

import tiktoken

from ..utils import split_text
from .base import Chunker


class TokenChunker(Chunker):
    """Baseline chunker: the original RAPTOR sentence/token splitter."""

    def __init__(self, max_tokens: int = 100, overlap: int = 0, tokenizer=None):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tokenizer = tokenizer or tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return split_text(text, self.tokenizer, self.max_tokens, self.overlap)
