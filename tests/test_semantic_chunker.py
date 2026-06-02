import tiktoken

from raptor.chunking.base import Chunker
from raptor.chunking.semantic_chunker import SemanticChunker


def test_semantic_chunker_is_a_chunker():
    sc = SemanticChunker(embed_fn=lambda xs: [[1.0, 0.0] for _ in xs])
    assert isinstance(sc, Chunker)


def test_empty_returns_no_chunks():
    sc = SemanticChunker(embed_fn=lambda xs: [[1.0, 0.0] for _ in xs])
    assert sc.chunk("") == []
    assert sc.chunk("   ") == []


def test_splits_at_semantic_boundary():
    # Six sentences: first 3 about cats (~[1,0]), next 3 about finance (~[0,1]).
    text = (
        "Cats are small carnivorous mammals. "
        "Cats purr when they are content. "
        "Cats sleep for many hours each day. "
        "Stocks rose sharply on the exchange. "
        "Bond yields fell after the report. "
        "Investors moved capital into equities."
    )

    def embed_fn(sentences):
        vecs = []
        for s in sentences:
            low = s.lower()
            if "cat" in low:
                vecs.append([1.0, 0.0])
            else:
                vecs.append([0.0, 1.0])
        return vecs

    sc = SemanticChunker(embed_fn=embed_fn, max_tokens=1000, breakpoint_percentile=85)
    chunks = sc.chunk(text)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)


def test_oversized_group_is_token_split():
    # All sentences embed identically -> no semantic breakpoint -> one big group that
    # must be repaired by the token fallback because it exceeds max_tokens.
    text = ". ".join(f"clause number {i}" for i in range(120)) + "."

    def embed_fn(sentences):
        return [[1.0, 0.0] for _ in sentences]

    tokenizer = tiktoken.get_encoding("cl100k_base")
    sc = SemanticChunker(embed_fn=embed_fn, max_tokens=20, tokenizer=tokenizer)
    chunks = sc.chunk(text)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)
    for c in chunks:
        assert len(tokenizer.encode(c)) <= 20


def test_single_sentence_uses_fallback():
    sc = SemanticChunker(
        embed_fn=lambda xs: [[1.0, 0.0] for _ in xs], max_tokens=1000
    )
    chunks = sc.chunk("Just one sentence with no terminator")
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)
