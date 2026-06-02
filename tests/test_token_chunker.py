from raptor.chunking.token_chunker import TokenChunker
from raptor.chunking.base import Chunker


def test_token_chunker_is_a_chunker():
    assert isinstance(TokenChunker(), Chunker)


def test_empty_text_returns_no_chunks():
    assert TokenChunker(max_tokens=50).chunk("") == []
    assert TokenChunker(max_tokens=50).chunk("   ") == []


def test_long_text_splits_into_multiple_chunks():
    text = ". ".join(f"sentence number {i}" for i in range(200)) + "."
    chunks = TokenChunker(max_tokens=20).chunk(text)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)
    # reassembled chunks contain all the original sentence markers
    joined = " ".join(chunks)
    assert "sentence number 0" in joined
    assert "sentence number 199" in joined
