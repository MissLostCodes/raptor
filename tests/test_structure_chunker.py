from raptor.chunking.structure_chunker import StructureChunker, locate_sections
from raptor.chunking.base import Chunker


DOC = (
    "Introduction\n"
    "This paper studies chunking for retrieval augmented generation.\n"
    "Methods\n"
    "We compare token chunking against structure aware chunking.\n"
    "Results\n"
    "Structure helps on papers but not on stories.\n"
)


def test_locate_sections_returns_increasing_offsets():
    offs = locate_sections(DOC, ["Introduction", "Methods", "Results"])
    assert offs[0] >= 0 and offs[1] > offs[0] and offs[2] > offs[1]


def test_locate_sections_marks_missing_titles_negative():
    offs = locate_sections(DOC, ["Introduction", "Nonexistent Heading"])
    assert offs[0] >= 0
    assert offs[1] == -1


def test_structure_chunker_is_a_chunker():
    assert isinstance(StructureChunker(parse_fn=lambda t: []), Chunker)


def test_splits_document_by_located_sections():
    sc = StructureChunker(
        parse_fn=lambda t: ["Introduction", "Methods", "Results"], max_tokens=1000
    )
    chunks = sc.chunk(DOC)
    # one chunk per section (no preamble before the first heading here)
    assert len(chunks) == 3
    assert chunks[0].startswith("Introduction")
    assert "Methods" in chunks[1]
    assert "Results" in chunks[2]
    assert all(c.strip() for c in chunks)


def test_falls_back_to_token_chunking_when_no_structure():
    # parser finds nothing -> must not return empty or whole-doc garbage
    sc = StructureChunker(parse_fn=lambda t: [], max_tokens=20)
    chunks = sc.chunk(DOC)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)


def test_oversized_section_is_token_split_for_repair():
    big_body = ". ".join(f"clause {i}" for i in range(300)) + "."
    doc = f"OnlySection\n{big_body}"
    sc = StructureChunker(parse_fn=lambda t: ["OnlySection"], max_tokens=20)
    chunks = sc.chunk(doc)
    # single located section but oversized -> repaired into many sub-chunks
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_empty_input_returns_empty():
    sc = StructureChunker(parse_fn=lambda t: ["X"], max_tokens=20)
    assert sc.chunk("") == []
