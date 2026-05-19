from StructureChunker.build_chunks import (
    build_structure_chunks
)

chunks = build_structure_chunks(
    "StructureChunker/test_docs/sample.txt"
)

for chunk in chunks:
    print("=" * 80)
    print("TITLE:", chunk.title)
    print("LEVEL:", chunk.level)
    print("PAGES:", chunk.start_page, "-", chunk.end_page)
    print()
    print(chunk.text[:1000])