from StructureChunker.utils import extract_pdf_pages , extract_document_pages

from StructureChunker.parser import extract_document_structure

from StructureChunker.node_builder import (
    build_hierarchy,
    assign_end_pages
)

from StructureChunker.recursive_splitter import (
    recursively_split_large_nodes
)

from StructureChunker.chunk_exporter import export_chunks


def attach_text_to_nodes(nodes, pages):

    flat = []
    print(f"Attaching text to nodes ---- initial flat[] = {flat}")
    def dfs(node):
        print(f"dfs called : node = {node} ")
        flat.append(node)
        print(f"Updated flat: {flat}")
        for child in node.children:
            print(f"dfs called : child = {child} ")
            dfs(child)

    for n in nodes:
        print(f"dfs called : n = {n} ")
        dfs(n)

    for node in flat:
        start = node.start_page - 1
        end = node.end_page
        print(f"dfs called : node = {node} ")
        print(f"start page = {node.start_page} end page = {node.end_page} ")
        text = ""
        for p in pages[start:end]:
            text += p["text"] + "\n"
        print(f"text : {text}")
        node.text = text


def build_structure_chunks(pdf_path):
    print(f"Building structure chunks ---- pdf_path = {pdf_path} .... build_structure_chunks called ")
    pages = extract_document_pages(pdf_path)

    flat_nodes = extract_document_structure(pages)

    hierarchy = build_hierarchy(flat_nodes)

    assign_end_pages(hierarchy, len(pages))

    attach_text_to_nodes(hierarchy, pages)

    for node in hierarchy:

        recursively_split_large_nodes(
            node,
            pages
        )

    chunks = export_chunks(hierarchy)
    print(f"Chunks : {chunks}")
    return chunks

def build_structure_chunks_from_text(text):

    pages = [{
        "page_num": 1,
        "text": text
    }]

    structure = extract_document_structure(
        pages
    )

    hierarchy = build_hierarchy(structure)

    assign_end_pages(
        hierarchy,
        len(pages)
    )

    recursively_split_large_nodes(
        hierarchy , pages
    )

    chunks = export_chunks(
        hierarchy,
        pages
    )

    return chunks