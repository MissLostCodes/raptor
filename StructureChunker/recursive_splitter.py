from .utils import count_tokens
from .parser import extract_document_structure
from .schemas import SectionNode
from .config import *


def recursively_split_large_nodes(node, pages):
    """
    recursively_split_large_nodes(node, pages)
    :param node:
    :param pages:
    :return: node
    """
    print(f"recursively_split_large_nodes(node, pages) = {node}")
    token_count = count_tokens(node.text)
    print(f"Token count: {token_count}")
    if token_count < RECURSIVE_SPLIT_THRESHOLD:
        return node

    relevant_pages = pages[
        node.start_page - 1 : node.end_page
    ]

    sub_nodes = extract_document_structure(relevant_pages)

    if len(sub_nodes) <= 1:
        return node
    node.children = sub_nodes

    for child in node.children:
        recursively_split_large_nodes(child, relevant_pages)

    return node