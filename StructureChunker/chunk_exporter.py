import uuid

from .schemas import StructuredChunk


def export_chunks(nodes):
    """
    export chunks into pdf
    :param nodes:
    :return:
    """
    chunks = []
    def dfs(node, parent=None):

        if not node.children:
            chunk = StructuredChunk(
                chunk_id=str(uuid.uuid4()),
                title=node.title,
                level=node.level,
                text=node.text,
                start_page=node.start_page,
                end_page=node.end_page,
                parent_title=parent
            )
            chunks.append(chunk)
        for child in node.children:
            dfs(child, node.title)

    for node in nodes:
        dfs(node)
    print("Exported chunks")
    return chunks