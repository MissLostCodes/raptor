from .schemas import SectionNode


def build_hierarchy(flat_nodes):
    """
    builds a hierarchy from flat_nodes
    :param flat_nodes:
    :return:
    """
    print(f"Building hierarchy flat_nodes : {flat_nodes} ....  build_hierarchy(flat_nodes) called ")
    if not flat_nodes:
        return []
    stack = []
    root_nodes = []
    for node in flat_nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            root_nodes.append(node)
        stack.append(node)
    print(f"Root nodes : {root_nodes}")
    return root_nodes


def assign_end_pages(nodes, total_pages):
    """
    assign end pages to nodes
    :param nodes:
    :param total_pages:
    :return:
    """
    print(f"Assigning end pages to nodes : {nodes}")
    flat = []
    def dfs(node):
        flat.append(node)
        for child in node.children:
            dfs(child)
    for n in nodes:
        dfs(n)
    for i in range(len(flat)):
        if i < len(flat) - 1:
            flat[i].end_page = max(
                flat[i].start_page,
                flat[i + 1].start_page - 1
            )
        else:
            flat[i].end_page = total_pages