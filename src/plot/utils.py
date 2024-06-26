"""Common plotting helper utility functions.  These functions assist in formatting, extracting, or displaying information from the Networkx graph.
"""
import random

import networkx as nx
import numpy as np

# Constants
MIN_EDGE_WIDTH = 0.2


def get_node_attrs(G: nx.Graph) -> tuple[list, list]:
    """Get the node attributes based on user settings.

    Args:
        G (nx.Graph): The networkx graph.

    Returns:
        tuple[list, list]: ids, colors
    """
    # Configure node properties
    node_ids = []
    node_colors = []

    for node_id in G.nodes():
        node_ids.append(node_id)

        if node_id[0] == "R":
            node_colors.append("mistyrose")
        elif node_id[0] == "T":
            node_colors.append("lightgrey")
        elif node_id[0] == "B":
            node_colors.append("lightcyan")
        else:
            node_colors.append("navajowhite")

    return node_ids, node_colors


def get_edge_attrs(G: nx.Graph) -> tuple[list, list, list]:
    """Get the edge attribtues.  These are based on user settings.

    Args:
        G (nx.Graph): The networkx graph.

    Returns:
        tuple[list, list, list]: edges, colors, widths
    """

    # Configure edge properties
    eps = np.spacing(np.float32(1.0))
    edge_links = []
    edge_colors = []
    edge_widths = []

    for u, v, data in G.edges(data=True):
        edge_links.append((u, v))

        try:  # Set edge attributes based on data
            m = data["s"]["m"]
            n = data["s"]["n"] + eps
            edge_width = max((m / n * 5) ** 2, MIN_EDGE_WIDTH)
            edge_widths.append(edge_width)

            if data["player"] == "cat":
                edge_colors.append("red")
            elif data["player"] == "dog":
                edge_colors.append("blue")
            else:
                edge_colors.append("grey")

        except:  # Default values if no edge data available
            edge_widths.append(MIN_EDGE_WIDTH)
            edge_colors.append("grey")

    return edge_links, edge_colors, edge_widths


def hierarchy_pos(
    G,
    root=None,
    width=1.0,
    vert_gap=0.2,
    vert_loc=0,
    xcenter=0.5,
) -> dict[str, tuple]:
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Args:
        G: the graph (must be a tree)
            root: the root node of current branch
            - if the tree is directed and this is not given,
            the root will be found and used
            - if the tree is directed and this is given, then
            the positions will be just for the descendants of this node.
            - if the tree is undirected and not given,
            then a random choice will be used.

        width: horizontal space allocated for this branch - avoids overlap with other branches
        vert_gap: gap between levels of hierarchy
        vert_loc: vertical location of root
        xcenter: horizontal location of root

    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G,
        root,
        width=1.0,
        vert_gap=0.2,
        vert_loc=0,
        xcenter=0.5,
        pos=None,
        parent=None,
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
