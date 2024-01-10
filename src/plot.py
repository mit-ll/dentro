import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

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

        # Set width of edge based on decision probability
        edge_width = max((data["s"]["m"] / (data["s"]["n"] + eps) * 5) ** 1.5, MIN_EDGE_WIDTH)
        edge_widths.append(edge_width)

        if data["player"] == "red":
            edge_colors.append("red")
        elif data["player"] == "blue":
            edge_colors.append("blue")
        else:
            edge_colors.append("grey")

    return edge_links, edge_colors, edge_widths


def format_nodes(G: nx.Graph, pos: dict):
    """Format the node colors based on the prefix of the node's name.

    ### TODO: This should probably be done in the data of the node instead.

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of every node in `(x, y)` coordinates.
    """

    _, node_colors = get_node_attrs(G)

    # Plot nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")


def format_edges(G: nx.Graph, pos: dict):
    """Formats the edges based on player type and user settings.

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of every node in `(x, y)` coordinates.
    """
    # Configure edge properties
    edge_links, edge_colors, edge_widths = get_edge_attrs(G)

    # Plot edges
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edge_links,
        width=edge_widths,
        alpha=0.5,
        edge_color=edge_colors,
        style="solid",
    )


def format_edge_labels(G: nx.Graph, pos: dict, target_edges: list[tuple[str, str]] | None = None):
    """Perform formatting of the edge labels for a specific set of edges.  This is done to draw special attention to a specific set of edges (i.e. when updating the probability distribution of a specific layer).

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of every node in `(x, y)` coordinates.
        target_edges (list[tuple[str, str]] | None, optional): The target edges to perform formatting upon.  This is usually for updating a specific set of edges where the designer wants to draw attention to. Defaults to None.
    """
    if target_edges is None:
        return

    eps = np.spacing(np.float32(1.0))
    edge_labels = {}
    edge_labelcolor = {}
    key_edges = set(target_edges)

    for u, v, data in G.edges(data=True):
        edge = (u, v)

        # Add the probability of decision
        old_decision_probability = data["s"]["label"]
        new_decision_probability = round(data["s"]["m"] / (data["s"]["n"] + eps) * 100)

        edge_labels[(u, v)] = f"{new_decision_probability}%"

        check1 = edge in key_edges
        check2 = new_decision_probability > old_decision_probability
        check3 = new_decision_probability < old_decision_probability

        if check1 and check2:
            edge_labelcolor[(u, v)] = "green"
        elif check1 and check3:
            edge_labelcolor[(u, v)] = "red"
        else:
            edge_labelcolor[(u, v)] = "black"

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={edge: edge_labels[edge]}, font_color=edge_labelcolor[edge]
        )


def add_decision_labels(G: nx.Graph, pos: dict):
    """Displays the `action` of each edge.  Helpful for understanding what each edge's decision/action means.

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of every node in `(x, y)` coordinates.
    """
    for u, v, data in G.edges(data=True):
        ux, uy = pos[u]
        vx, vy = pos[v]

        if data.get("action", False):
            plt.text(
                vx,
                vy + 0.03,
                s=data["action"],
                bbox=dict(facecolor="white", alpha=0.75),
                horizontalalignment="center",
                fontsize=8,
            )


def add_expected_values(G: nx.Graph, pos: dict):
    """Display the expected values for each player below every node.  This assumes that your game is non-zero sum which means that each player can have different expected values.  If your game is zero-sum, the nodes would only need to display a single value.

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of every node in `(x, y)` coordinates.
    """
    # Configure node properties
    for name, data in G.nodes(data=True):
        # Get (x,y) coordinates
        x, y = pos[name]

        # Get expected values
        ev_blue = round(data["ev"]["blue"], 2)
        ev_red = round(data["ev"]["red"], 2)

        # Plot on graph in textbox
        plt.text(
            x,
            y - 0.03,
            s=ev_blue,
            bbox=dict(facecolor="white", alpha=0.75),
            horizontalalignment="center",
            fontsize=7,
            color="blue",
            fontweight="bold",
        )
        plt.text(
            x,
            y - 0.05,
            s=ev_red,
            bbox=dict(facecolor="white", alpha=0.75),
            horizontalalignment="center",
            fontsize=7,
            color="red",
            fontweight="bold",
        )


def add_aliasing(G: nx.Graph, pos: dict):
    """Draws a dashed line between nodes that are aliased.  Aliasing means that there are two or more nodes and an agent cannot distinguish between them.  Hence, the agent must make a decision without knowledge of which node (or state) the agent is in.

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of every node in `(x, y)` coordinates.
    """
    # key variables
    alias_set = set()

    # Configure node properties
    for u, v, d in G.edges(data=True):
        if d["s"].get("alias", False):
            if d["s"]["alias"]:
                alias_set.add(u)

    # Iterate over all aliased ndoes
    alias_list = sorted(alias_set)
    for ii in range(0, len(alias_list) - 1):
        x1, y1 = pos[alias_list[ii]]
        x2, y2 = pos[alias_list[ii + 1]]

        plt.plot(
            (x1, x2),
            (y1, y2),
            color="gray",
            alpha=0.5,
            linewidth=2,
            linestyle="--",
            marker="o",
            markerfacecolor="None",
            markeredgewidth=2,
            markersize=5,
            markeredgecolor="gray",
            label="aliasing",
        )


def add_layer_updates(pos: dict, target_edges: list[tuple[str, str]] | None = None):
    """Highlight the target nodes to indicate that they have been updated.  This is purely for improving visualization of the plots.

    Args:
        pos (dict): Position of every node in `(x, y)` coordinates.
        target_edges (list[tuple[str, str]] | None, optional): A list of edges to apply this formatting to.  This function will only use the parent edge. Defaults to None.
    """
    if target_edges is None:
        return

    # Filter out only the parent nodes from the edges!
    updated_nodes = set(list(map(lambda x: x[0], target_edges)))

    for node in updated_nodes:
        x1, y1 = pos[node]

        plt.plot(
            x1,
            y1,
            color="green",
            alpha=0.5,
            linewidth=2,
            linestyle="--",
            marker="o",
            markerfacecolor="None",
            markeredgewidth=7,
            markersize=35,
            markeredgecolor="green",
            label="updating",
        )


def add_custom_legend():
    """Adds a custom legend to the plot.  The information in the legend is user defined."""

    # Create a legend
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2, alpha=0.5, linestyle="--", label="Aliasing"),
        Line2D(
            [0],
            [0],
            marker="o",
            alpha=0.5,
            linestyle="None",
            color="green",
            label="Updating",
            markerfacecolor="g",
            markersize=15,
        ),
    ]

    # Create the figure
    ax = plt.gca()
    ax.legend(handles=legend_elements, loc="upper right")


def graph_tree(
    G: nx.Graph,
    x_size: int = 12,
    y_size: int = 8,
    root: str = "root",
    target_edges: list[tuple[str, str]] | None = None,
) -> tuple[Figure, Axes]:
    """Graph the entire tree using all of the helper functions available.  The following functions are applied:

    * `hierarchy_pos`: Plot nodes and edges to follow top-down tree hierarchy.
    * `add_aliasing`: Plot aliasing between nodes.
    * `add_layer_updates`: Plot highlighting of nodes being updated.
    * `format_nodes`: Plot node colors based on name.
    * `format_edges`: Plot edge colors based on player Id.
    * `format_edge_labels`: Plot highlighting of edges being updated.
    * `add_decision_labels`: Plot the `action` of each edge.
    * `add_expected_values`: Plot the expected value of each node (assumes non-zero game where each player has their own expected values).
    * `add_custom_legend`: Plot a custom legend defined by user.

    Args:
        G (nx.Graph): Networkx graph.
        x_size (int, optional): The x-dimension of the figure. Defaults to 12.
        y_size (int, optional): The y-dimension of the figure. Defaults to 8.
        root (str, optional): The root node. Defaults to "root".
        target_edges (list[tuple[str, str]] | None, optional): A list of target edges to apply highlighting to.  This is reserved for situations where the designer want to point out a specific property of the graph. Defaults to None.

    Returns:
        tuple[Figure, Axes]: The figure and axes handles to the plot.
    """

    # Generate the size of the figure
    fig = plt.figure(figsize=(x_size, y_size), facecolor="white")
    ax = plt.gca()
    plt.box(False)
    pos = hierarchy_pos(G, root)

    # Underlays behind the graph nodes and edges
    add_aliasing(G, pos)
    add_layer_updates(pos, target_edges)

    # Format the existing nodes and edges
    format_nodes(G, pos)
    format_edges(G, pos)
    format_edge_labels(G, pos, target_edges)

    # Overlays on top of the graph
    add_decision_labels(G, pos)
    add_expected_values(G, pos)

    # Legend
    add_custom_legend()

    return fig, ax


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
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
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


@ray.remote
def create_plot(
    G: nx.Graph,
    filepath: str,
    x_size: int,
    y_size: int,
    layer_rollouts: list[tuple[str, str]],
    step: int,
    iteration: int,
) -> bool:
    """Ray wrapper function for generating plots.  This is used for parallel processing of many plots.

    Args:
        G (nx.Graph): Networkx graph.
        filepath (str): File path to save the plots.
        x_size (int): X dimension of the figure.
        y_size (int): Y dimension of the figure.
        layer_rollouts (list[tuple[str, str]]): Rollout decisions for a specific layer.  The downstream functions will make this into a set to eliminiate repeat edges.  Not ideal but sufficient for plotting purposes.
        step (int): The step (or layer) being updated for this particular plot.
        iteration (int): The current iteration of being updated.

    Returns:
        bool: Returns True if successful
    """
    # Show graph as image
    graph_tree(G, x_size=x_size, y_size=y_size, target_edges=layer_rollouts)
    plt.title(
        f"Iteration: {iteration} - Step: {step}!", fontname="Times New Roman Bold", weight="bold"
    )
    plt.savefig(f"{filepath}/iter:{iteration}-step{step}", dpi=300)
    plt.close()

    return True


def show_plot():
    """Helper function for removing plot axis and tightening layout.  Operates on the current figure that the user has active."""

    # General plotting
    ax = plt.gca()
    ax.margins(0.08)
    plt.tight_layout()
    plt.axis("off")
    plt.show()
