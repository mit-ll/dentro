import uuid
from copy import deepcopy

import networkx as nx


def add_node_alias_info(G: nx.Graph, aliased_nodes: list):
    """Add node aliasing information.  This is used to make it easier to decipher which nodes are aliased with other nodes.

    Args:
        G (nx.Graph): Networkx graph.
        aliased_nodes (list): Nodes to be aliased.
    """
    # Assign alias links to the edge paths
    for aliased_node in aliased_nodes:
        # Update nodes with alias information
        current_node_aliases = deepcopy(aliased_nodes)
        current_node_aliases.remove(aliased_node)
        G.nodes[aliased_node]["aliases"] = current_node_aliases


def add_node_aliases(G: nx.Graph, aliased_nodes: list, aliased_stats: list):
    """Used to ensure that edges of a list of nodes are properly aliased.  When two or more nodes are aliased it means that their edge probablity distributions must be equal.  An agent cannot distinguished between aliased nodes and therefore must have a common set of probability distributions across aliased nodes.

    Args:
        G (nx.Graph): Networkx graph.
        aliased_nodes (list): A list of aliased nodes.
        aliased_stats (list): These are the initial probabilities that are assigned to the aliased nodes.  It must be equal to the number of decision edges that are common across all aliased nodes.

    Raises:
        ValueError: Throws an error when the number of decision edges do not match the number of `aliased_stats`.
    """
    # Assign alias links to the edge paths
    for aliased_node in aliased_nodes:
        # Identify add edge paths from the parent node
        aliased_edges = list(nx.bfs_edges(G, aliased_node, depth_limit=1))
        aliased_edge_ids = [str(uuid.uuid4()) for ii in range(0, len(aliased_edges))]

        # The number of sucessors must equal the number of aliased links
        check1 = len(aliased_edges) == len(aliased_stats)
        if check1 is False:
            raise ValueError("Number of decision points do not match the alias links!")

        # Pair each edge with the alias link
        for aliased_edge, aliased_stat, aliased_edge_id in zip(
            aliased_edges,
            aliased_stats,
            aliased_edge_ids,
        ):
            current_edge_aliases = deepcopy(aliased_edges)
            current_edge_aliases.remove(aliased_edge)

            aliased_stat["id"] = aliased_edge_id
            G.edges[aliased_edge]["s"] = aliased_stat


def add_aliasing(G: nx.Graph, aliased_nodes: list, aliased_stats: list):
    """Used to ensure that edges of a list of nodes are properly aliased.  When two or more nodes are aliased it means that their edge probablity distributions must be equal.  An agent cannot distinguished between aliased nodes and therefore must have a common set of probability distributions across aliased nodes.

    Args:
        G (nx.Graph): Networkx graph.
        aliased_nodes (list): A list of aliased nodes.
        aliased_stats (list): These are the initial probabilities that are assigned to the aliased nodes.  It must be equal to the number of decision edges that are common across all aliased nodes.

    Raises:
        ValueError: Throws an error when the number of decision edges do not match the number of `aliased_stats`.
    """

    add_node_alias_info(G, aliased_nodes)
    add_node_aliases(G, aliased_nodes, aliased_stats)


def init_ev(G: nx.Graph):
    """Initialize the expected values for a new Graph.  Assumes that all expected values are zero at initialization.

    Args:
        G (nx.Graph): Networkx graph.
    """
    for node in G.nodes():
        if G.nodes[node].get("ev") is None:
            G.nodes[node]["ev"] = {"blue": 0, "red": 0}


def init_edges(G: nx.Graph):
    """Initialize the edge probabilities for a new Graph.  Will naturally default to uniform distribution across all decisions at initialization.

    Args:
        G (nx.Graph): Networkx graph.
    """
    for edge in G.edges():
        if G.edges[edge]["s"].get("label") is None:
            m = G.edges[edge]["s"]["m"]
            n = G.edges[edge]["s"]["n"]
            G.edges[edge]["s"]["label"] = round(m / n * 100)


def init_nodes(G: nx.Graph):
    """Initialize all nodes to have consistent variables.

    Args:
        G (nx.Graph): Networkx graph.
    """
    for node in G.nodes():
        if G.nodes[node].get("aliases") is None:
            G.nodes[node]["aliases"] = ""
