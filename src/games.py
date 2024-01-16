import uuid
from copy import deepcopy

import networkx as nx


def add_node_aliasing(G: nx.Graph, aliased_nodes: list):
    """Add node aliasing information.

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


def add_edge_aliasing(G: nx.Graph, aliased_nodes: list, aliased_stats: list):
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
            str_edge_aliases = str(current_edge_aliases)  # must be string for GML

            aliased_stat["id"] = aliased_edge_id
            aliased_stat["aliases"] = str_edge_aliases
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

    add_node_aliasing(G, aliased_nodes)
    add_edge_aliasing(G, aliased_nodes, aliased_stats)


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
            G.edges[edge]["s"]["label"] = round(
                G.edges[edge]["s"]["m"] / G.edges[edge]["s"]["n"] * 100
            )


def rock_paper_scissors() -> nx.Graph:
    """An example of Rock-Paper-Scissors game where Red's decisions are aliased from the perspective of Blue.  This is a zero-sum game where neither player starts off using the Nash Equilibrium strategy.

    Returns:
        nx.Graph: Networkx graph.
    """
    G = nx.DiGraph()

    # Root starting point
    G.add_edge("root", "R1", type="random", player="arbiter", s={"m": 1, "n": 1})

    # Red makes moves first
    G.add_edge(
        "R1", "B1", type="decision", player="red", s={"m": 1, "n": 5}, action="rock"
    )
    G.add_edge(
        "R1", "B2", type="decision", player="red", s={"m": 3, "n": 5}, action="paper"
    )
    G.add_edge(
        "R1", "B3", type="decision", player="red", s={"m": 1, "n": 5}, action="scissors"
    )

    # Blue possible moves
    G.add_edge("B1", "T1", type="decision", player="blue", action="rock")
    G.add_edge("B1", "T2", type="decision", player="blue", action="paper")
    G.add_edge("B1", "T3", type="decision", player="blue", action="scissors")

    G.add_edge("B2", "T4", type="decision", player="blue", action="rock")
    G.add_edge("B2", "T5", type="decision", player="blue", action="paper")
    G.add_edge("B2", "T6", type="decision", player="blue", action="scissors")

    G.add_edge("B3", "T7", type="decision", player="blue", action="rock")
    G.add_edge("B3", "T8", type="decision", player="blue", action="paper")
    G.add_edge("B3", "T9", type="decision", player="blue", action="scissors")

    # Set terminal values
    G.add_node("T1", ev={"blue": 0, "red": 0}, type="terminal")
    G.add_node("T2", ev={"blue": 1, "red": -1}, type="terminal")
    G.add_node("T3", ev={"blue": -1, "red": 1}, type="terminal")

    G.add_node("T4", ev={"blue": -1, "red": 1}, type="terminal")
    G.add_node("T5", ev={"blue": 0, "red": 0}, type="terminal")
    G.add_node("T6", ev={"blue": 1, "red": -1}, type="terminal")

    G.add_node("T7", ev={"blue": 1, "red": -1}, type="terminal")
    G.add_node("T8", ev={"blue": -1, "red": 1}, type="terminal")
    G.add_node("T9", ev={"blue": 0, "red": 0}, type="terminal")

    # Aliased nodes
    add_aliasing(
        G,
        aliased_nodes=["B1", "B2", "B3"],
        aliased_stats=[{"m": 1, "n": 3}, {"m": 1, "n": 3}, {"m": 1, "n": 3}],
    )

    # Initialize missing variables
    init_ev(G)
    init_edges(G)

    return G


if __name__ == "__main__":
    rock_paper_scissors()
