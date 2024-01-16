import networkx as nx

from src.game.utils import add_aliasing
from src.game.utils import init_edges
from src.game.utils import init_ev
from src.game.utils import init_nodes


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
    init_nodes(G)

    return G


if __name__ == "__main__":
    rock_paper_scissors()
