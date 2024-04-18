from pathlib import Path

import networkx as nx
from rich.traceback import install

from src.game.rps import rock_paper_scissors
from src.utils import load_aliases

install(show_locals=True)


def test_save_load_network():
    """Test the code needed to save/load a Networkx graph.  Alias loading is tested here to ensure that they have matching values."""

    source_G = rock_paper_scissors()

    save_path = "save/rock-paper-scissors/graphs"
    graph_path = f"{save_path}/rps.networkx"

    Path(save_path).mkdir(parents=True, exist_ok=True)

    nx.gml.write_gml(source_G, graph_path)
    loaded_G = nx.gml.read_gml(graph_path)
    aliases = load_aliases(loaded_G)

    # Verify original data
    assert loaded_G.edges[("B1", "T1")]["type"] == "decision"
    assert loaded_G.edges[("B1", "T1")]["player"] == "dog"
    assert loaded_G.edges[("B1", "T1")]["action"] == "rock"
    assert loaded_G.edges[("B1", "T1")]["s"]["m"] == 1
    assert loaded_G.edges[("B1", "T1")]["s"]["n"] == 3

    # Modify an aliased edge
    for index, (uuid, aliased_edge) in enumerate(aliases.items()):
        if index == 0:
            aliased_edge["m"] = 100
        elif index == 1:
            aliased_edge["m"] = 50
        elif index == 2:
            aliased_edge["m"] = 25
        else:
            raise ValueError("invalid index")

    # Verify that all aliased edges have been changed
    assert loaded_G.edges[("B1", "T1")]["s"]["m"] == 100
    assert loaded_G.edges[("B1", "T2")]["s"]["m"] == 50
    assert loaded_G.edges[("B1", "T3")]["s"]["m"] == 25

    assert loaded_G.edges[("B2", "T4")]["s"]["m"] == 100
    assert loaded_G.edges[("B2", "T5")]["s"]["m"] == 50
    assert loaded_G.edges[("B2", "T6")]["s"]["m"] == 25

    assert loaded_G.edges[("B3", "T7")]["s"]["m"] == 100
    assert loaded_G.edges[("B3", "T8")]["s"]["m"] == 50
    assert loaded_G.edges[("B3", "T9")]["s"]["m"] == 25
