from networkx import bfs_edges
from rich.traceback import install

from src.game.rps import rock_paper_scissors
from src.plot.bokeh import graph_tree as bokeh_graph_tree
from src.plot.matplotlib import graph_tree as matplotlib_graph_tree
from src.plot.matplotlib import show_plot

install(show_locals=True)


def test_get_successors():
    """Test function for getting successors."""
    G = rock_paper_scissors()

    # Get list of all successors
    successors = list(bfs_edges(G, "B1", depth_limit=1))

    # Verify that successors are correct
    assert successors[0] == ("B1", "T1")
    assert successors[1] == ("B1", "T2")
    assert successors[2] == ("B1", "T3")


def test_modify_edge():
    """Test modifying edge data variables.  A basic check to ensure that data is saved to properly to edges."""
    G = rock_paper_scissors()

    # Modify the value of an edge
    edge_data = G.get_edge_data("R1", "B1")
    edge_data["weight"] = 100  # type: ignore

    # Check that the value was changed
    assert G.get_edge_data("R1", "B1")["weight"] == 100


def test_matplotlib_rps():
    """Test that graphing a network using `matplotlib`.  Does not verify that the plot is correct, only that the plotting functions can be called without error."""

    G = rock_paper_scissors()
    edge_updates = G.edges()

    matplotlib_graph_tree(G, x_size=14, y_size=9, target_edges=edge_updates)
    show_plot()


def test_bokeh_rps(tmp_path):
    """Renders the RPS game using Bokeh.  This provides a HTML experience with interactive features not availble in the standard Matplotlib library."""

    G = rock_paper_scissors()

    bokeh_graph_tree(
        G=G,
        save_path=f"{str(tmp_path)}/bokeh_test.html",
        html=True,
    )
