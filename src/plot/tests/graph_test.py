from bokeh.models import TabPanel
from bokeh.models import Tabs
from bokeh.plotting import show
from networkx import bfs_edges
from rich.traceback import install

from src.games import rock_paper_scissors
from src.plot.bokeh import plot_EdgesAndLinkedNodes
from src.plot.bokeh import plot_NodesAndAdjacentNodes
from src.plot.bokeh import plot_NodesAndLinkedEdges
from src.plot.matplotlib import graph_tree
from src.plot.matplotlib import show_plot
from src.plot.utils import hierarchy_pos

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
    edge_data["weight"] = 100

    # Check that the value was changed
    assert G.get_edge_data("R1", "B1")["weight"] == 100


def test_plot_rps():
    """Test that graphing a network if functional.  Does not verify that the plot is correct, only that the plotting functions can be called without error."""
    G = rock_paper_scissors()

    edge_updates = G.edges()
    graph_tree(G, x_size=14, y_size=9, target_edges=edge_updates)
    show_plot()


def test_bokeh_rps():
    """Renders the RPS game using Bokeh instead of Matplotlib.  This provides a HTML experience with interactive features not availble in the standard Matplotlib library."""

    # We need a RPS game where all nodes are integers
    G = rock_paper_scissors()
    pos_dict = hierarchy_pos(G, "root")

    # TODO: Add aliasing into the networkx graph!

    ## ------- Add interactive functions ------
    plot1 = plot_NodesAndLinkedEdges(G, pos_dict)
    plot2 = plot_EdgesAndLinkedNodes(G, pos_dict, plot1)
    plot3 = plot_NodesAndAdjacentNodes(G, pos_dict, plot1)

    # Create tabs and link them
    tab1 = TabPanel(child=plot1, title="Nodes and Linked Edges")
    tab2 = TabPanel(child=plot2, title="Edges and Linked Nodes")
    tab3 = TabPanel(child=plot3, title="Nodes and Adjacent Nodes")

    # Generate the plot
    show(Tabs(tabs=[tab1, tab2, tab3], sizing_mode="scale_both"))


if __name__ == "__main__":
    test_bokeh_rps()
