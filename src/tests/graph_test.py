from networkx import bfs_edges
from rich.traceback import install
import time
from src.games import rock_paper_scissors, rock_paper_scissors_int
from src.plot import graph_tree
from src.plot import show_plot
from bokeh.models import (
    BoxSelectTool,
    Circle,
    HoverTool,
    MultiLine,
    NodesAndLinkedEdges,
    Plot,
    Range1d,
    TapTool,
)
from src.plot import hierarchy_pos
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx, show
import networkx as nx
import math
from bokeh.plotting import figure, show
from bokeh.models import GraphRenderer, Ellipse, StaticLayoutProvider
from bokeh.palettes import Spectral8, Plasma256
import networkx as nx
import networkx as nx

from bokeh.models import (
    BoxSelectTool,
    Circle,
    HoverTool,
    MultiLine,
    NodesAndAdjacentNodes,
    Plot,
    Range1d,
    TapTool,
    PanTool,
    WheelZoomTool,
    ResetTool,
)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx, show

from bokeh.models import (
    BoxSelectTool,
    Circle,
    EdgesAndLinkedNodes,
    HoverTool,
    MultiLine,
    Plot,
    Range1d,
    TapTool,
)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx, show


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
    G = rock_paper_scissors_int()
    pos_dict = hierarchy_pos(G, 1000)

    ## ------- Add interactive functions ------
    plot_NodesAndLinkedEdges(G, pos_dict)
    time.sleep(1)
    plot_EdgesAndLinkedNodes(G, pos_dict)
    time.sleep(1)
    plot_NodesAndAdjacentNodes(G, pos_dict)


def plot_NodesAndLinkedEdges(G, pos_dict):
    graph_renderer = preprocess(G, pos_dict)
    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesAndLinkedEdges()

    plot = Plot(width=1200, height=600)
    plot.title.text = "Graph Interaction - Nodes & Linked Edge"
    plot.add_tools(HoverTool(), TapTool(), BoxSelectTool(), PanTool(), WheelZoomTool(), ResetTool())
    plot.renderers.append(graph_renderer)

    show(plot)


def plot_EdgesAndLinkedNodes(G, pos_dict):
    graph_renderer = preprocess(G, pos_dict)
    graph_renderer.selection_policy = EdgesAndLinkedNodes()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    plot = Plot(width=1200, height=600)
    plot.title.text = "Graph Interaction - Edges & Linked Nodes"
    plot.add_tools(HoverTool(), TapTool(), BoxSelectTool(), PanTool(), WheelZoomTool(), ResetTool())
    plot.renderers.append(graph_renderer)

    show(plot)


def plot_NodesAndAdjacentNodes(G, pos_dict):
    graph_renderer = preprocess(G, pos_dict)
    graph_renderer.selection_policy = NodesAndAdjacentNodes()
    graph_renderer.inspection_policy = NodesAndAdjacentNodes()

    plot = Plot(width=1200, height=600)
    plot.title.text = "Graph Interaction - Nodes & Adjacent Nodes"
    plot.add_tools(HoverTool(), TapTool(), BoxSelectTool(), PanTool(), WheelZoomTool(), ResetTool())
    plot.renderers.append(graph_renderer)

    show(plot)


def preprocess(G, pos_dict):
    # Set its height, width, and fill_color
    graph_renderer = GraphRenderer()
    graph_renderer.node_renderer.glyph = Circle(size=10, fill_color="colors")

    # assign a palette to ``fill_color`` and add it to the data source
    graph_renderer.node_renderer.data_source.data = dict(
        index=list(G.nodes()), fill_color=Plasma256
    )

    # Assign the edges
    start, end = [], []
    for x, y in G.edges():
        start.append(x)
        end.append(y)

    # This renders the edges between nodes
    graph_renderer.edge_renderer.data_source.data = dict(start=start, end=end)
    # This renders the positions of the nodes
    graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=pos_dict)

    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="#CCCCCC", line_alpha=0.8, line_width=5
    )
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

    return graph_renderer
