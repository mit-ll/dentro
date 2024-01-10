import networkx as nx
from bokeh.model import Model
from bokeh.models import BoxSelectTool
from bokeh.models import Circle
from bokeh.models import EdgesAndLinkedNodes
from bokeh.models import GraphRenderer
from bokeh.models import HoverTool
from bokeh.models import MultiLine
from bokeh.models import NodesAndAdjacentNodes
from bokeh.models import NodesAndLinkedEdges
from bokeh.models import PanTool
from bokeh.models import Plot
from bokeh.models import ResetTool
from bokeh.models import StaticLayoutProvider
from bokeh.models import TabPanel
from bokeh.models import Tabs
from bokeh.models import TapTool
from bokeh.models import Text
from bokeh.models import WheelZoomTool
from bokeh.palettes import Spectral4
from bokeh.plotting import show
from networkx import bfs_edges
from networkx import Graph
from rich.traceback import install

from src.games import rock_paper_scissors
from src.plot import get_edge_attrs
from src.plot import get_node_attrs
from src.plot import graph_tree
from src.plot import hierarchy_pos
from src.plot import show_plot
from src.utils import convertFromNumber
from src.utils import convertToNumber
from src.utils import relabel_nodes_int2str
from src.utils import relabel_nodes_str2int

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

    # Create tabs
    tab1 = TabPanel(child=plot1, title="Nodes and Linked Edges")
    tab2 = TabPanel(child=plot2, title="Edges and Linked Nodes")
    tab3 = TabPanel(child=plot3, title="Nodes and Adjacent Nodes")

    # Generate the plot
    show(Tabs(tabs=[tab1, tab2, tab3], sizing_mode="scale_both"))


def plot_NodesAndLinkedEdges(G: Graph, pos_dict: dict) -> Model:
    graph_renderer = bokeh_preprocess(G, pos_dict)
    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesAndLinkedEdges()

    plot = Plot(sizing_mode="scale_both")
    plot.title.text = "Graph Interaction - Nodes & Linked Edge"

    hover = HoverTool(
        tooltips=[("Node Id", "@node_id"), ("EV blue", "@ev_blue"), ("EV red", "@ev_red")]
    )
    plot.add_tools(hover, TapTool(), BoxSelectTool(), PanTool(), WheelZoomTool(), ResetTool())
    plot.renderers.append(graph_renderer)

    return plot


def plot_EdgesAndLinkedNodes(G: Graph, pos_dict: dict, linked_plot: Model) -> Model:
    graph_renderer = bokeh_preprocess(G, pos_dict)
    graph_renderer.selection_policy = EdgesAndLinkedNodes()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    plot = Plot(sizing_mode="scale_both", x_range=linked_plot.x_range, y_range=linked_plot.y_range)
    plot.title.text = "Graph Interaction - Edges & Linked Nodes"

    hover = HoverTool(
        tooltips=[
            ("Edge Id", "@edge_id"),
            ("Player", "@player"),
            ("m", "@stat_m"),
            ("n", "@stat_n"),
        ]
    )
    plot.add_tools(hover, TapTool(), BoxSelectTool(), PanTool(), WheelZoomTool(), ResetTool())
    plot.renderers.append(graph_renderer)

    return plot


def plot_NodesAndAdjacentNodes(G: Graph, pos_dict: dict, linked_plot: Model) -> Model:
    graph_renderer = bokeh_preprocess(G, pos_dict)
    graph_renderer.selection_policy = NodesAndAdjacentNodes()
    graph_renderer.inspection_policy = NodesAndAdjacentNodes()

    plot = Plot(sizing_mode="scale_both", x_range=linked_plot.x_range, y_range=linked_plot.y_range)
    plot.title.text = "Graph Interaction - Nodes & Adjacent Nodes"

    hover = HoverTool(
        tooltips=[("Node Id", "@node_id"), ("EV blue", "@ev_blue"), ("EV red", "@ev_red")]
    )
    plot.add_tools(hover, TapTool(), BoxSelectTool(), PanTool(), WheelZoomTool(), ResetTool())
    plot.renderers.append(graph_renderer)

    return plot


def bokeh_node_colors(G: nx.Graph) -> dict:
    """Takes a Graph with integer named nodes and assigns colors based on their string representation.

    Args:
        G (nx.Graph): _description_

    Returns:
        dict: Mapping of nodes to colors.
    """

    # Configure node properties
    node_colors = {}

    for node in G.nodes():
        node_str = convertFromNumber(node)
        if node_str[0] == "R":
            node_colors[node] = "mistyrose"
        elif node_str[0] == "T":
            node_colors[node] = "lightgrey"
        elif node_str[0] == "B":
            node_colors[node] = "lightcyan"
        else:
            node_colors[node] = "navajowhite"

    return node_colors


def networkx_datasync(G: Graph, graph: Model) -> Model:
    """Bokeh uses `ColumnDataSource` as its primary data sourcing server.  In order for Networkx data to be useful to Bokeh, it needs to be merged into the format Bokeh expects.  This function is designed to copy the data entries in Networkx into Bokeh's `ColumnDataSource`.

    Args:
        G (Graph): Networkx graph.
        graph (Model): Bokeh graph renderer.
    """

    G_int = relabel_nodes_str2int(G)

    # In order to set the attributes of a node you will need to add it to the
    # `graph.node_renderer.data_source.data`.  Then render the node color using the field name.
    node_ids, node_colors = get_node_attrs(G)
    edge_ids, edge_colors, edge_widths = get_edge_attrs(G)

    # The `ColumnDataSource` of the edge sub-renderer must have a "start" and "end" column.
    node_ev_blue, node_ev_red = [], []
    edge_start, edge_end, edge_player, edge_m, edge_n, edge_label = [], [], [], [], [], []

    for _, data in G.nodes(data=True):
        # Get expected values
        node_ev_blue.append(round(data["ev"]["blue"], 2))
        node_ev_red.append(round(data["ev"]["red"], 2))

    for x, y, data in G.edges(data=True):
        edge_start.append(x)
        edge_end.append(y)
        edge_player.append(data["player"])
        edge_m.append(data["s"]["m"])
        edge_n.append(data["s"]["n"])
        edge_label.append(data["s"]["label"])

    # Set the index for all `ColumnDataSource`
    graph.node_renderer.data_source.data["index"] = list(G_int.nodes())
    graph.edge_renderer.data_source.data["index"] = list(G_int.edges())

    # The ColumnDataSource of the edge sub-renderer must have a "start" and "end" column.
    edge_start, edge_end = [], []
    for x, y in G_int.edges():
        edge_start.append(x)
        edge_end.append(y)

    # Add fields to `ColumnDataSource` for `node_renderer`
    graph.node_renderer.data_source.data["node_id"] = node_ids
    graph.node_renderer.data_source.data["node_color"] = node_colors
    graph.node_renderer.data_source.data["ev_blue"] = node_ev_blue
    graph.node_renderer.data_source.data["ev_red"] = node_ev_red

    # Add fields to `ColumnDataSource` for `edge_renderer`
    graph.edge_renderer.data_source.data["edge_id"] = edge_ids
    graph.edge_renderer.data_source.data["edge_color"] = edge_colors
    graph.edge_renderer.data_source.data["edge_width"] = edge_widths
    graph.edge_renderer.data_source.data["start"] = edge_start
    graph.edge_renderer.data_source.data["end"] = edge_end
    graph.edge_renderer.data_source.data["player"] = edge_player
    graph.edge_renderer.data_source.data["stat_m"] = edge_m
    graph.edge_renderer.data_source.data["stat_n"] = edge_n
    graph.edge_renderer.data_source.data["label"] = edge_label

    return graph


def bokeh_preprocess(G: Graph, pos_dict: dict) -> GraphRenderer:
    """The GraphRenderer model maintains separate sub-GlyphRenderers for graph nodes and edges.
    This lets you customize nodes by modifying the `node_renderer` property of GraphRenderer.
    Likewise, you can cutomize the edges by modifying the `edge_renderer` property of GraphRenderer.

    In order to customize the nodes and edges you must modify the `ColumnDataSource` directly.  All
    plotting attributes are dervied from the `ColumnDataSource` which is a custom dictionary.  The
    renderers must source their values from `ColumnDataSource` when assigning attributes like color,
    width, etc.

    Args:
        G (Graph): Networkx Graph.
        pos_dict (dict): Dictionary with position values for all nodes/edges.

    Returns:
        GraphRenderer: A Bokeh graph renderer.

    Refs:
        * https://docs.bokeh.org/en/latest/docs/user_guide/topics/graph.html
        * https://docs.bokeh.org/en/latest/docs/user_guide/basic/data.html
    """

    # The GraphRenderer model maintains separate sub-GlyphRenderers for graph nodes and edges.
    # This lets you customize nodes by modifying the `node_renderer` property of GraphRenderer.
    # Likewise, you can cutomize the edges by modifying the `edge_renderer` property of GraphRenderer.
    graph = GraphRenderer()
    networkx_datasync(G, graph)

    # Generate glyphs
    graph.node_renderer.glyph = Circle(size=25, fill_color="node_color")
    graph.edge_renderer.glyph = MultiLine(
        line_color="edge_color",
        line_alpha=0.5,
        line_width="edge_width",
    )

    # Set the layout of the nodes according to their positions
    G_int = relabel_nodes_str2int(G)
    pos_dict = hierarchy_pos(G_int, convertToNumber("root"))
    graph.layout_provider = StaticLayoutProvider(graph_layout=pos_dict)

    # Set rendering options for selection tools
    graph.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
    graph.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])
    graph.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
    graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

    return graph


if __name__ == "__main__":
    test_bokeh_rps()
