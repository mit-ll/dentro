import os
import pathlib
from copy import deepcopy

import networkx as nx
import numpy as np
import ray
from bokeh.io import output_file
from bokeh.io import save
from bokeh.model import Model
from bokeh.models import BoxSelectTool
from bokeh.models import Circle
from bokeh.models import ColumnDataSource
from bokeh.models import EdgesAndLinkedNodes
from bokeh.models import EdgesOnly
from bokeh.models import GraphRenderer
from bokeh.models import HoverTool
from bokeh.models import MultiLine
from bokeh.models import NodesAndAdjacentNodes
from bokeh.models import NodesAndLinkedEdges
from bokeh.models import NodesOnly
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

from src.plot.utils import get_edge_attrs
from src.plot.utils import get_node_attrs
from src.plot.utils import hierarchy_pos
from src.utils import convertFromNumber
from src.utils import convertToNumber
from src.utils import relabel_nodes_int2str
from src.utils import relabel_nodes_str2int


def link_plots(pri_plot: Model, src_plot: Model | None = None) -> Model:
    """Link an secondary plot to the axes of the primary plot.  This is needed when you want to sychronize multiple plots with the mouse wheel.

    Refs
    ```text
    [1](https://docs.bokeh.org/en/latest/docs/user_guide/interaction/linking.html)
    ```

    Args:
        pri_plot (Model): Primary plot.
        src_plot (Model | None, optional): Source plot to derive axes from. Defaults to None.

    Returns:
        Model: Plot model.
    """

    if src_plot is None:
        return pri_plot

    pri_plot.x_range = src_plot.x_range
    pri_plot.y_range = src_plot.y_range
    pri_plot.sizing_mode = "scale_both"

    return pri_plot


def create_networkx_plot(
    G: nx.Graph,
    pos_dict: dict,
    title: str,
    hover: Model,
    graph: Model,
    source_plot: Model | None = None,
) -> Model:
    """Add overlay information ontop of Networkx graph.  This will also add the overlay information needed for the tools.

    Args:
        G (nx.Graph): Networkx graph.
        pos_dict (dict): Position of all nodes.
        title (str): Title of the plot.
        hover (Model): Hover tool.
        graph (Model): Bokeh graph renderer.
        source_plot (Model | None, optional): Source plot to link plot axes to. Defaults to None.

    Returns:
        Model: Bokeh plot model.
    """
    plot = Plot(sizing_mode="scale_both", title=title)
    plot = link_plots(plot, source_plot)
    plot.add_tools(
        hover,
        TapTool(),
        BoxSelectTool(),
        PanTool(),
        WheelZoomTool(),
        ResetTool(),
    )
    plot.renderers.append(graph)

    # Add labels
    plot = add_node_labels(G, pos_dict, plot)
    plot = add_edge_labels(G, pos_dict, plot)
    # plot = add_aliasing(G, pos_dict, plot)

    return plot


def render_nodes(
    G: nx.Graph,
    pos_dict: dict,
    source_plot: Model | None = None,
) -> Model:
    """Highlight nodes when hovering over them.

    Refs
    ```
    [1](https://docs.bokeh.org/en/latest/docs/examples/topics/graph/interaction_nodeslinkededges.html)
    ```

    Args:
        G (nx.Graph): Networkx graph.
        pos_dict (dict): Position of nodes.
        source_plot (Model | None, optional): Source plot to link to. Defaults to None.

    Returns:
        Model: Plot model.
    """
    graph = preprocess(G, pos_dict)

    # Set linking
    graph.selection_policy = NodesOnly()
    graph.inspection_policy = NodesOnly()

    # Configure tools
    hover = HoverTool(
        tooltips=[
            ("Node Id", "@node_id"),
            ("EV blue", "@ev_blue"),
            ("EV red", "@ev_red"),
        ],
        renderers=[graph],
    )

    # Configure plot
    plot = create_networkx_plot(
        G=G,
        pos_dict=pos_dict,
        title="Graph Interaction - Nodes",
        source_plot=source_plot,
        hover=hover,
        graph=graph,
    )

    return plot


def render_edges(
    G: nx.Graph,
    pos_dict: dict,
    source_plot: Model | None = None,
) -> Model:
    """Highlight edges when hovering over them.

    Refs
    ```
    [1](https://docs.bokeh.org/en/latest/docs/examples/topics/graph/interaction_nodeslinkededges.html)
    ```

    Args:
        G (nx.Graph): Networkx graph.
        pos_dict (dict): Position of nodes.
        source_plot (Model | None, optional): Source plot to link to. Defaults to None.

    Returns:
        Model: Plot model.
    """
    graph = preprocess(G, pos_dict)

    # Set linking
    graph.selection_policy = EdgesOnly()
    graph.inspection_policy = EdgesOnly()

    # Configure tools
    hover = HoverTool(
        tooltips=[
            ("Edge Id", "@edge_id"),
            ("Player", "@player"),
            ("m", "@m"),
            ("n", "@n"),
            ("label", "@label"),
        ],
        renderers=[graph],
    )

    # Configure plot
    plot = create_networkx_plot(
        G=G,
        pos_dict=pos_dict,
        title="Graph Interaction - Edges",
        source_plot=source_plot,
        hover=hover,
        graph=graph,
    )

    return plot


def render_nodes_and_edges(
    G: nx.Graph,
    pos_dict: dict,
    source_plot: Model | None = None,
) -> Model:
    """Highlight nodes and edges.

    Refs
    ```
    [1](https://docs.bokeh.org/en/latest/docs/examples/topics/graph/interaction_nodeslinkededges.html)
    ```

    Args:
        G (nx.Graph): Networkx graph.
        pos_dict (dict): Position of nodes.
        source_plot (Model | None, optional): Source plot to link to. Defaults to None.

    Returns:
        Model: Plot model.
    """
    graph = preprocess(G, pos_dict)

    # Set linking
    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    # Configure tools
    hover = HoverTool(
        tooltips=[
            ("Node Id", "@node_id"),
            ("EV blue", "@ev_blue"),
            ("EV red", "@ev_red"),
        ],
        renderers=[graph],
    )

    # Configure plot
    plot = Plot(sizing_mode="scale_both", title="Graph Interaction - Nodes & Linked Edge")
    plot = link_plots(plot, source_plot)
    plot.add_tools(
        hover,
        TapTool(),
        BoxSelectTool(),
        PanTool(),
        WheelZoomTool(),
        ResetTool(),
    )
    plot.renderers.append(graph)

    # Add labels
    plot = add_node_labels(G, pos_dict, plot)
    plot = add_edge_labels(G, pos_dict, plot)

    return plot


def render_edges_and_linknodes(
    G: nx.Graph,
    pos_dict: dict,
    source_plot: Model | None = None,
) -> Model:
    """Highlight edges and their nodes attached to them.

    Refs
    ```
    [1](https://docs.bokeh.org/en/3.3.2/docs/examples/topics/graph/interaction_edgeslinkednodes.html)
    ```

    Args:
        G (nx.Graph): Networkx graph.
        pos_dict (dict): Position of nodes.
        source_plot (Model | None, optional): Source plot to link to. Defaults to None.

    Returns:
        Model: Plot model.
    """
    graph = preprocess(G, pos_dict)

    # Set linking
    graph.selection_policy = EdgesAndLinkedNodes()
    graph.inspection_policy = EdgesAndLinkedNodes()

    # Configure tools
    hover = HoverTool(
        tooltips=[
            ("Edge Id", "@edge_id"),
            ("Player", "@player"),
            ("m", "@m"),
            ("n", "@n"),
        ],
        renderers=[graph],
    )

    # Configure plot
    plot = Plot(
        sizing_mode="scale_both", title="Graph Interaction - Edges & Linked Nodes"
    )
    plot = link_plots(plot, source_plot)
    plot.add_tools(
        hover,
        TapTool(),
        BoxSelectTool(),
        PanTool(),
        WheelZoomTool(),
        ResetTool(),
    )
    plot.renderers.append(graph)

    # Add labels
    plot = add_node_labels(G, pos_dict, plot)
    plot = add_edge_labels(G, pos_dict, plot)

    return plot


def render_nodes_and_adjnodes(
    G: nx.Graph,
    pos_dict: dict,
    source_plot: Model | None = None,
) -> Model:
    """Highlight nodes and adjacent nodes.

    Refs
    ```
    [1](https://docs.bokeh.org/en/latest/docs/examples/topics/graph/interaction_nodesadjacentnodes.html)
    ```

    Args:
        G (nx.Graph): Networkx graph.
        pos_dict (dict): Position of nodes.
        source_plot (Model | None, optional): Source plot to link to. Defaults to None.

    Returns:
        Model: Plot model.
    """
    graph = preprocess(G, pos_dict)

    # Set linking
    graph.selection_policy = NodesAndAdjacentNodes()
    graph.inspection_policy = NodesAndAdjacentNodes()

    # Configure tools
    hover = HoverTool(
        tooltips=[
            ("Node Id", "@node_id"),
            ("EV blue", "@ev_blue"),
            ("EV red", "@ev_red"),
        ],
        renderers=[graph],
    )

    # Configure plot
    plot = Plot(
        sizing_mode="scale_both", title="Graph Interaction - Nodes & Adjacent Nodes"
    )
    plot = link_plots(plot, source_plot)
    plot.add_tools(
        hover,
        TapTool(),
        BoxSelectTool(),
        PanTool(),
        WheelZoomTool(),
        ResetTool(),
    )
    plot.renderers.append(graph)

    # Add labels
    plot = add_node_labels(G, pos_dict, plot)
    plot = add_edge_labels(G, pos_dict, plot)

    return plot


def get_node_colors(G: nx.Graph) -> dict:
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


def networkx_datasync(G: nx.Graph, graph: Model) -> Model:
    """Bokeh uses `ColumnDataSource` as its primary data sourcing server.  In order for Networkx data to be useful to Bokeh, it needs to be merged into the format Bokeh expects.  This function is designed to copy the data entries in Networkx into Bokeh's `ColumnDataSource`.

    Args:
        G (Graph): Networkx graph.
        graph (Model): Bokeh graph renderer.

    Returns:
        Model: Bokeh graph renderer.
    """

    G_int = relabel_nodes_str2int(G)

    # In order to set the attributes of a node you will need to add it to the
    # `graph.node_renderer.data_source.data`.  Then render the node color using the field name.
    node_id, node_color = get_node_attrs(G)
    edge_id, edge_color, edge_width = get_edge_attrs(G)

    # Preallocate emptry lists
    ev_blue, ev_red, start, end, player, m, n, label, action = ([] for _ in range(0, 9))

    for _, data in G.nodes(data=True):
        # Get expected values
        ev_blue.append(round(data["ev"]["dog"], 2))
        ev_red.append(round(data["ev"]["cat"], 2))

    for x, y, data in G.edges(data=True):
        try:  # Populate using data from Networkx graph
            player.append(data["player"])
            m.append(data["s"]["m"])
            n.append(data["s"]["n"])
            label.append(data["s"]["label"])
            action.append(data.get("action", "none"))
        except:  # Insert default values if no data available
            player.append(None)
            m.append(None)
            n.append(None)
            label.append(None)
            action.append(None)

    # Set the index for all `ColumnDataSource`
    graph.node_renderer.data_source.data["index"] = list(G_int.nodes())
    graph.edge_renderer.data_source.data["index"] = list(G_int.edges())

    # The ColumnDataSource of the edge sub-renderer must have a "start" and "end" column.
    for x, y in G_int.edges():
        start.append(x)
        end.append(y)

    # Add fields to `ColumnDataSource` for `node_renderer`
    graph.node_renderer.data_source.data["node_id"] = node_id
    graph.node_renderer.data_source.data["node_color"] = node_color
    graph.node_renderer.data_source.data["ev_blue"] = ev_blue
    graph.node_renderer.data_source.data["ev_red"] = ev_red

    # Add fields to `ColumnDataSource` for `edge_renderer`
    graph.edge_renderer.data_source.data["edge_id"] = edge_id
    graph.edge_renderer.data_source.data["edge_color"] = edge_color
    graph.edge_renderer.data_source.data["edge_width"] = edge_width
    graph.edge_renderer.data_source.data["start"] = start
    graph.edge_renderer.data_source.data["end"] = end
    graph.edge_renderer.data_source.data["player"] = player
    graph.edge_renderer.data_source.data["m"] = m
    graph.edge_renderer.data_source.data["n"] = n
    graph.edge_renderer.data_source.data["label"] = label
    graph.edge_renderer.data_source.data["action"] = action

    return graph


def build_node_datasource(G: nx.Graph, pos: dict) -> ColumnDataSource:
    """Build the node `ColumnDataSource` object and populate it with the data from Networkx graph.

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of all nodes.

    Returns:
        ColumnDataSource: Data source object.
    """
    x, y, text = ([] for _ in range(0, 3))

    for node, data in G.nodes(data=True):
        # Edge points in x, y
        cx, cy = pos[node]

        x.append(cx)
        y.append(cy)
        text.append(node)

    # Create a `ColumnDataSource`
    source = ColumnDataSource(dict(x=x, y=y, text=text))

    return source  # type: ignore


def add_node_labels(
    G: nx.Graph,
    pos: dict,
    plot: Model,
    text_font_size: str = "15px",
) -> Model:
    """Displays the name of each node.

    Refs
    ```
    [1](https://docs.bokeh.org/en/latest/docs/reference/models/glyphs/text.html)
    ```

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of every node in `(x, y)` coordinates.
        plot (Model): Plot model.
        text_font_size (str): Size of font in pixel space.

    Returns:
        Model: Plot model.
    """

    source = build_node_datasource(G, pos)

    glyph = Text(
        name="node_labels",
        x="x",
        y="y",
        text="text",
        text_align="center",
        text_baseline="middle",
        text_font_size=text_font_size,
        angle=0,
        text_color="black",
    )

    plot.add_glyph(source, glyph)

    return plot


def build_edge_datasource(G: nx.Graph, pos: dict) -> ColumnDataSource:
    """Create a `ColumnDataSource` for the edge properties.

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of all nodes.

    Returns:
        ColumnDataSource: ColumnDataSource for edge properties.
    """

    eps = np.spacing(np.float32(1.0))
    x, y, text, theta, label, color = ([] for _ in range(0, 6))

    for u, v, data in G.edges(data=True):
        # Edge points in x, y
        ux, uy = pos[u]  # x, y of parent node
        vx, vy = pos[v]  # x, y of child node

        # Edge coordinates for center
        cx = (ux + vx) / 2  # center x position
        cy = (uy + vy) / 2  # center y position
        dx = ux - vx  # delta x
        dy = uy - vy  # delta y
        ctheta = np.arctan2(dx, dy) * 180 / np.pi  # angle of edge
        action_str = data.get("action", None)
        prev_prob = data["s"]["label"]
        m = data["s"]["m"]
        n = data["s"]["n"]

        # Add the probability of decision
        old_decision_probability = data["s"]["label"]
        new_decision_probability = round(data["s"]["m"] / (data["s"]["n"] + eps) * 100)

        if new_decision_probability > old_decision_probability:
            edge_color = "green"
        elif new_decision_probability < old_decision_probability:
            edge_color = "red"
        else:
            edge_color = "black"

        x.append(cx)
        y.append(cy)
        text.append(action_str)
        label.append(f"{round(m/n*100)}%")
        theta.append(ctheta)
        color.append(edge_color)

    # Create a `ColumnDataSource`
    source = ColumnDataSource(
        dict(x=x, y=y, text=text, label=label, theta=theta, color=color)
    )

    return source  # type: ignore


def add_edge_labels(
    G: nx.Graph,
    pos: dict,
    plot: Model,
    text_font_size: str = "15px",
) -> Model:
    """Displays the `action` of each edge.  Helpful for understanding what each edge's decision/action means.

    Refs
    ```
    [1](https://docs.bokeh.org/en/latest/docs/reference/models/glyphs/text.html)
    ```

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of every node in `(x, y)` coordinates.
        plot (Model): Plot model.
        text_font_size (str): Size of font in pixel space.

    Returns:
        Model: Plot model.
    """

    source = build_edge_datasource(G, pos)

    glyph_action = Text(
        x="x",
        y="y",
        text="text",
        text_align="center",
        text_baseline="top",
        text_font_size=text_font_size,
        angle=0,
        text_color="color",
    )
    glyph_prob = Text(
        x="x",
        y="y",
        text="label",
        text_align="center",
        text_baseline="bottom",
        text_font_size=text_font_size,
        angle=0,
        text_color="color",
    )

    plot.add_glyph(source, glyph_action)
    plot.add_glyph(source, glyph_prob)

    return plot


def add_aliasing_edges(G: nx.Graph) -> nx.Graph:
    """Modified the original Networkx graph to add aliasing edges.  This is a post-processing step after the original Networkx graph has already been created.

    Args:
        G (nx.Graph): Networkx graph.
        pos (dict): Position of every node in `(x, y)` coordinates.

    Returns:
        nx.Graph: The modified Networkx graph with aliasing.
    """
    # key variables
    G_copy = deepcopy(G)

    # Configure node properties
    for node, data in G.nodes(data=True):
        # Check whether an alias exists for the edge
        for alias in data["aliases"]:
            G_copy.add_edge(node, alias)

    return G_copy


def preprocess(
    G: nx.Graph,
    pos_dict: dict,
    circle_radius: float = 0.01,
    alpha: float = 0.5,
) -> Model:
    """The GraphRenderer model maintains separate sub-GlyphRenderers for graph nodes and edges. This lets you customize nodes by modifying the `node_renderer` property of GraphRenderer. Likewise, you can cutomize the edges by modifying the `edge_renderer` property of GraphRenderer.

    In order to customize the nodes and edges you must modify the `ColumnDataSource` directly.  All plotting attributes are dervied from the `ColumnDataSource` which is a custom dictionary.  The renderers must source their values from `ColumnDataSource` when assigning attributes like color, width, etc.

    Refs
    ```
    [1](https://docs.bokeh.org/en/latest/docs/user_guide/topics/graph.html)
    [2](https://docs.bokeh.org/en/latest/docs/user_guide/basic/data.html)
    ```

    Args:
        G (Graph): Networkx Graph.
        pos_dict (dict): Dictionary with position values for all nodes/edges.
        circle_radius (float): Radius of circle in data units.
        alpha (float): Alpha of circle.

    Returns:
        GraphRenderer: A Bokeh graph renderer.
    """

    # The GraphRenderer model maintains separate sub-GlyphRenderers for graph nodes and edges.  This lets you customize nodes by modifying the `node_renderer` property of GraphRenderer.  Likewise, you can cutomize the edges by modifying the `edge_renderer` property of GraphRenderer.
    graph = GraphRenderer()
    pos_dict = hierarchy_pos(relabel_nodes_str2int(G), convertToNumber("root"))
    G_copy = add_aliasing_edges(G)

    # Synchronize the networkx graph
    networkx_datasync(G_copy, graph)

    # Set the layout of the nodes according to their positions
    graph.layout_provider = StaticLayoutProvider(graph_layout=pos_dict)

    # Set selection glyphs
    graph.node_renderer.selection_glyph = Circle(
        radius=circle_radius, fill_color=Spectral4[2]
    )
    graph.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)

    # Set hover glyphs
    graph.node_renderer.hover_glyph = Circle(
        radius=circle_radius, fill_color=Spectral4[1]
    )
    graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

    # Generate glyphs
    graph.node_renderer.glyph = Circle(
        radius=circle_radius,
        fill_color="node_color",
        fill_alpha=alpha,
    )
    graph.edge_renderer.glyph = MultiLine(
        line_color="edge_color",
        line_alpha=alpha,
        line_width="edge_width",
    )

    return graph


def graph_tree(
    G: nx.Graph,
    save_path: str,
    html: bool = False,
):
    """Create a bokeh plot of the Networkx graph.  The plot will feature 5 HTML tabs that provide different highlighting options:

    * Node
    * Edges
    * Nodes + Linked Edges
    * Edges + Linked Nodes
    * Nodes + Adjacent Nodes

    All of the graphs will have custom hover annotations when hovering over specific objects.

    Args:
        G (nx.Graph): Networkx graph.
        save_path (str): Save path of HTML.
        html (bool) = Whether to render HTML.  Defaults to False,
    """

    # We need a RPS game where all nodes are integers
    pos_dict = hierarchy_pos(G, "root")

    # TODO: Add aliasing into the networkx graph!

    tab_plot0 = render_nodes(G, pos_dict)
    tab_plot1 = render_edges(G, pos_dict, tab_plot0)
    tab_plot2 = render_nodes_and_edges(G, pos_dict, tab_plot0)
    tab_plot3 = render_edges_and_linknodes(G, pos_dict, tab_plot0)
    tab_plot4 = render_nodes_and_adjnodes(G, pos_dict, tab_plot0)

    # Create tabs and link them
    tab0 = TabPanel(child=tab_plot0, title="Nodes")
    tab1 = TabPanel(child=tab_plot1, title="Edges")
    tab2 = TabPanel(child=tab_plot2, title="Nodes+Linked Edges")
    tab3 = TabPanel(child=tab_plot3, title="Edges+Linked Nodes")
    tab4 = TabPanel(child=tab_plot4, title="Nodes+Adjacent Nodes")

    # Set save path
    save_dir = os.path.dirname(save_path)
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_file(save_path)

    # Show to user
    if html:
        show(Tabs(tabs=[tab0, tab1, tab2, tab3, tab4], sizing_mode="scale_both"))  # type: ignore
    else:
        save(Tabs(tabs=[tab0, tab1, tab2, tab3, tab4], sizing_mode="scale_both"))  # type: ignore


@ray.remote
def ray_graph_tree(
    G: nx.Graph,
    save_path: str,
    step: int,
    iteration: int,
):
    """Create a bokeh plot of the Networkx graph.  The plot will feature 5 HTML tabs that provide different highlighting options:

    * Node
    * Edges
    * Nodes + Linked Edges
    * Edges + Linked Nodes
    * Nodes + Adjacent Nodes

    All of the graphs will have custom hover annotations when hovering over specific objects.

    Args:
        G (nx.Graph): Networkx graph.
        save_path (str): Save path of HTML.
        step (int): The step (or layer) being updated for this particular plot.
        iteration (int): The current iteration of being updated.
        html (bool) = Flag to display HTML. Defaults to False.
    """

    graph_tree(
        G,
        save_path=f"{save_path}/bokeh/iterations-{iteration}-step-{step}.html",
        html=False,
    )
