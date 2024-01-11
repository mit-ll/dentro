from copy import deepcopy
from pathlib import Path
from typing import Dict
from typing import Literal

import networkx as nx
import numpy as np
import ray

from src.plot.matplotlib import create_plot
from src.utils import rollout


def get_expected_value(G: nx.Graph, target_node: str, id: Literal["red", "blue"]) -> float:
    """This function multiplies the probability distribution edges of one parent node to the expected values of its children nodes.  The output is the expected value for the specified target node.  This is a helper function for `src.cfr.update_node_evs`.

    .. note::
        The expected value for a specific node is simply the probability distribution of the edges (decisions) from a parent node multiplied by the expected value of the children nodes.  This is a recursive operation because each of the children nodes can also be a parent node and have their own edges and children.

    Args:
        G (nx.Graph): Networkx graph.
        target_node (Node): The target node to calculate the expected value.
        id (Literal[red, blue]): Id of node.

    Returns:
        float: expected value
    """

    # If terminal node just return value (do not calculate edge probabilities!)
    if G.nodes[target_node].get("type", False) == "terminal":
        return G.nodes[target_node]["ev"][id]

    # Calculate the parent expected value using edge probabilities and children expected values
    #     o     expected value
    #    /|\    decision probabilities
    #   o o o   expected values
    decision_probs = []
    expected_values = []
    for u, v, d in G.edges(target_node, data=True):
        decision_probs.append(d["s"]["m"] / d["s"]["n"])  # decision probability
        expected_values.append(G.nodes[v]["ev"][id])  # expected value

    # Check data validity
    check1 = np.isclose(sum(decision_probs), 1)
    if check1 is False:
        raise ValueError("Invalid probability detected")

    # Expected value is the sum of decision edges x expected values
    ev = sum(np.array(decision_probs) * np.array(expected_values))

    return ev


def update_node_evs(G: nx.Graph, players: list):
    """
    This function updates every node in the graph starting with the last node (deepest) and recursively updating nodes as it traces to the top (root).  Recursive updating is necessary because nodes higher up rely on the expected values of their children.

    .. note::
        The expected value for a specific node is simply the probability distribution of the edges (decisions) from a parent node multiplied by the expected value of the children nodes.  This is a recursive operation because each of the children nodes can also be a parent node and have their own edges and children.

    Args:
        G (nx.Graph): Networkx graph.
        players (list): A list of player ids (i.e. ["blue", "red"]).
    """

    # Perform reverse recursive iterations
    for node in reversed(list(G.nodes())):
        # Do not update terminal nodes
        if G.nodes[node].get("type", False) == "terminal":
            continue

        # Update non-terminal nodes for each player
        for player in players:
            ev = get_expected_value(G, node, player)
            G.nodes[node]["ev"][player] = ev


@ray.remote
def calc_regret_single(G: nx.Graph, layer_rollout: tuple) -> dict:
    """This function is designed to calculate the regret for a single parent and its edges.  The function expects a single tuple representing an edge (two points connecting parent to child).  This is a helper function for `src.cfr.calc_regret_batch`.

    .. note::
        To calculate the regret of a rollout decision you need to compare it against every other possible decision.  This is done by calculating the delta of the chosen decision's expected value versus all possible decisions' expected values.  The delta is the *regret* of not having choosen the other possible decision.

    Args:
        G (nx.Graph): Networkx graph.
        layer_rollout (tuple): The decisions made at a specific layer.  For example rock-paper-scissors is a two layer game excluding the root.  For rock-paper-scissors you would have a parent with its corresponding edges for a single layer.

    Raises:
        ValueError: Check that edges should be sourced from a common parent.  If this check fails it means there is something wrong with your graph.

    Returns:
        dict: This is a dict containing all regrets for each decision made (per parent -> children).
    """

    #     o       parent node
    #    /|\      edges
    #   o o o     children node expected values
    regret: dict = {}

    # Get the rollout and playerId
    u_chosen, v_chosen = layer_rollout
    playerId = G.edges[(u_chosen, v_chosen)]["player"]

    # Check for expected value
    if G.nodes[v_chosen]["ev"].get(playerId) is None:
        return regret

    ev_chosen = G.nodes[v_chosen]["ev"][playerId]

    # Get all edges
    edges_all = list(nx.bfs_edges(G, u_chosen, depth_limit=1))

    # If the edge does not belong to the player id skip it
    check1 = any([G.edges[edge]["player"] != playerId for edge in edges_all])
    if check1:
        raise ValueError("Edges must be sourced from a common parent!")

    # Get the expected value of the alternative decision
    for edge_alt in edges_all:
        u_alt, v_alt = edge_alt
        ev_alt = G.nodes[v_alt]["ev"][playerId]

        # Here we calculate the regret of not taking the alternative action
        regret[(u_alt, v_alt)] = ev_alt - ev_chosen

    return regret


def calc_regret_batch(G: nx.Graph, layer_rollouts: list) -> dict:
    """
    This function is designed to calculate the regret for a **batch of parent -> edges decisions**.  Because this is a batching operation, it will generate cumulative regrets over many rollouts.  The cumulative regrets are used to update the probability distribution of the agent's decision making for a specific parent -> edge probabilities.

    .. note::
        To calculate the regret of a rollout decision you need to compare it against every other possible decision.  This is done by calculating the delta of the chosen decision's expected value versus all possible decisions' expected values.  The delta is the *regret* of not having choosen the other possible decision.

    Args:
        G (nx.Graph): Networkx graph.
        layer_rollouts (list): A batch of rollouts for a given parent -> edge configuration.

    Returns:
        dict: Sorted dictionary of total regrest.
    """

    # Calculate regret
    total_regrets: dict = {}
    layer_regrets: dict = {}

    # Process the regrets in parallel
    futures = [calc_regret_single.remote(G, layer_rollout) for layer_rollout in layer_rollouts]
    layer_regrets = ray.get(futures)

    # Create a dictionary that holds all regret values based on possible edges
    for layer_regret in layer_regrets:
        for edge, regret_val in layer_regret.items():
            # Initialize the dict
            if total_regrets.get(edge) is None:
                total_regrets[edge] = {"m": 0}
            total_regrets[edge]["m"] += regret_val

    return dict(sorted(total_regrets.items(), key=lambda x: x[0]))


def get_regret_floor_vals(regret: dict) -> dict:
    """Given a dict of regrets of `(u, v), data` pairings, find the minimum regret value of each `u` parent node.  The user can enter one or more `parent -> edge` batches using the input dict.

    Args:
        regret (dict): A dictionary of regrets.  Must follow the convention of `(u, v), data` where:
            * `u` is the parent node
            * `v` is the child node.
            * `data` is the data dict

    Returns:
        dict: A dictionary with the minimum value of each parent node `u` provided from the input dict.
    """

    # Normalize m with floor minimum
    min_val = np.inf
    my_dict: Dict[tuple, int] = {}

    for (u, v), data in regret.items():
        m = data["m"]

        if my_dict.get(u) is None:
            my_dict[u] = min(min_val, m)

        my_dict[u] = min(my_dict[u], m)

    return my_dict


def normalize_m_regret(regret: dict):
    """Normalize all regret values by subtracting the minimum regret (floor value).  This ensures that all regrets are greater than or equal to zero.  This is needed for updating the cumulative regret of CFRM to ensure no negative values are used as updates.

    Args:
        regret (dict): A dictionary of regrets.  Must follow the convention of `(u, v), data` where:
            * `u` is the parent node
            * `v` is the child node.
            * `data` is the data dict
    """
    # Get all floor values for each parent node
    floor_dict: Dict[tuple, int] = get_regret_floor_vals(regret)

    # Determine n from m
    for (u, v), data in regret.items():
        m = data["m"]
        regret[(u, v)]["m"] = m - floor_dict[u]


def normalize_n_regret(regret: dict):
    """To generate a probability distribution, you will need to normalize the total regrets by a common denominiator.  This function calculates the denominator `n` variable needed to normalize across all regrets of a parent node and its edges.

    Args:
        regret (dict): A dictionary of regrets.  Must follow the convention of `(u, v), data` where:
            * `u` is the parent node
            * `v` is the child node.
            * `data` is the data dict
    """
    # Calculate n from summing all m values
    n_total: Dict[tuple, int] = {}

    for (u, v), data in regret.items():
        if n_total.get(u) is None:
            n_total[u] = 0

        n_total[u] += data["m"]

    # Perform assignment of n to regret values
    for (u, v), data in regret.items():
        regret[(u, v)]["n"] = n_total[u]


def update_edge_probs(G: nx.Graph, regret: dict):
    """For all edges in a Graph, update their `m` and `n` values where:
        * `m`: regret of decision
        * `n`: normalization function

    This assumes that `regrets` have been computed for the entire tree and that it only needs to be applied to update the probability distributions.  This function should only be called after the `regret` calculations have completed.

    Args:
        G (nx.Graph): Networkx graph.
        regret (dict): A dictionary of all the regrets of the graph.
    """
    # Exit if empty
    if len(regret) == 0:
        return

    normalized_regret = deepcopy(regret)

    normalize_m_regret(normalized_regret)
    normalize_n_regret(normalized_regret)

    # Update graph probabilities through m/n
    for edge in G.edges():
        if regret.get(edge) is None:
            continue

        G.edges[edge]["s"]["m"] += normalized_regret[edge]["m"]
        G.edges[edge]["s"]["n"] += normalized_regret[edge]["n"]


def run_cfr(
    G: nx.Graph,
    players: list[str],
    n_iterations: int = 1,
    n_rollouts: int = 100,
    save_path: str = "save/",
    graph_id: str = "default.networkx",
    fig_x_size: int = 18,
    fig_y_size: int = 9,
):
    """Runs the full CFRM algorithm by sequentially performing the following operations:

        1. Generating rollout trajectories
        2. Recursively perform operations starting from the bottom layer of the tree:
            * calculate the regret of decisions made
            * update the edge probabilities
            * update expected values of nodes based on new decision probabilties
            * generating plots
            * adding labels to plots
        3. Save to XML file

    Args:
        G (nx.Graph): The networkx graph.
        players (list[str]): List of player to iterate over.
        n_iterations (int, optional): Number of iterations to run the algorithm.  More difficult games will need higher iterations to converge. Defaults to 1.
        n_rollouts (int, optional): The number of rollouts to perform on each iteration.  Higher is better but uses more computing resources. Defaults to 100.
        save_path (str, optional): The path to save the files to. Defaults to "save/".
        graph_id (str, optional): Name of folder to put graphs in. Defaults to "default.networkx".
        fig_x_size (int, optional): The figure size of the plots (x-dimension). Defaults to 18.
        fig_y_size (int, optional): The figure size of the plots (y-dimension). Defaults to 9.
    """
    # Initialize variables
    futures: list = []

    # Initialize paths
    graph_dir = f"{save_path}/graphs"
    graph_path = f"{graph_dir}/{graph_id}"
    plots_dir = f"{save_path}/plots"

    Path(graph_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    for iteration in range(0, n_iterations):
        futures = [rollout.remote(G, "root") for _ in range(n_rollouts)]
        rollouts = ray.get(futures)

        # Here we need to compute regret by layer (deepest first -> root)
        rollout_lengths = max(list(map(len, rollouts)))

        for step, layer in enumerate(range(rollout_lengths - 1, 0, -1)):
            # Filter the rollouts based on the layer
            layer_rollouts = list(map(lambda x: x[layer], rollouts))
            layer_regrets = calc_regret_batch(G, layer_rollouts)

            # Update decision probabilities and expected values
            update_edge_probs(G, layer_regrets)
            update_node_evs(G, players)

            # Show graph as image
            futures.append(
                create_plot.remote(
                    G, plots_dir, fig_x_size, fig_y_size, layer_rollouts, step, iteration
                )
            )

            # Update the label for each edge (this is for debugging purposes)
            for u, v, data in G.edges(data=True):
                edge = (u, v)
                G.edges[edge]["s"]["label"] = round(data["s"]["m"] / (data["s"]["n"]) * 100)

    # Wait for all tasks to complete
    ray.get(futures)

    # Save off learned graph weights
    nx.gml.write_gml(G, graph_path)
