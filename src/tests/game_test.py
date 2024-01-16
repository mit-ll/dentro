import numpy as np
import ray
from rich.traceback import install

from src.cfr import calc_regret_batch
from src.cfr import run_cfr
from src.cfr import update_node_evs
from src.game.rps import rock_paper_scissors


install(show_locals=True)


def test_setting_expected_values():
    """Test whether terminal EV values are properly propagated into the parent nodes.  Here we assign a terminal value to a single terminal node.  All other terminal nodes are zeroed.  This means that the probabilities should be the same as our assigned terminal value multiplied by the probabilities."""

    ray.init(ignore_reinit_error=True)

    # Initialize the game
    G = rock_paper_scissors()
    players = ["blue", "red"]

    # Assign a terminal value
    for _, data in G.nodes(data=True):
        data["ev"]["blue"] = 0
        data["ev"]["red"] = 0

    G.nodes["T1"]["ev"]["blue"] = 100
    G.nodes["T1"]["ev"]["red"] = 100

    # Update EV for each node
    update_node_evs(G, players)

    # Check the EV values are valid
    edge_B1_T1 = G.edges[("B1", "T1")]  # probability assigned to this path
    edge_R1_B1 = G.edges[("R1", "B1")]  # probability assigned to this path

    # Get the probabilities
    prob_T1_B1 = edge_B1_T1["s"]["m"] / edge_B1_T1["s"]["n"]
    prob_B1_R1 = edge_R1_B1["s"]["m"] / edge_R1_B1["s"]["n"]

    # Verify that exepcted values are correct
    assert np.isclose(
        G.nodes["B1"]["ev"]["blue"],
        prob_T1_B1 * 100,
    ), "Invalid EV update"

    assert np.isclose(
        G.nodes["R1"]["ev"]["red"],
        prob_T1_B1 * prob_B1_R1 * 100,
    ), "Invalid EV update"


def test_calc_regret_batch():
    """Verify that the regrets are consistent with a sample rollout.  In this test we create two rollouts.  In one rollout both player played scissors.  In the second rollout both players played paper.

    From red's perspective, there is no regret because all EVs are zero.  From blue's perspective, it regrets not playing the move that would of countered red's action. The test will check that `m` in the numerator of the regret is incremented up.
    """

    ray.init(ignore_reinit_error=True)
    G = rock_paper_scissors()

    # A sample rollout
    rollouts = [
        [("root", "R1"), ("R1", "B3"), ("B3", "T9")],  # both played scissors
        [("root", "R1"), ("R1", "B2"), ("B2", "T5")],  # both played paper
    ]

    rollout_lengths = max(list(map(len, rollouts)))

    for layer in range(rollout_lengths - 1, 0, -1):
        # Filter the rollouts based on the layer
        layer_rollouts = list(map(lambda x: x[layer], rollouts))
        regrets = calc_regret_batch(G, layer_rollouts)

        # Red has zero regret because all EV are zero
        if layer == 1:
            assert regrets[("R1", "B1")]["m"] == 0
            assert regrets[("R1", "B2")]["m"] == 0
            assert regrets[("R1", "B3")]["m"] == 0

        # Blue has regrets are as follows:
        # Rollout 1: red played scissors, blue played scissors (regret not playing rock )
        # Rollout 2: red played paper, blue played paper (regret not playing scissors)
        # Note: We expect that the m values should increment playing rock + scissors!
        if layer == 2:
            # Rollout 1: red played scissors
            assert regrets[("B3", "T7")]["m"] == 1  # regret playing rock
            assert regrets[("B3", "T8")]["m"] == -1  # regret playing paper
            assert regrets[("B3", "T9")]["m"] == 0  # regret playing scissors

            # Rollout 2: red played paper
            assert regrets[("B2", "T4")]["m"] == -1  # regret playing rock
            assert regrets[("B2", "T5")]["m"] == 0  # regret playing paper
            assert regrets[("B2", "T6")]["m"] == 1  # regret playing rock


def test_rock_paper_scissors():
    """Run the full Rock-Paper-Scissors experiment using CFRM.  This is integration testing exercising the entire pipeline.  Since Ray is non-determinisitic the checks here are bounded by a range.  It is not possible to exactly replicate the rollouts every single time."""

    ray.init(ignore_reinit_error=True)
    G = rock_paper_scissors()

    run_cfr(
        G,
        players=["red", "blue"],
        n_iterations=10,
        n_rollouts=1000,
        save_path="save/rock-paper-scissors",
        fig_x_size=14,
        fig_y_size=9,
    )

    # Blue probs
    edge_B1_T1 = G.edges[("B1", "T1")]
    edge_B1_T2 = G.edges[("B1", "T2")]
    edge_B1_T3 = G.edges[("B1", "T3")]
    edge_B2_T4 = G.edges[("B2", "T4")]
    edge_B2_T5 = G.edges[("B2", "T5")]
    edge_B2_T6 = G.edges[("B2", "T6")]
    edge_B3_T7 = G.edges[("B3", "T7")]
    edge_B3_T8 = G.edges[("B3", "T8")]
    edge_B3_T9 = G.edges[("B3", "T9")]

    # Red probs
    edge_R1_B1 = G.edges[("R1", "B1")]
    edge_R1_B2 = G.edges[("R1", "B2")]
    edge_R1_B3 = G.edges[("R1", "B3")]

    # Checks to ensure probs are within reasonable ranges.  Blue will have converged to the Nash equilibrium at this point.  Red has converged to a correlated equilibrium where it plays paper more than other moves.
    check1 = 0.35 > (edge_B1_T1["s"]["m"] / edge_B1_T1["s"]["n"]) > 0.31
    check2 = 0.35 > (edge_B1_T2["s"]["m"] / edge_B1_T2["s"]["n"]) > 0.31
    check3 = 0.35 > (edge_B1_T3["s"]["m"] / edge_B1_T3["s"]["n"]) > 0.31
    check4 = 0.35 > (edge_B2_T4["s"]["m"] / edge_B2_T4["s"]["n"]) > 0.31
    check5 = 0.35 > (edge_B2_T5["s"]["m"] / edge_B2_T5["s"]["n"]) > 0.31
    check6 = 0.35 > (edge_B2_T6["s"]["m"] / edge_B2_T6["s"]["n"]) > 0.31
    check7 = 0.35 > (edge_B3_T7["s"]["m"] / edge_B3_T7["s"]["n"]) > 0.31
    check8 = 0.35 > (edge_B3_T8["s"]["m"] / edge_B3_T8["s"]["n"]) > 0.31
    check9 = 0.35 > (edge_B3_T9["s"]["m"] / edge_B3_T9["s"]["n"]) > 0.31

    check10 = 0.30 > (edge_R1_B1["s"]["m"] / edge_R1_B1["s"]["n"]) > 0.25
    check11 = 0.47 > (edge_R1_B2["s"]["m"] / edge_R1_B2["s"]["n"]) > 0.43
    check12 = 0.30 > (edge_R1_B3["s"]["m"] / edge_R1_B2["s"]["n"]) > 0.25

    assert (
        check1
        and check2
        and check3
        and check4
        and check5
        and check6
        and check7
        and check8
        and check9
        and check10
        and check11
        and check12
    ), "Invalid edge probabilities!"
