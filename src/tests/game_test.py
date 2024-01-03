import numpy as np
import ray
from rich.traceback import install

from src.cfr import calc_regret_batch
from src.cfr import run_cfr
from src.cfr import update_node_evs
from src.games import rock_paper_scissors


install(show_locals=True)


def test_setting_expected_values():
    """Test whether terminal EV values are properly propagated into the parent nodes."""

    ray.init(ignore_reinit_error=True)
    G = rock_paper_scissors()
    players = ["blue", "red"]

    # Blue update
    G.nodes["T1"]["ev"]["blue"] = 100
    G.nodes["T1"]["ev"]["red"] = 100

    # Update EV for each node
    update_node_evs(G, players)

    # Check the EV values are valid
    assert np.isclose(G.nodes["B1"]["ev"]["blue"], 1 / 3 * 100), "Invalid EV update"
    assert np.isclose(G.nodes["R1"]["ev"]["red"], 1 / 3 * 1 / 5 * 100), "Invalid EV update"


def test_calc_regret_batch():
    """Verify that the regrets are consistent with a sample rollout."""

    ray.init(ignore_reinit_error=True)
    G = rock_paper_scissors()

    # A sample rollout
    rollouts = [
        [("root", "R1"), ("R1", "B3"), ("B3", "T9")],
        [("root", "R1"), ("R1", "B2"), ("B2", "T5")],
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
    """Run the full Rock-Paper-Scissors experiment using CFRM.  This is integration testing exercising the entire pipeline.  The designer will have to verify whether the algorithm is performing as intended by manually inspecting the plots."""

    ray.init(ignore_reinit_error=True)
    G = rock_paper_scissors()

    run_cfr(
        G,
        players=["red", "blue"],
        n_iterations=10,
        n_rollouts=1000,
        save_path="save/rock-paper-scissors",
        graph_id="rps.networkx",
        fig_x_size=14,
        fig_y_size=9,
    )
