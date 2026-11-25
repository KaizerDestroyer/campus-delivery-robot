"""Integration tests: the three algorithms working together."""

import pytest

from robot.grid import Grid, load_map, random_grid
from robot.search import astar
from robot.simulation import POLICIES, SimConfig, evaluate, run_episode

CAMPUS = load_map("Campus 9x12")


@pytest.mark.parametrize("policy", list(POLICIES))
def test_every_policy_delivers(policy):
    for seed in range(30):
        ep = run_episode(CAMPUS, policy, seed)
        assert ep.delivered
        assert ep.ticks >= astar(CAMPUS).cost        # can't beat the shortest path


def test_same_seed_same_result():
    a = run_episode(CAMPUS, "bayes_mdp", 11)
    b = run_episode(CAMPUS, "bayes_mdp", 11)
    assert (a.ticks, a.collisions, a.waits, a.reroutes) == (b.ticks, b.collisions,
                                                            b.waits, b.reroutes)


def test_log_is_consistent():
    ep = run_episode(CAMPUS, "bayes_mdp", 5)
    moves = [s for s in ep.log if s.outcome == "moved"]
    assert moves[-1].target == CAMPUS.goal
    assert sum(s.outcome == "collision" for s in ep.log) == ep.collisions
    assert sum(s.outcome == "waited" for s in ep.log) == ep.waits
    for s in ep.log:
        assert 0.0 <= s.prior <= 1.0 and 0.0 <= s.posterior <= 1.0
        assert s.posterior > s.prior if s.alert else s.posterior <= s.prior


def test_full_system_is_safer_and_cheaper_than_naive():
    naive = evaluate(CAMPUS, "naive", episodes=150)
    full = evaluate(CAMPUS, "bayes_mdp", episodes=150)
    assert full["avg_collisions"] < 0.25 * naive["avg_collisions"]
    assert full["avg_cost"] < naive["avg_cost"]


def test_full_system_is_never_much_worse_than_any_fixed_rule():
    results = {p: evaluate(CAMPUS, p, episodes=200)["avg_cost"] for p in POLICIES}
    assert results["bayes_mdp"] <= 1.05 * min(results.values())


@pytest.mark.parametrize("seed", range(25))
def test_stability_on_random_maps(seed):
    grid = random_grid(10, 10, wall_prob=0.2, seed=seed, hotspot_prob=0.15)
    if not astar(grid).found:
        with pytest.raises(ValueError):
            run_episode(grid, "bayes_mdp", seed)
        return
    ep = run_episode(grid, "bayes_mdp", seed, SimConfig(max_ticks=300))
    assert ep.delivered


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        run_episode(CAMPUS, "teleport", 0)


@pytest.mark.parametrize("text", ["", "S . .\n. .", "S . X G", "S . . ."])
def test_bad_maps_are_rejected(text):
    with pytest.raises(ValueError):
        Grid.from_ascii(text)
