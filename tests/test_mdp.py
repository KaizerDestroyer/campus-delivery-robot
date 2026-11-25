"""Tests for topic 3: MDP + value iteration."""

import pytest

from robot.bayes import ObstacleDynamics, SensorModel
from robot.mdp import (MDP, MOVE, PASSED, REROUTE, WAIT, Costs, Encounter,
                       build_belief_mdp, decide, move_threshold, solve, value_iteration)

HOTSPOT = ObstacleDynamics.from_risk(0.3)


def test_value_iteration_on_a_textbook_chain():
    """A -> B -> End with reward -1 per step: V(A) = -1 - gamma, V(B) = -1."""
    mdp = MDP(
        states=["A", "B", "End"],
        actions={"A": ["go"], "B": ["go"]},
        transitions={("A", "go"): [(1.0, "B")], ("B", "go"): [(1.0, "End")]},
        rewards={("A", "go"): -1.0, ("B", "go"): -1.0},
        terminals=frozenset({"End"}),
        gamma=0.9,
    )
    r = value_iteration(mdp)
    assert r.values["B"] == pytest.approx(-1.0)
    assert r.values["A"] == pytest.approx(-1.9)
    assert r.values["End"] == 0.0


def test_value_iteration_picks_the_better_action():
    """Safe: -2 then done. Risky: 50% done for -1, 50% stuck for -10. Safe is better."""
    mdp = MDP(
        states=["S", "Done"],
        actions={"S": ["safe", "risky"]},
        transitions={("S", "safe"): [(1.0, "Done")],
                     ("S", "risky"): [(0.5, "Done"), (0.5, "S")]},
        rewards={("S", "safe"): -2.0, ("S", "risky"): -5.5},
        terminals=frozenset({"Done"}),
        gamma=0.9,
    )
    assert value_iteration(mdp).policy["S"] == "safe"


def test_invalid_mdp_is_rejected():
    with pytest.raises(ValueError):
        MDP(["S"], {"S": ["a"]}, {("S", "a"): [(0.7, "S")]}, {("S", "a"): 0.0},
            frozenset(), gamma=0.9)
    with pytest.raises(ValueError):
        MDP(["S"], {}, {}, {}, frozenset(), gamma=1.0)


def test_belief_mdp_is_a_proper_mdp():
    mdp = build_belief_mdp(Encounter(HOTSPOT, detour_extra=2))
    assert len(mdp.states) == 102                # 101 belief levels + Passed
    for outcomes in mdp.transitions.values():    # __post_init__ checks sums too
        assert sum(p for p, _ in outcomes) == pytest.approx(1.0)


def test_value_iteration_converges():
    r = solve(Encounter(HOTSPOT))
    assert r.deltas[-1] < 1e-6
    assert r.iterations < 1000
    assert r.values[PASSED] == 0.0


def test_decisions_make_sense():
    enc = Encounter(HOTSPOT)                       # no detour available
    assert decide(0.0, enc)[0] == MOVE             # surely clear -> go
    assert decide(0.659, enc)[0] == WAIT           # after an alert -> wait
    assert decide(1.0, enc)[0] == WAIT             # surely blocked -> never move


def test_cheap_detour_beats_waiting_expensive_detour_does_not():
    assert decide(0.659, Encounter(HOTSPOT, detour_extra=0))[0] == REROUTE
    assert decide(0.659, Encounter(HOTSPOT, detour_extra=10))[0] == WAIT


def test_higher_collision_penalty_makes_robot_more_careful():
    careless = move_threshold(Encounter(HOTSPOT, costs=Costs(collision=5)))
    careful = move_threshold(Encounter(HOTSPOT, costs=Costs(collision=100)))
    assert careful < careless


def test_q_values_order_at_high_risk():
    _, q = decide(0.9, Encounter(HOTSPOT, SensorModel(), detour_extra=4))
    assert q[MOVE] < q[WAIT]
    assert q[MOVE] < q[REROUTE]
