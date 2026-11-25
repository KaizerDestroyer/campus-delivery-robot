"""Topic 3 -- Sequential decision making: an MDP solved with value iteration.

When the robot stands in front of a cell that may be blocked it can

    Move     -- step forward (if the cell is actually blocked: collision),
    Wait     -- stay one tick, then read the sensor again,
    Reroute  -- stop, re-plan with A* and take the detour (extra steps).

The robot never sees the true state, only its Bayes belief b = P(blocked).
So the MDP is built over belief states (a "belief MDP"):

    S      = {b = 0.00, 0.01, ..., 1.00} + terminal "Passed"
    A      = {Move, Wait, Reroute}
    P      = Bayes filter: after one tick the obstacle may change (time update)
             and the sensor beeps or stays silent (measurement update),
             which moves the belief to a new value
    R      = -1 per tick, -collision penalty if we bump into an obstacle,
             -(1 + extra + replan) for the detour
    gamma  = 0.95

Value iteration repeats the Bellman optimality update until it converges:

    V(s) <- max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) * V(s') ]

Beliefs that fall between grid points are split between the two nearest
points (linear interpolation), so the belief MDP is an ordinary finite MDP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import floor
from typing import Hashable

from .bayes import ObstacleDynamics, SensorModel, bayes_update, predict

PASSED = "Passed"
MOVE, WAIT, REROUTE = "Move", "Wait", "Reroute"

State = Hashable
Action = str
Outcomes = list[tuple[float, State]]


# ---------------------------------------------------------------------------
# Generic finite MDP + value iteration
# ---------------------------------------------------------------------------

@dataclass
class MDP:
    """A finite MDP. Terminal states have value 0 and no actions."""

    states: list[State]
    actions: dict[State, list[Action]]
    transitions: dict[tuple[State, Action], Outcomes]
    rewards: dict[tuple[State, Action], float]
    terminals: frozenset[State]
    gamma: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1) so value iteration converges")
        for key, outcomes in self.transitions.items():
            total = sum(p for p, _ in outcomes)
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"transition probabilities for {key} sum to {total}")


@dataclass
class VIResult:
    values: dict[State, float]
    q: dict[State, dict[Action, float]]
    policy: dict[State, Action]
    iterations: int
    deltas: list[float] = field(default_factory=list)  # max |V change| per sweep


def q_value(mdp: MDP, values: dict[State, float], s: State, a: Action) -> float:
    """One-step look-ahead: R(s,a) + gamma * E[V(s')]."""
    future = sum(p * values[s2] for p, s2 in mdp.transitions[(s, a)])
    return mdp.rewards[(s, a)] + mdp.gamma * future


def value_iteration(mdp: MDP, tol: float = 1e-6, max_iter: int = 10_000) -> VIResult:
    """Solve the MDP with synchronous Bellman optimality updates."""
    values = {s: 0.0 for s in mdp.states}
    deltas: list[float] = []
    iterations = 0
    for iterations in range(1, max_iter + 1):
        new_values = {
            s: 0.0 if s in mdp.terminals
            else max(q_value(mdp, values, s, a) for a in mdp.actions[s])
            for s in mdp.states
        }
        delta = max(abs(new_values[s] - values[s]) for s in mdp.states)
        values = new_values
        deltas.append(delta)
        if delta < tol:
            break
    q = {s: {a: q_value(mdp, values, s, a) for a in mdp.actions[s]}
         for s in mdp.states if s not in mdp.terminals}
    policy = {s: max(qs, key=qs.get) for s, qs in q.items()}
    return VIResult(values, q, policy, iterations, deltas)


# ---------------------------------------------------------------------------
# The robot's decision problem: Move / Wait / Reroute in front of a cell
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Costs:
    """Reward model, written as costs (rewards are the negatives)."""

    step: float = 1.0        # every tick (moving or waiting) costs time
    collision: float = 50.0  # extra penalty for bumping into an obstacle (or a person)
    replan: float = 1.0      # one tick spent stopping and re-planning a detour


@dataclass(frozen=True)
class Encounter:
    """Everything the MDP needs to know about one suspicious cell."""

    dynamics: ObstacleDynamics
    sensor: SensorModel = SensorModel()
    costs: Costs = Costs()
    detour_extra: int | None = None   # extra steps of the A* detour; None = no detour
    gamma: float = 0.95
    resolution: int = 100             # belief grid: 0.00, 0.01, ..., 1.00

    @property
    def actions(self) -> list[Action]:
        return [MOVE, WAIT] + ([REROUTE] if self.detour_extra is not None else [])


def _spread(belief: float, n: int) -> Outcomes:
    """Split a belief between its two nearest grid points (linear interpolation)."""
    x = min(max(belief, 0.0), 1.0) * n
    lo = min(floor(x), n)
    w_hi = x - lo
    if w_hi < 1e-12:
        return [(1.0, lo)]
    return [(1.0 - w_hi, lo), (w_hi, lo + 1)]


def _merge(outcomes: Outcomes) -> Outcomes:
    merged: dict[State, float] = {}
    for p, s in outcomes:
        if p > 0.0:
            merged[s] = merged.get(s, 0.0) + p
    return [(p, s) for s, p in merged.items()]


def _next_beliefs(belief: float, enc: Encounter) -> Outcomes:
    """Belief one tick later, after the next sensor reading (Bayes filter step)."""
    prior = predict(belief, enc.dynamics)                       # time update
    out: Outcomes = []
    for alert in (True, False):
        p_obs = (enc.sensor.likelihood(alert, True) * prior
                 + enc.sensor.likelihood(alert, False) * (1.0 - prior))
        if p_obs > 0.0:
            posterior = bayes_update(prior, alert, enc.sensor)  # measurement update
            out += [(p_obs * w, s) for w, s in _spread(posterior, enc.resolution)]
    return _merge(out)


def _model(belief: float, action: Action, enc: Encounter) -> tuple[float, Outcomes]:
    """Expected reward R(b, a) and transition list P(. | b, a)."""
    c = enc.costs
    if action == MOVE:
        # Clear with prob 1-b: we pass. Blocked with prob b: we bump and stay,
        # and now know the cell was blocked (belief 1 before the next tick).
        reward = -(c.step + belief * c.collision)
        after_bump = [(belief * p, s) for p, s in _next_beliefs(1.0, enc)]
        return reward, _merge([(1.0 - belief, PASSED)] + after_bump)
    if action == WAIT:
        return -c.step, _next_beliefs(belief, enc)
    if action == REROUTE and enc.detour_extra is not None:
        return -(c.step * (1 + enc.detour_extra) + c.replan), [(1.0, PASSED)]
    raise ValueError(f"action {action!r} is not available")


def build_belief_mdp(enc: Encounter) -> MDP:
    """Finite MDP over the belief grid; transitions come from the Bayes filter."""
    n = enc.resolution
    grid_states = list(range(n + 1))      # state i means belief i / n
    transitions, rewards = {}, {}
    for i in grid_states:
        for a in enc.actions:
            rewards[(i, a)], transitions[(i, a)] = _model(i / n, a, enc)
    return MDP(
        states=grid_states + [PASSED],
        actions={i: enc.actions for i in grid_states},
        transitions=transitions,
        rewards=rewards,
        terminals=frozenset({PASSED}),
        gamma=enc.gamma,
    )


@lru_cache(maxsize=256)
def solve(enc: Encounter) -> VIResult:
    """Value iteration on the belief MDP (cached: each encounter type is solved once)."""
    return value_iteration(build_belief_mdp(enc))


def action_values(belief: float, enc: Encounter) -> dict[Action, float]:
    """Q(b, a) for any belief b, using the converged value function."""
    values = solve(enc).values
    scores = {}
    for a in enc.actions:
        reward, outcomes = _model(belief, a, enc)
        scores[a] = reward + enc.gamma * sum(p * values[s] for p, s in outcomes)
    return scores


def decide(belief: float, enc: Encounter) -> tuple[Action, dict[Action, float]]:
    """Best action for the current belief. Ties go to Move, then Wait."""
    scores = action_values(belief, enc)
    return max(scores, key=scores.get), scores


def move_threshold(enc: Encounter, steps: int = 1000) -> float:
    """Smallest belief at which the optimal action is no longer Move."""
    for k in range(steps + 1):
        if decide(k / steps, enc)[0] != MOVE:
            return k / steps
    return 1.0
