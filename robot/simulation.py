"""Putting it together: A* plans, the Bayes filter tracks, the MDP decides.

Each tick the robot
  1. looks at the next cell of its A* route and reads the noisy sensor,
  2. updates P(blocked) for that cell with Bayes' rule,
  3. asks A* how long a detour around the cell would be,
  4. picks Move / Wait / Reroute with the policy under test,
  5. acts in a hidden "true" world where obstacles appear and disappear.

Four policies are compared (an ablation study):
  naive      -- A* only, ignores the sensor and always moves
  reactive   -- trusts the raw sensor: alert -> Wait, otherwise Move
  bayes      -- Bayes belief + fixed rule: Wait if P(blocked) > 0.5
  bayes_mdp  -- full system: Bayes belief + belief-MDP (value iteration)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .bayes import ObstacleDynamics, SensorModel, bayes_update, predict
from .grid import Cell, Grid
from .mdp import MOVE, REROUTE, WAIT, Costs, Encounter, decide
from .search import astar

POLICIES = {
    "naive": "A* only (ignore sensor)",
    "reactive": "Raw sensor (wait on alert)",
    "bayes": "Bayes + 0.5 threshold",
    "bayes_mdp": "Bayes + MDP (full system)",
}


@dataclass(frozen=True)
class SimConfig:
    sensor: SensorModel = SensorModel()
    costs: Costs = Costs()
    gamma: float = 0.95
    mixing: float = 0.5          # how quickly obstacles change between ticks
    max_ticks: int = 200


@dataclass
class StepLog:
    tick: int
    position: Cell
    target: Cell                 # next cell on the route that was sensed
    truly_blocked: bool          # hidden ground truth (robot does not see it)
    alert: bool
    prior: float
    posterior: float
    detour_extra: int | None
    scores: dict[str, float]
    action: str
    outcome: str                 # moved | collision | waited | rerouted
    route: list[Cell]            # remaining planned route after acting


@dataclass
class EpisodeResult:
    policy: str
    delivered: bool
    ticks: int
    collisions: int
    waits: int
    reroutes: int
    total_cost: float            # ticks + collision penalty * collisions
    initial_route: list[Cell]
    log: list[StepLog] = field(default_factory=list)


class World:
    """Hidden ground truth: which cells are blocked right now."""

    def __init__(self, grid: Grid, rng: random.Random, mixing: float) -> None:
        self.rng = rng
        self.dynamics = {c: ObstacleDynamics.from_risk(grid.risk(c), mixing)
                         for c in grid.free_cells()}
        self.blocked = {c: rng.random() < grid.risk(c) for c in grid.free_cells()}

    def tick(self) -> None:
        for cell, dyn in self.dynamics.items():
            p_change = dyn.p_clear if self.blocked[cell] else dyn.p_block
            if self.rng.random() < p_change:
                self.blocked[cell] = not self.blocked[cell]


def _choose(policy: str, alert: bool, belief: float, dynamics: ObstacleDynamics,
            detour_extra: int | None, cfg: SimConfig) -> tuple[str, dict[str, float]]:
    if policy == "naive":
        return MOVE, {}
    if policy == "reactive":
        return (WAIT if alert else MOVE), {}
    if policy == "bayes":
        return (WAIT if belief > 0.5 else MOVE), {}
    if policy == "bayes_mdp":
        return decide(belief, Encounter(dynamics, cfg.sensor, cfg.costs, detour_extra,
                                        cfg.gamma))
    raise ValueError(f"unknown policy {policy!r}; choose from {list(POLICIES)}")


def run_episode(grid: Grid, policy: str = "bayes_mdp", seed: int = 0,
                cfg: SimConfig = SimConfig()) -> EpisodeResult:
    """Simulate one delivery. Same seed => same world, so policies are comparable."""
    world = World(grid, random.Random(seed), cfg.mixing)
    sensor_rng = random.Random(seed + 1_000_003)  # separate stream for sensor noise

    plan = astar(grid)
    if not plan.found:
        raise ValueError("goal is unreachable on this map")
    route: list[Cell] = plan.path
    pos = grid.start
    beliefs: dict[Cell, float] = {}   # tracked beliefs for cells we have sensed
    avoid: set[Cell] = set()          # cells we decided to route around
    log: list[StepLog] = []
    ticks = collisions = waits = reroutes = 0

    while pos != grid.goal and ticks < cfg.max_ticks:
        target = route[1]
        dyn = world.dynamics[target]

        # 1-2. Sense the next cell and update the belief with Bayes' rule.
        truth = world.blocked[target]
        p_alert = cfg.sensor.sensitivity if truth else cfg.sensor.false_alarm
        alert = sensor_rng.random() < p_alert
        prior = beliefs.get(target, grid.risk(target))
        posterior = bayes_update(prior, alert, cfg.sensor)
        beliefs[target] = posterior

        # 3. How much longer is the A* detour that avoids this cell?
        detour = astar(grid, pos, grid.goal, blocked=avoid | {target})
        remaining = len(route) - 1
        extra = int(detour.cost - remaining) if detour.found else None

        # 4. Decide.
        action, scores = _choose(policy, alert, posterior, dyn, extra, cfg)

        # 5. Act in the true world.
        here = pos
        if action == REROUTE:
            avoid.add(target)
            route = detour.path
            reroutes += 1
            outcome = "rerouted"           # stopping to re-plan takes one tick
        elif action == WAIT:
            waits += 1
            outcome = "waited"
        elif truth:                        # MOVE into a blocked cell
            collisions += 1
            beliefs[target] = 1.0          # now we know it is blocked
            outcome = "collision"
        else:                              # MOVE into a clear cell
            pos = target
            route = route[1:]
            outcome = "moved"

        log.append(StepLog(ticks, here, target, truth,
                           alert, prior, posterior, extra, scores, action, outcome,
                           list(route)))

        # Time passes: the world changes and our beliefs drift (Bayes time update).
        world.tick()
        ticks += 1
        beliefs = {c: predict(b, world.dynamics[c]) for c, b in beliefs.items()}

    total = ticks + cfg.costs.collision * collisions
    return EpisodeResult(policy, pos == grid.goal, ticks, collisions, waits, reroutes,
                         total, plan.path, log)


def evaluate(grid: Grid, policy: str, episodes: int = 300, cfg: SimConfig = SimConfig(),
             seed0: int = 0) -> dict[str, float]:
    """Average metrics of a policy over many seeded episodes."""
    results = [run_episode(grid, policy, seed0 + i, cfg) for i in range(episodes)]
    n = len(results)
    return {
        "policy": policy,
        "success_rate": sum(r.delivered for r in results) / n,
        "avg_ticks": sum(r.ticks for r in results) / n,
        "avg_collisions": sum(r.collisions for r in results) / n,
        "collision_free_rate": sum(r.collisions == 0 for r in results) / n,
        "avg_waits": sum(r.waits for r in results) / n,
        "avg_reroutes": sum(r.reroutes for r in results) / n,
        "avg_cost": sum(r.total_cost for r in results) / n,
    }
