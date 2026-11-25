"""Reproduce every number in the report:  python experiments/run_experiments.py

Writes results/RESULTS.md (tables) and results/results.json (raw numbers).
All experiments are seeded, so re-running gives identical results.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robot import (ALGORITHMS, MAPS, POLICIES, Costs, Encounter, ObstacleDynamics,  # noqa: E402
                   SensorModel, SimConfig, bayes_update, decide, evaluate, filter_readings,
                   load_map, move_threshold)
from robot.grid import random_grid  # noqa: E402
from robot.mdp import solve  # noqa: E402

EPISODES = 1000
out: list[str] = []
raw: dict = {}


def emit(line: str = "") -> None:
    print(line)
    out.append(line)


def table(headers: list[str], rows: list[list]) -> None:
    emit("| " + " | ".join(headers) + " |")
    emit("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        emit("| " + " | ".join(str(x) for x in row) + " |")
    emit()


def exp_search() -> None:
    emit("## E1. Search: A* vs UCS vs BFS vs Greedy\n")
    rows = []
    for name in MAPS:
        grid = load_map(name)
        for alg, fn in ALGORITHMS.items():
            r = fn(grid)
            rows.append([name, alg, r.cost, r.expanded])
    table(["Map", "Algorithm", "Path length", "Nodes expanded"], rows)

    n_maps, stats = 0, {alg: {"expanded": 0, "optimal": 0} for alg in ALGORITHMS}
    for seed in range(2000):
        grid = random_grid(15, 15, wall_prob=0.3, seed=seed, hotspot_prob=0.0)
        best = ALGORITHMS["BFS"](grid)
        if not best.found:
            continue
        n_maps += 1
        for alg, fn in ALGORITHMS.items():
            r = fn(grid)
            stats[alg]["expanded"] += r.expanded
            stats[alg]["optimal"] += r.cost == best.cost
        if n_maps == 500:
            break
    rows = [[alg, f"{s['expanded'] / n_maps:.1f}", f"{100 * s['optimal'] / n_maps:.1f}%"]
            for alg, s in stats.items()]
    emit(f"Random 15x15 maps with 30% walls ({n_maps} solvable maps):\n")
    table(["Algorithm", "Avg nodes expanded", "Optimal path found"], rows)
    raw["search_random"] = {"maps": n_maps, **stats}


def exp_bayes() -> None:
    emit("## E2. Bayes filter: formula vs Monte-Carlo frequency\n")
    sensor = SensorModel()
    rng = random.Random(7)
    rows = []
    for prior in (0.30, 0.02):
        for alert in (True, False):
            hits = matches = 0
            for _ in range(200_000):
                blocked = rng.random() < prior
                a = rng.random() < (sensor.sensitivity if blocked else sensor.false_alarm)
                if a == alert:
                    matches += 1
                    hits += blocked
            rows.append([prior, "ALERT" if alert else "quiet",
                         f"{bayes_update(prior, alert, sensor):.4f}", f"{hits / matches:.4f}"])
    table(["Prior", "Reading", "Bayes formula", "Simulated frequency (200k)"], rows)
    seq = [True, True, False, True]
    beliefs = filter_readings(0.30, seq, sensor)
    emit("Sequential readings at a hotspot (prior 0.30):\n")
    table(["Reading #", "Reading", "P(blocked)"],
          [[i + 1, "ALERT" if a else "quiet", f"{b:.3f}"] for i, (a, b) in
           enumerate(zip(seq, beliefs))])


def exp_mdp() -> None:
    emit("## E3. MDP: value iteration and the resulting policy\n")
    rows = []
    for label, risk in (("hotspot", 0.30), ("ordinary", 0.02)):
        dyn = ObstacleDynamics.from_risk(risk)
        for extra in (None, 0, 2, 4, 6):
            enc = Encounter(dyn, detour_extra=extra)
            res = solve(enc)
            action, _ = decide(0.659, enc)
            rows.append([label, "none" if extra is None else f"+{extra}", res.iterations,
                         f"{move_threshold(enc):.3f}", action])
    table(["Cell", "Detour", "VI sweeps", "Move only if P(blocked) <",
           "Action at P=0.659"], rows)


def exp_policies() -> None:
    emit(f"## E4. Policy comparison ({EPISODES} deliveries per policy, same worlds)\n")
    emit("Cost = delivery ticks + 50 x collisions (lower is better).\n")
    raw["policies"] = {}
    for name in MAPS:
        grid = load_map(name)
        rows = []
        for policy, label in POLICIES.items():
            r = evaluate(grid, policy, EPISODES)
            raw["policies"].setdefault(name, {})[policy] = r
            rows.append([label, f"{r['avg_cost']:.2f}", f"{r['avg_ticks']:.2f}",
                         f"{r['avg_collisions']:.3f}", f"{100 * r['collision_free_rate']:.1f}%",
                         f"{r['avg_waits']:.2f}", f"{r['avg_reroutes']:.2f}",
                         f"{100 * r['success_rate']:.0f}%"])
        emit(f"**{name}**\n")
        table(["Policy", "Avg cost", "Avg ticks", "Collisions", "Collision-free",
               "Waits", "Reroutes", "Delivered"], rows)


def exp_sensitivity() -> None:
    emit(f"## E5. Sensitivity to the collision penalty (Campus map, {EPISODES} episodes)\n")
    grid = load_map("Campus 9x12")
    rows, raw["sensitivity"] = [], {}
    for penalty in (5, 20, 50, 100):
        cfg = SimConfig(costs=Costs(collision=penalty))
        costs = {p: evaluate(grid, p, EPISODES, cfg)["avg_cost"] for p in POLICIES}
        raw["sensitivity"][penalty] = costs
        best = min(costs, key=costs.get)
        rows.append([penalty] + [f"{costs[p]:.2f}" for p in POLICIES] + [POLICIES[best]])
    table(["Collision penalty"] + list(POLICIES.values()) + ["Best"], rows)

    emit(f"## E6. Sensitivity to sensor quality (Campus map, 500 episodes)\n")
    rows = []
    for fa in (0.1, 0.2, 0.4):
        cfg = SimConfig(sensor=SensorModel(0.9, fa))
        costs = {p: evaluate(grid, p, 500, cfg)["avg_cost"] for p in POLICIES}
        best = min(costs, key=costs.get)
        rows.append([fa] + [f"{costs[p]:.2f}" for p in POLICIES] + [POLICIES[best]])
    table(["False-alarm rate"] + list(POLICIES.values()) + ["Best"], rows)


def main() -> None:
    t0 = time.time()
    emit("# Experiment results\n")
    emit("Generated by `python experiments/run_experiments.py` (seeded, reproducible).\n")
    for exp in (exp_search, exp_bayes, exp_mdp, exp_policies, exp_sensitivity):
        exp()
    emit(f"_Total runtime: {time.time() - t0:.1f} s_")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "RESULTS.md").write_text("\n".join(out) + "\n")
    (ROOT / "results" / "results.json").write_text(json.dumps(raw, indent=2, default=str))


if __name__ == "__main__":
    main()
