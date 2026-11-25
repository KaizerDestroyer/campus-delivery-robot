"""Command-line demo of the Campus Delivery Robot (standard library only).

    python main.py                       # full walkthrough on the campus map
    python main.py --map "Classic 5x5"   # the original 5x5 map
    python main.py --seed 7 --policy reactive
    python main.py --compare 500         # compare all policies over 500 deliveries
"""

from __future__ import annotations

import argparse

from robot import (ALGORITHMS, MAPS, POLICIES, Encounter, ObstacleDynamics, SensorModel,
                   SimConfig, bayes_update, decide, evaluate, filter_readings, load_map,
                   move_threshold, run_episode)
from robot.mdp import solve

LINE = "=" * 66


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def demo_search(grid) -> None:
    section("[1] SEARCH -- A* route planning")
    print(f"Start {grid.start} -> Goal {grid.goal}, walls: {len(grid.walls)}, "
          f"hotspots: {len(grid.hotspots)}")
    results = [alg(grid) for alg in ALGORITHMS.values()]
    print(f"\n  {'Algorithm':<16}{'Path length':>12}{'Nodes expanded':>16}")
    for r in results:
        print(f"  {r.algorithm:<16}{r.cost:>12}{r.expanded:>16}")
    best = results[0]
    print("\nA* route (* = path, o = hotspot, # = building):\n")
    print("  " + grid.render(best.path).replace("\n", "\n  "))


def demo_bayes(grid, sensor: SensorModel) -> float:
    section("[2] PROBABILISTIC REASONING -- Bayes filter")
    prior = grid.hotspot_risk
    post = bayes_update(prior, True, sensor)
    evidence = sensor.sensitivity * prior + sensor.false_alarm * (1 - prior)
    print(f"Prior P(blocked) at a hotspot     = {prior:.2f}")
    print(f"P(alert | blocked)                = {sensor.sensitivity:.2f}")
    print(f"P(alert | clear)                  = {sensor.false_alarm:.2f}")
    print(f"P(alert) = {sensor.sensitivity}*{prior} + {sensor.false_alarm}*{1 - prior:.1f}"
          f"      = {evidence:.2f}")
    print(f"Posterior P(blocked | alert)      = {sensor.sensitivity * prior:.2f} / "
          f"{evidence:.2f} = {post:.3f}")
    print(f"Posterior P(blocked | no alert)   = {bayes_update(prior, False, sensor):.3f}")
    print(f"Same alert on an ordinary cell    = {bayes_update(grid.normal_risk, True, sensor):.3f}"
          "   (rare obstacles -> alert is probably false)")
    readings = [True, True, False]
    beliefs = filter_readings(prior, readings, sensor)
    trace = "  ->  ".join(f"{'ALERT' if a else 'quiet'}: {b:.3f}" for a, b in zip(readings, beliefs))
    print(f"Sequence of readings              : {trace}")
    return post


def demo_mdp(grid, sensor: SensorModel, cfg: SimConfig, belief: float) -> None:
    section("[3] DECISION MAKING -- MDP solved by value iteration")
    dyn = ObstacleDynamics.from_risk(grid.hotspot_risk, cfg.mixing)
    c = cfg.costs
    print(f"States : belief P(blocked) = 0.00, 0.01, ..., 1.00  (+ terminal 'Passed')")
    print(f"Actions: Move, Wait, Reroute")
    print(f"Rewards: -{c.step:g} per tick, -{c.collision:g} collision, "
          f"-(1 + extra + {c.replan:g}) detour;  gamma = {cfg.gamma}")
    for extra in (None, 2, 6):
        enc = Encounter(dyn, sensor, c, extra, cfg.gamma)
        res = solve(enc)
        action, scores = decide(belief, enc)
        label = "no detour" if extra is None else f"detour +{extra} steps"
        q = ", ".join(f"Q({a})={v:.2f}" for a, v in scores.items())
        print(f"\n  [{label}] value iteration converged in {res.iterations} sweeps")
        print(f"     at P(blocked)={belief:.3f}: {q}")
        print(f"     -> best action: {action.upper()}   "
              f"(Move is optimal only while P(blocked) < {move_threshold(enc):.3f})")


def demo_mission(grid, policy: str, seed: int, cfg: SimConfig) -> None:
    section(f"[4] MISSION -- policy '{policy}' ({POLICIES[policy]}), seed {seed}")
    ep = run_episode(grid, policy, seed, cfg)
    for s in ep.log:
        if s.outcome == "moved" and not s.alert and s.posterior < 0.05:
            continue  # skip boring quiet steps
        sensor = "ALERT" if s.alert else "quiet"
        truth = "blocked" if s.truly_blocked else "clear"
        print(f"  t={s.tick:>3} at {s.position} next {s.target}: sensor {sensor:<5} "
              f"P(blocked) {s.prior:.2f}->{s.posterior:.2f}  => {s.action:<7} "
              f"[{s.outcome}, truth: {truth}]")
    print(f"\n  Delivered: {ep.delivered}  ticks: {ep.ticks}  collisions: {ep.collisions}  "
          f"waits: {ep.waits}  reroutes: {ep.reroutes}  total cost: {ep.total_cost:g}")


def compare(grid, episodes: int, cfg: SimConfig) -> None:
    section(f"POLICY COMPARISON over {episodes} deliveries (same worlds for every policy)")
    print(f"  {'Policy':<28}{'Cost':>8}{'Ticks':>8}{'Collisions':>12}{'No-crash %':>12}")
    for policy, label in POLICIES.items():
        r = evaluate(grid, policy, episodes, cfg)
        print(f"  {label:<28}{r['avg_cost']:>8.2f}{r['avg_ticks']:>8.2f}"
              f"{r['avg_collisions']:>12.3f}{100 * r['collision_free_rate']:>11.1f}%")
    print("\n  Cost = delivery time + collision penalty x collisions (lower is better)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", default="Campus 9x12", choices=list(MAPS))
    parser.add_argument("--policy", default="bayes_mdp", choices=list(POLICIES))
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--compare", type=int, metavar="N",
                        help="only run the policy comparison over N episodes")
    args = parser.parse_args()

    grid = load_map(args.map)
    cfg = SimConfig()
    print(LINE)
    print("Campus Delivery Robot -- AI Fundamentals Final Project")
    print("A* search  +  Bayes filter  +  MDP (value iteration)")
    print(LINE)

    if args.compare:
        compare(grid, args.compare, cfg)
        return
    demo_search(grid)
    posterior = demo_bayes(grid, cfg.sensor)
    demo_mdp(grid, cfg.sensor, cfg, posterior)
    demo_mission(grid, args.policy, args.seed, cfg)


if __name__ == "__main__":
    main()
