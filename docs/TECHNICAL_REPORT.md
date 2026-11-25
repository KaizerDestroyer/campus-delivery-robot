# Campus Delivery Robot: Route Planning, Belief Tracking and Decision Making under Uncertainty

**Technical report — AI Fundamentals, Final Project**
Author: Karim · November 2026 · Version 2.0

---

## Abstract

We build a simulated campus delivery robot that must bring a package from the main gate
to a dormitory on a grid map. Busy areas are often temporarily blocked, and the robot's
obstacle sensor is unreliable (90% detection rate, 20% false-alarm rate). The system
combines three algorithms from three course topics: **A\* search** plans the route and
prices detours, a **Bayes filter** converts noisy sensor readings into a probability that
the next cell is blocked, and a **Markov decision process over beliefs**, solved by
**value iteration**, chooses between *Move*, *Wait* and *Reroute*. In 1000 simulated
deliveries on the campus map the full system reduces collisions from 2.173 to 0.139 per
delivery compared with a robot that only uses A\* (−94%), and achieves the lowest
average cost (35.75 vs 129.82). An ablation study with four policies and two
sensitivity studies show where each component helps and where the model's simplifying
assumptions cost performance.

---

## 1. Introduction and problem statement

Small sidewalk delivery robots are already used on university campuses. Their main
difficulty is not finding a path on a known map, but moving safely among people when
sensors are imperfect. A robot that trusts every sensor alarm stops too often; a robot
that ignores its sensor runs into people.

We model this as follows:

- The campus is a 9 × 12 grid with buildings (walls), a start **S** (main gate) and a
  goal **G** (dormitory). The robot moves up, down, left or right, one cell per tick.
- Some cells are **hotspots** (busy areas). At any time a hotspot is blocked by a
  temporary obstacle with probability 0.30; an ordinary cell with probability 0.02.
  Obstacles appear and disappear over time.
- Before each step the robot reads an obstacle sensor pointed at the next cell. The
  sensor beeps with probability 0.90 if the cell is blocked and 0.20 if it is clear.
- The robot's goal is to minimise **cost = delivery time (ticks) + 50 × collisions**.

**Objectives.** (1) Plan an optimal route on the known map. (2) Maintain a calibrated
probability that the next cell is blocked. (3) Make decisions that are optimal with
respect to the stated costs. (4) Show experimentally that each component contributes.

---

## 2. Course topics used

| Topic | Algorithm | Why it fits |
|---|---|---|
| Search | A\* with the Manhattan heuristic | The static map is fully known and step costs are uniform, so shortest-path search is the natural tool. A\* is optimal with an admissible heuristic and expands far fewer nodes than uninformed search. It is also used again inside the loop to compute the length of detours. |
| Probabilistic reasoning | Bayes filter (Bayes' rule + Markov time update) | The true state of the next cell is hidden and the sensor is noisy. Bayes' rule combines prior knowledge (hotspot or not) with the sensor evidence; the time update accounts for obstacles changing while the robot waits. |
| Markov decision processes | Belief MDP + value iteration | Waiting, moving and rerouting have long-term consequences (waiting now gives a new reading later). An MDP captures these sequential trade-offs, and value iteration computes the optimal action for every belief level. |

The three are not independent demonstrations: the output of each is an input of the
next (route → sensed cell → belief → decision → new route or new belief).

---

## 3. System design

### 3.1 Architecture

```text
robot/grid.py        map parsing, prior risk per cell, random maps
robot/search.py      A*, UCS, BFS, Greedy                         (Topic 1)
robot/bayes.py       SensorModel, ObstacleDynamics, Bayes filter  (Topic 2)
robot/mdp.py         generic MDP, value iteration, belief MDP     (Topic 3)
robot/simulation.py  hidden world, decision loop, policies, evaluation
main.py              command-line demo
app.py               Streamlit web interface
experiments/         reproducible experiments E1–E6
```

### 3.2 Decision loop (one tick)

1. Take the next cell on the current A\* route.
2. Read the sensor; update the belief $b = P(\text{blocked})$ with Bayes' rule.
3. Run A\* again with that cell forbidden to get the detour length; *extra* = detour
   length − remaining route length.
4. Choose an action with the policy under test (the full system uses the belief MDP).
5. Execute it in the hidden world: *Move* into a clear cell advances; *Move* into a
   blocked cell is a collision (the robot stays and pays the penalty); *Wait* stays;
   *Reroute* replaces the route by the detour.
6. The world ticks: each obstacle may appear or disappear, and all tracked beliefs are
   propagated with the time update.

### 3.3 Hidden world model

Each free cell $c$ has a long-run risk $r_c$ (0.30 for hotspots, 0.02 otherwise, 0 for
S and G). Its obstacle follows a two-state Markov chain:

$$P(\text{clear}\to\text{blocked}) = r_c \cdot m, \qquad P(\text{blocked}\to\text{clear}) = (1-r_c)\cdot m$$

with mixing rate $m = 0.5$. The stationary probability of being blocked is then exactly
$r_c$. For a hotspot this gives $p_{block} = 0.15$, $p_{clear} = 0.35$; for an ordinary
cell $p_{block} = 0.01$, $p_{clear} = 0.49$. The robot knows this model but never sees
the true state.

---

## 4. Methods

### 4.1 A\* search

A\* expands the frontier node with the smallest

$$f(n) = g(n) + h(n)$$

where $g(n)$ is the number of steps from the start and $h(n)$ is the Manhattan distance
$|r_n - r_G| + |c_n - c_G|$.

*Admissibility.* Each move changes either the row or the column by exactly 1, so it
reduces the Manhattan distance by at most 1. Any path from $n$ to $G$ therefore has at
least $h(n)$ steps: $h$ never overestimates.

*Consistency.* For neighbours $n, n'$: $h(n) \le 1 + h(n')$ (triangle inequality with
step cost 1). With a consistent heuristic, the first time A\* removes a node from the
priority queue its $g$ value is optimal, so a closed set is safe and the returned path is
optimal.

*Implementation details.* A binary heap stores entries $(f, h, \text{counter}, g, n)$.
Ties on $f$ are broken by smaller $h$ (prefer nodes closer to the goal), then by
insertion order; the counter means cells are never compared directly. Stale heap
entries are skipped when popped. The path is rebuilt from a `came_from` map instead of
copying a path list into every heap entry, as version 1 did.

*Comparison algorithms.* The same best-first routine gives **UCS/Dijkstra** ($f = g$,
optimal but uninformed) and **Greedy best-first** ($f = h$, fast but not optimal);
**BFS** is implemented with a queue and is optimal for unit step costs. BFS is used as
the ground truth in tests.

### 4.2 Bayes filter

*Measurement update.* With $H$ = "next cell blocked" and evidence $e$:

$$P(H \mid e) = \frac{P(e \mid H)\,P(H)}{P(e \mid H)\,P(H) + P(e \mid \neg H)\,(1-P(H))}$$

Worked example at a hotspot after an alert:

$$P(H \mid \text{alert}) = \frac{0.9 \cdot 0.3}{0.9 \cdot 0.3 + 0.2 \cdot 0.7} = \frac{0.27}{0.41} = 0.659$$

After a quiet reading: $\frac{0.1 \cdot 0.3}{0.1 \cdot 0.3 + 0.8 \cdot 0.7} = \frac{0.03}{0.59} = 0.051$.
The same alert at an ordinary cell gives only 0.084: when obstacles are rare, most
alerts are false alarms (the base-rate effect).

*Time update.* While the robot waits, the obstacle may change:

$$b_{t+1}^{-} = b_t\,(1 - p_{clear}) + (1 - b_t)\,p_{block}$$

Example from the demo run: after an alert ($b = 0.659$) the robot waits one tick, the
belief drifts to $0.659 \cdot 0.65 + 0.341 \cdot 0.15 = 0.48$, and a quiet reading then
lowers it to 0.10.

The filter rejects probabilities outside [0, 1] and observations that are impossible
under the model ($P(e) = 0$).

### 4.3 Belief MDP and value iteration

The robot does not observe whether the cell is blocked, so we define the MDP over what
it *does* know — its belief.

- **States:** 101 belief levels $b \in \{0.00, 0.01, \dots, 1.00\}$ plus a terminal state
  *Passed* (the obstacle is behind us).
- **Actions:** *Move*, *Wait*, and *Reroute* (only if A\* finds a detour).
- **Rewards** (costs are negative rewards):
  - *Move*: $-(1 + 50\,b)$ — one tick plus the expected collision penalty;
  - *Wait*: $-1$;
  - *Reroute*: $-(1 + \text{extra} + 1)$ — the step, the extra detour steps, and one tick
    to stop and re-plan.
- **Transitions:**
  - *Move*: with probability $1-b$ the robot passes (→ *Passed*); with probability $b$
    it bumps, now knows the cell was blocked (belief 1), and the next belief is obtained
    by one filter step from 1.
  - *Wait*: one filter step from $b$ — time update, then the sensor beeps with
    probability $P(\text{alert}) = 0.9\,b^- + 0.2\,(1-b^-)$ or stays quiet, and Bayes'
    rule gives the new belief in each case.
  - *Reroute*: → *Passed*.
- **Discount:** $\gamma = 0.95$.

A posterior such as 0.6585 falls between grid points; it is split between the two
nearest points (0.65 with weight 0.15 and 0.66 with weight 0.85). This linear
interpolation keeps every transition row a valid probability distribution, so the belief
MDP is an ordinary finite MDP (a grid-based belief-space approximation, cf. Lovejoy 1991).

**Value iteration** applies the Bellman optimality update to all states until the largest
change is below $10^{-6}$:

$$V_{k+1}(s) = \max_a \Big[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, V_k(s') \Big]$$

Then $Q(s,a)$ is computed with the converged $V$ and the policy is
$\pi(s) = \arg\max_a Q(s,a)$. At run time the robot evaluates $Q(b,a)$ for its exact
belief with one look-ahead step using $V$, and picks the best action (ties go to *Move*,
then *Wait*).

Example (hotspot, no detour, $b = 0.659$): $Q(\text{Move}) = -37.81$,
$Q(\text{Wait}) = -6.49$ → **Wait**. Moving is optimal only while $b < 0.064$.
With a detour of +2 steps: $Q(\text{Reroute}) = -4.00$ beats $Q(\text{Wait}) = -4.78$
→ **Reroute**.

The generic `MDP` class and `value_iteration` function are independent of the robot;
they are tested on textbook MDPs with known solutions.

---

## 5. Implementation

- **Language and dependencies.** Python 3.10+. The `robot` package and the CLI use only
  the standard library; the web UI uses Streamlit; tests use pytest.
- **Code quality.** Type hints, docstrings that state the formulas, frozen dataclasses
  for model parameters (`SensorModel`, `ObstacleDynamics`, `Costs`, `Encounter`,
  `SimConfig`), input validation with clear error messages.
- **Complexity.** A\* runs in $O(E \log V)$ with a binary heap; on the 108-cell campus map
  it expands 22 nodes. One value-iteration sweep costs $O(|S|\cdot|A|\cdot k)$ with
  $|S| = 102$, $|A| \le 3$ and $k \le 4$ successors; it converged in 5–50 sweeps in our
  settings.
- **Caching.** The belief MDP depends only on the cell type, detour length, sensor,
  costs and $\gamma$. `solve()` is wrapped in `functools.lru_cache`, so each encounter type
  is solved once; 4000 simulated deliveries run in a few seconds.
- **Reproducibility.** Every episode is seeded. The world and the sensor use **separate
  random streams**, so for a given seed all four policies face exactly the same sequence
  of obstacles (common random numbers). This makes policy comparisons fair and lowers
  their variance. `experiments/run_experiments.py` regenerates every table in about 30 s.

---

## 6. Testing

The suite has **216 pytest tests** for the core (most are parametrized), plus a smoke
test that runs the Streamlit UI headlessly.

| Area | What is tested |
|---|---|
| Search (158) | v1 result reproduced (8 steps on the 5×5 map); all four algorithms return valid paths (adjacent steps, no walls, no repeats); **property test: A\* cost equals BFS cost on 150 random 12×12 maps** and A\* never expands more nodes than UCS; Manhattan admissible for every cell of the campus map; Greedy returns 18 steps where A\* returns 16; unreachable goal; forbidden cells; start = goal |
| Bayes (11) | 0.27/0.41 hand calculation; quiet reading 0.03/0.59; base-rate effect; two alerts → 0.897; certain priors unchanged; uninformative sensor; invalid inputs and impossible observations raise errors; time update converges to the stationary risk; **Monte-Carlo test with 200 000 samples** |
| MDP (9) | value iteration on a two-step chain ($V(A) = -1.9$ for $\gamma = 0.9$) and on a safe/risky choice; malformed MDPs rejected; belief MDP has 102 states and valid transition rows; convergence; Move at $b=0$, Wait at $b = 0.659$ and $b = 1$; cheap detour → Reroute, expensive → Wait; higher penalty → lower Move threshold |
| Simulation (38) | every policy delivers on 30 seeds and never beats the shortest path; same seed → same result; log consistent with the counters; full system has < 25% of the naive collisions; full system within 5% of the best policy; **stability on 25 random maps**; bad maps and unknown policies rejected |

All 216 core tests pass in under 2 seconds.

---

## 7. Results

All numbers come from `results/RESULTS.md`.

### E1. Search

| Map | A\* (length / expanded) | UCS | BFS | Greedy |
|---|---|---|---|---|
| Campus 9×12 | 19 / 22 | 19 / 69 | 19 / 69 | 19 / 22 |
| Classic 5×5 | 8 / 9 | 8 / 22 | 8 / 22 | 8 / 9 |

On 500 solvable random 15×15 maps with 30% walls:

| Algorithm | Avg nodes expanded | Optimal path found |
|---|---|---|
| A\* | 50.0 | 100.0% |
| UCS (Dijkstra) | 145.6 | 100.0% |
| BFS | 145.8 | 100.0% |
| Greedy | 42.5 | 36.8% |

A\* is always optimal and expands about 2.9× fewer nodes than uninformed search. Greedy
expands slightly fewer nodes but finds the shortest path on only 36.8% of the maps. This
is why the robot uses A\*.

### E2. Bayes filter

| Prior | Reading | Formula | Simulated frequency (200k) |
|---|---|---|---|
| 0.30 | ALERT | 0.6585 | 0.6596 |
| 0.30 | quiet | 0.0508 | 0.0506 |
| 0.02 | ALERT | 0.0841 | 0.0828 |
| 0.02 | quiet | 0.0025 | 0.0025 |

The computed posteriors match empirical frequencies to within about 0.001, so the
beliefs are calibrated. A sequence ALERT, ALERT, quiet, ALERT at a hotspot gives
0.659 → 0.897 → 0.520 → 0.830: evidence accumulates, and one quiet reading does not
erase two alerts.

### E3. MDP policy

| Cell | Detour | VI sweeps | Move only if $b <$ | Action at $b = 0.659$ |
|---|---|---|---|---|
| hotspot | none | 50 | 0.064 | Wait |
| hotspot | +0 | 5 | 0.020 | Reroute |
| hotspot | +2 | 8 | 0.052 | Reroute |
| hotspot | +4 | 23 | 0.063 | Reroute |
| hotspot | +6 | 50 | 0.064 | Wait |
| ordinary | none | 25 | 0.027 | Wait |
| ordinary | +0 | 5 | 0.020 | Reroute |
| ordinary | +2 | 19 | 0.027 | Reroute |
| ordinary | +4 | 25 | 0.027 | Wait |
| ordinary | +6 | 25 | 0.027 | Wait |

The robot moves only when it is quite sure the cell is clear, because a collision costs
50 ticks. With a short detour available it reroutes; when the detour is long it waits,
because obstacles clear quickly (a hotspot obstacle disappears with probability 0.35
per tick). At ordinary cells obstacles clear even faster (0.49 per tick), so a +4 detour
is already not worth it there, while it still is at a hotspot. Value iteration converges
in at most 50 sweeps.

### E4. Policy comparison (ablation)

Four policies, 1000 deliveries each, identical worlds:

- **A\* only** — ignores the sensor and always moves;
- **Raw sensor** — waits whenever the sensor beeps;
- **Bayes + 0.5 threshold** — waits if $b > 0.5$;
- **Bayes + MDP** — the full system.

Campus 9×12:

| Policy | Avg cost | Avg ticks | Collisions | Collision-free | Waits | Reroutes |
|---|---|---|---|---|---|---|
| A\* only | 129.82 | 21.17 | 2.173 | 37.9% | 0.00 | 0.00 |
| Raw sensor | 38.29 | 26.39 | 0.238 | 80.8% | 7.15 | 0.00 |
| Bayes + 0.5 | 48.23 | 22.03 | 0.524 | 63.4% | 2.51 | 0.00 |
| **Bayes + MDP** | **35.75** | 28.80 | **0.139** | **87.6%** | 7.76 | 1.28 |

Classic 5×5:

| Policy | Avg cost | Avg ticks | Collisions | Collision-free |
|---|---|---|---|---|
| A\* only | 65.48 | 9.13 | 1.127 | 58.6% |
| Raw sensor | 15.74 | 11.14 | 0.092 | 91.2% |
| Bayes + 0.5 | 19.93 | 9.43 | 0.210 | 81.9% |
| **Bayes + MDP** | **14.88** | 12.43 | **0.049** | **95.2%** |

All policies deliver in 100% of the episodes. Search alone is fastest but collides about
twice per delivery. Adding the sensor removes most collisions. A Bayes belief with an
arbitrary 0.5 threshold is *worse* than the raw sensor: the threshold is not tied to the
costs, so it moves at beliefs where the expected collision cost is far larger than the
cost of waiting. The MDP chooses its threshold from the costs (0.064 at a hotspot) and
can also reroute. It pays about 7.6 extra ticks compared with A\* only, and in return
cuts collisions by 94% and gives the lowest total cost on both maps.

### E5. Sensitivity to the collision penalty (campus, 1000 episodes)

| Penalty | A\* only | Raw sensor | Bayes + 0.5 | Bayes + MDP | Best |
|---|---|---|---|---|---|
| 5 | 32.04 | 27.58 | 24.65 | **24.52** | Bayes + MDP |
| 20 | 64.63 | **31.15** | 32.51 | 31.31 | Raw sensor |
| 50 | 129.82 | 38.29 | 48.23 | **35.75** | Bayes + MDP |
| 100 | 238.47 | 50.19 | 74.43 | **43.53** | Bayes + MDP |

The fixed rules each work well only in one regime: the 0.5 threshold is good when
collisions are cheap, the raw-sensor rule when they cost about 20. The MDP re-solves
for each penalty and is the best policy at 5, 50 and 100. At penalty 20 the raw-sensor
rule is marginally better (31.15 vs 31.31). This difference is within run-to-run noise,
but we do not claim the MDP wins there.

### E6. Sensitivity to sensor quality (campus, 500 episodes)

| False-alarm rate | A\* only | Raw sensor | Bayes + 0.5 | Bayes + MDP | Best |
|---|---|---|---|---|---|
| 0.1 | 131.91 | 36.54 | 48.50 | **32.50** | Bayes + MDP |
| 0.2 | 131.91 | 39.41 | 50.62 | **36.16** | Bayes + MDP |
| 0.4 | 131.91 | **48.25** | 77.21 | 51.72 | Raw sensor |

With a very noisy sensor (40% false alarms) the raw-sensor rule beats the full system
(48.25 vs 51.72). In a diagnostic re-run of this setting (same 500 seeds) the full
system is still safer (0.216 vs 0.268 collisions per delivery), but slower (40.9 vs 34.8 ticks): it waits
more (18.0 vs 15.6 times per delivery) and reroutes 2.2 times per delivery.

The cause is a simplification in the decision model: **each encounter is decided in
isolation.** The MDP prices a detour only by its length. It ignores the obstacle risk on
the detour and on the rest of the route. In the first 200 of those episodes, 405 of 430
detours (94%) passed through at least one other hotspot, so a reroute often leads straight into
another alert. With 40% false alarms this happens so often that the extra time outweighs
the safety gain. A higher discount factor does not fix it ($\gamma = 0.99$ gave 54.16), which suggests
the one-encounter scope is the main problem, not discounting. Note that in this
experiment the robot knows the true sensor model; with a mis-estimated sensor model,
which is the realistic case, the gap could be larger (see Section 9).

---

## 8. Discussion and practical value

The experiments support the main claim: **search, probabilistic reasoning and decision
making each fix a different failure.** Search alone gives a short but unsafe route.
Raw sensing is safer but cannot tell a rare false alarm from a likely obstacle. Bayes
gives calibrated probabilities, but probabilities alone do not say what to do — the 0.5
threshold was the second-worst policy. Only when the belief is combined with explicit
costs in an MDP does the robot behave well under different cost settings.

For a real campus robot, the useful property is that the trade-off is explicit and
adjustable. An operator can say "a collision is as bad as 50 seconds of delay" and the
policy re-derives itself, instead of hand-tuning thresholds. The same structure (plan →
filter → decide) is used in real mobile robots, with richer maps and sensor models.

---

## 9. Limitations and future work

- **Myopic decision scope.** Solve a POMDP over the whole remaining route (or use online
  POMDP planning) so that detour risk and later hotspots are priced in.
- **Known sensor model.** The robot assumes its sensor model (0.9 / 0.2) and the
  obstacle dynamics are exactly right. A real robot should learn the detection and
  false-alarm rates and the dynamics from logged data (for example by maximum
  likelihood or EM) and update them online.
- **Unit-cost grid.** Use a real campus map (OpenStreetMap) with weighted edges (slopes,
  crossings), and continuous space with a motion model.
- **One robot.** Several robots could share sensor readings, which would make the
  beliefs about common hotspots sharper.
- **Belief discretisation.** A finer grid or point-based value iteration would reduce
  the interpolation error.

---

## 10. Changes from version 1

Version 1 was a single 185-line script by the original team. Version 2 keeps its
scenario and its numbers (sensor 0.9 / 0.2, hotspot prior 0.30, the 5×5 map) and changes:

| Version 1 | Version 2 |
|---|---|
| A\* only, path copied into every heap entry | A\* with `came_from`, tie-breaking, forbidden cells for detours; UCS/BFS/Greedy for comparison |
| One Bayes update with fixed numbers | Bayes filter with measurement and time updates, sequential readings, validation |
| "MDP" = one-step expected reward of Move vs Wait (no transitions, no value iteration) | Real MDP over beliefs with transitions, three actions and value iteration |
| Three separate printouts | One decision loop in a simulated world with hidden obstacles |
| No tests, no experiments | 216 tests, six reproducible experiments, ablation study |
| Console only | CLI and Streamlit web UI |

The v1 README also gave the Move/Wait break-even point as ≈ 0.355. For its own reward
table ($E[R \mid \text{Move}] = 10 - 30p$, $E[R \mid \text{Wait}] = -1$) the correct value is
$p = 11/30 \approx 0.367$. Version 2 computes thresholds in code (`move_threshold`)
instead of stating them by hand.

---

## References

1. P. E. Hart, N. J. Nilsson, B. Raphael. "A Formal Basis for the Heuristic Determination
   of Minimum Cost Paths." *IEEE Transactions on Systems Science and Cybernetics*, 4(2),
   1968.
2. S. Russell, P. Norvig. *Artificial Intelligence: A Modern Approach*, 4th ed.
   Pearson, 2020 (Ch. 3 search, Ch. 12–14 probabilistic reasoning, Ch. 17 MDPs).
3. R. Bellman. *Dynamic Programming.* Princeton University Press, 1957.
4. S. Thrun, W. Burgard, D. Fox. *Probabilistic Robotics.* MIT Press, 2005.
5. L. P. Kaelbling, M. L. Littman, A. R. Cassandra. "Planning and Acting in Partially
   Observable Stochastic Domains." *Artificial Intelligence*, 101, 1998.
6. W. S. Lovejoy. "Computationally Feasible Bounds for Partially Observed Markov
   Decision Processes." *Operations Research*, 39(1), 1991.
