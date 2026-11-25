# Demo Video Script (2:30)

**Project:** Campus Delivery Robot — AI Fundamentals Final Project · Karim

Target length 2:30 (the limit is 2–3 minutes). Narration is about 330 words at a calm
pace. Read the lines exactly as written, or in your own words.

## Before recording

1. Open a terminal in the project folder and activate the environment:
   `source .venv/bin/activate`
2. Start the UI in a second terminal: `streamlit run app.py`. Leave the sidebar at its
   defaults: map **Campus 9x12**, policy **Bayes + MDP (full system)**, seed **5**,
   collision penalty **50**.
3. Make the terminal font large (Cmd + `+`) and close unrelated windows and
   notifications.

## How to record on macOS

Press **Cmd + Shift + 5** → choose *Record Selected Portion* (or *Record Entire Screen*)
→ *Options* → set *Microphone* to your built-in mic → *Record*. To stop, click the stop
button in the menu bar. The video is saved to the Desktop as a `.mov` file. To trim it,
open it in QuickTime Player → *Edit → Trim*.

## Script

| Time | On screen | Narration |
|---|---|---|
| 0:00–0:15 | README title on GitHub, or the UI header | "This is my Campus Delivery Robot. It carries a package from the main gate to a dormitory. Busy areas are often blocked by people, and the robot's sensor is noisy, so it must plan, reason under uncertainty, and decide what to do." |
| 0:15–0:40 | Terminal: run `python main.py`, scroll through sections 1–3 | "The project combines three course topics. First, A\* search finds the shortest route: 19 steps, expanding only 22 cells, against 69 for BFS. Second, a Bayes filter: at a busy cell, one alert raises the probability of an obstacle from 0.30 to 0.66. Third, an MDP solved with value iteration decides between Move, Wait and Reroute." |
| 0:40–1:15 | UI → **🚚 Mission** tab. Click **Next ▶** to the step at t=7, then to t=11 and t=12. Turn on **Show ground truth** | "Here is one delivery with seed 5. At tick 7 the sensor beeps in front of the hotspot at row 2, column 6. Bayes gives 0.66, a short detour exists, so the MDP chooses Reroute. Ground truth: that cell really was blocked. At tick 11 it beeps again in front of row 5, column 4. Here waiting is cheaper, so the robot waits, senses again, and continues. It delivers in 24 ticks with zero collisions." |
| 1:15–1:30 | Tick **Compare with "A\* only (ignore sensor)"** | "On exactly the same world, a robot that uses only A\* and ignores its sensor collides three times." |
| 1:30–1:45 | **🧭 A\* Search** tab: show the comparison table and the explored cells | "In the search tab you can compare A\*, Dijkstra, BFS and Greedy. A\* is always optimal and explores far fewer cells. Greedy is faster but often finds longer paths." |
| 1:45–2:00 | **📡 Bayes Filter** tab: keep the default readings `A A Q`, show the table and the chart | "The Bayes tab shows the update step by step. Two alerts give 0.90; a quiet reading lowers the belief again." |
| 2:00–2:15 | **🧠 MDP** tab: set *Cell type* = Hotspot and *A\* detour* = **no detour**; show the Q-value curves and the metric "Move is best while P(blocked) <" (0.064) | "The MDP tab shows the value of each action for every belief level. Moving is optimal only while the probability of an obstacle is below about six percent, because a collision costs 50 ticks." |
| 2:15–2:30 | **📊 Experiments** tab: run the comparison, show the table and bar chart | "Over 1000 deliveries, the full system has the lowest cost and cuts collisions by 94 percent compared with A\* alone. Thank you." |

## Numbers used in the narration (all checked against the program)

- A\*: 19 steps, 22 nodes expanded; BFS and UCS: 69 (`python main.py`, `results/RESULTS.md` E1).
- Bayes: 0.27 / 0.41 = 0.659 ≈ 0.66; two alerts → 0.897 ≈ 0.90 (E2).
- MDP: Move threshold 0.064 at a hotspot without detour (E3).
- Seed 5 mission: Reroute at t=7 in front of (2, 6); Wait at t=11 and t=12 in front of
  (5, 4); 24 ticks, 0 collisions. Naive policy on seed 5: 3 collisions
  (`python main.py --policy naive`).
- 1000 deliveries: cost 35.75 vs 129.82; collisions 0.139 vs 2.173 per delivery (−94%) (E4).

The UI's **Experiments** tab defaults to 200 episodes per policy, so its numbers will be
close to, but not exactly, the 1000-episode numbers. Set the slider to 1000 to match
the report.
