"""Streamlit web UI for the Campus Delivery Robot.

    streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import re

import altair as alt
import pandas as pd
import streamlit as st

from robot import (ALGORITHMS, MAPS, POLICIES, Costs, Encounter, ObstacleDynamics,
                   SensorModel, SimConfig, bayes_update, evaluate, load_map,
                   move_threshold, predict, run_episode)
from robot.mdp import MOVE, REROUTE, WAIT, action_values, solve

st.set_page_config(page_title="Campus Delivery Robot", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# Colours (explicit, so the map reads well in both light and dark themes)
# ---------------------------------------------------------------------------

CELL_STYLE = {  # kind -> (background, text)
    "wall": ("#4b5763", "#dfe6e9"),
    "free": ("#f1f2f6", "#2d3436"),
    "hotspot": ("#fdcb6e", "#4a3500"),
    "route": ("#74b9ff", "#0b2545"),
    "route_hot": ("#f0932b", "#2d1600"),
    "explored": ("#b8c7d9", "#1e272e"),
    "start": ("#00a878", "#ffffff"),
    "goal": ("#d63031", "#ffffff"),
}
ACTION_COLORS = {MOVE: "#1e9e57", WAIT: "#d68910", REROUTE: "#2471a3"}
OUTCOME_TEXT = {
    "moved": "moved forward",
    "collision": "COLLISION (stayed in place)",
    "waited": "waited one tick",
    "rerouted": "stopped to re-plan and switched to the detour",
}
TARGET_RING = "#e84118"


def grid_html(grid, *, route=(), robot=None, target=None, explored=(), start=None,
              goal=None, blocked_target=False, cell_px=34) -> str:
    """Render the campus map as an HTML table."""
    start, goal = start or grid.start, goal or grid.goal
    route_set, explored_set = set(route), set(explored)
    rows = []
    for r in range(grid.rows):
        tds = []
        for c in range(grid.cols):
            cell = (r, c)
            label = ""
            if cell in grid.walls:
                kind = "wall"
            elif cell == start:
                kind, label = "start", "S"
            elif cell == goal:
                kind, label = "goal", "G"
            elif cell in route_set:
                kind = "route_hot" if cell in grid.hotspots else "route"
            elif cell in explored_set:
                kind = "explored"
            elif cell in grid.hotspots:
                kind = "hotspot"
            else:
                kind = "free"
            if not label and cell in grid.hotspots:
                label = "o"
            if cell == target and blocked_target:
                label = "🚧"
            if cell == robot:
                label = "🤖"
            bg, fg = CELL_STYLE[kind]
            ring = (f"box-shadow:0 0 0 3px {TARGET_RING} inset;" if cell == target
                    else "box-shadow:0 0 0 1px rgba(0,0,0,0.10) inset;")
            tds.append(
                f'<td style="width:{cell_px}px;height:{cell_px}px;min-width:{cell_px}px;'
                f"background:{bg};color:{fg};text-align:center;vertical-align:middle;"
                f"font-weight:700;font-size:15px;border-radius:6px;padding:0;border:none;"
                f'{ring}" title="{cell}">{label}</td>'
            )
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (
        '<div style="overflow-x:auto;padding:2px 0 6px 0">'
        '<table style="border-collapse:separate;border-spacing:3px;margin:0;border:none">'
        + "".join(rows) + "</table></div>"
    )


def legend_html(items: list[tuple[str, str]]) -> str:
    chips = []
    for kind, text in items:
        if kind == "target":
            style = f"background:#f1f2f6;color:#2d3436;box-shadow:0 0 0 3px {TARGET_RING} inset"
        else:
            bg, fg = CELL_STYLE[kind]
            style = f"background:{bg};color:{fg}"
        chips.append(f'<span style="{style};padding:3px 9px;border-radius:6px;'
                     f'font-size:12px;font-weight:600;white-space:nowrap">{text}</span>')
    return ('<div style="display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px 0">'
            + "".join(chips) + "</div>")


def badge(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:#ffffff;padding:6px 16px;'
            f'border-radius:999px;font-weight:800;font-size:1.1rem;letter-spacing:.03em">'
            f"{text}</span>")


# ---------------------------------------------------------------------------
# Cached computations (arguments are plain values so caching is reliable)
# ---------------------------------------------------------------------------

def make_grid(map_name: str, hotspot_risk: float):
    return dataclasses.replace(load_map(map_name), hotspot_risk=hotspot_risk)


def make_cfg(sens: float, fa: float, collision: float, gamma: float, mixing: float) -> SimConfig:
    return SimConfig(sensor=SensorModel(sens, fa), costs=Costs(collision=collision),
                     gamma=gamma, mixing=mixing)


@st.cache_data(show_spinner=False)
def cached_episode(map_name, hot, policy, seed, sens, fa, collision, gamma, mixing):
    return run_episode(make_grid(map_name, hot), policy, seed,
                       make_cfg(sens, fa, collision, gamma, mixing))


@st.cache_data(show_spinner=False)
def cached_compare(map_name, hot, sens, fa, collision, gamma, mixing, episodes):
    grid, cfg = make_grid(map_name, hot), make_cfg(sens, fa, collision, gamma, mixing)
    return [evaluate(grid, p, episodes, cfg) for p in POLICIES]


@st.cache_data(show_spinner=False)
def cached_sensitivity(map_name, hot, sens, fa, gamma, mixing, episodes, penalties):
    rows = []
    for pen in penalties:
        grid, cfg = make_grid(map_name, hot), make_cfg(sens, fa, pen, gamma, mixing)
        costs = {p: evaluate(grid, p, episodes, cfg)["avg_cost"] for p in POLICIES}
        rows.append({"penalty": pen, **costs})
    return rows


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    map_names = list(MAPS)
    map_name = st.selectbox("Campus map", map_names,
                            index=map_names.index("Campus 9x12") if "Campus 9x12" in MAPS else 0,
                            key="map")
    policies = list(POLICIES)
    policy = st.selectbox("Robot policy", policies, index=policies.index("bayes_mdp"),
                          format_func=POLICIES.get, key="policy",
                          help="'Bayes + MDP' is the full system; the others are baselines.")
    seed = int(st.number_input("Random seed (world)", min_value=0, max_value=100_000, value=5,
                               step=1, key="seed",
                               help="Same seed = same hidden obstacles for every policy."))
    with st.expander("⚙️ Advanced model parameters", expanded=False):
        st.caption("The defaults reproduce every number in the report. "
                   "Change them only to explore what-if scenarios.")
        sens = st.slider("Sensitivity  P(alert | blocked)", 0.50, 1.00, 0.90, 0.01, key="sens")
        fa = st.slider("False alarm  P(alert | clear)", 0.00, 0.50, 0.20, 0.01, key="fa")
        if sens <= fa:
            st.warning("Sensitivity ≤ false-alarm rate: the sensor carries no useful information.")
        hot = st.slider("Hotspot obstacle risk  P(blocked)", 0.05, 0.80, 0.30, 0.01, key="hot")
        collision = float(st.slider("Collision penalty", 0, 200, 50, 5, key="collision"))
        gamma = st.slider("Discount factor γ", 0.50, 0.99, 0.95, 0.01, key="gamma")
        mixing = st.slider("Obstacle change rate", 0.10, 1.00, 0.50, 0.05, key="mixing",
                           help="How quickly temporary obstacles appear / disappear per tick.")
    with st.expander("About"):
        st.markdown(
            "**Topic 1 – Search:** A* plans the shortest route.  \n"
            "**Topic 2 – Probabilistic reasoning:** a Bayes filter turns noisy sensor "
            "beeps into P(blocked).  \n"
            "**Topic 3 – Decision making:** an MDP solved with value iteration picks "
            "Move / Wait / Reroute."
        )

grid = make_grid(map_name, hot)
cfg = make_cfg(sens, fa, collision, gamma, mixing)

st.title("🤖 Campus Delivery Robot")
st.caption("A* search + Bayes filter + MDP value iteration")
st.markdown(
    "A delivery robot carries a package from the **main gate (S)** to the **dormitory (G)**. "
    "It plans its route with **A\\***, tracks uncertain obstacles with a **Bayes filter** and "
    "chooses **Move / Wait / Reroute** with an **MDP**."
)

tab_mission, tab_search, tab_bayes, tab_mdp, tab_exp = st.tabs(
    ["🚚 Mission", "🧭 A* Search", "📡 Bayes Filter", "🧠 MDP", "📊 Experiments"])


# ---------------------------------------------------------------------------
# Tab 1: Mission
# ---------------------------------------------------------------------------

def explain(s, policy_key: str) -> str:
    if s.target == grid.goal:
        where = "the goal"
    elif s.target in grid.hotspots:
        where = "a hotspot"
    else:
        where = "an ordinary cell"
    obs = f"The sensor {'beeped' if s.alert else 'stayed quiet'} at {where}"
    if abs(s.posterior - s.prior) < 0.005:
        belief = f"belief stays at {s.posterior:.2f}"
    else:
        verb = "rose" if s.posterior > s.prior else "fell"
        belief = f"belief {verb} from {s.prior:.2f} to {s.posterior:.2f}"
    a = s.action
    if policy_key == "naive":
        why = "the A*-only robot ignores the sensor and always moves"
    elif policy_key == "reactive":
        why = ("the reactive robot waits whenever the sensor beeps" if s.alert
               else "no beep, so the reactive robot moves")
    elif policy_key == "bayes":
        side = "above" if s.posterior > 0.5 else "below"
        why = f"belief is {side} the 0.5 threshold, so it {'waits' if a == WAIT else 'moves'}"
    else:
        q = s.scores
        if a == MOVE:
            why = "moving has the best expected value because the collision risk is small"
        elif a == WAIT:
            why = (f"waiting one tick (Q = {q[WAIT]:.2f}) is cheaper than risking a collision "
                   f"(Q(Move) = {q[MOVE]:.2f})")
        else:
            why = (f"the A* detour costs only +{s.detour_extra} steps, which beats waiting "
                   f"(Q = {q[WAIT]:.2f}) or risking a collision")
    sentence = f"{obs}: {belief}; {why}."
    if s.outcome == "collision":
        sentence += f" It bumped into an obstacle (penalty −{collision:g})."
    return sentence


def _shift_step(delta: int, n: int) -> None:
    st.session_state.step = min(max(st.session_state.step + delta, 1), n)


def _jump_to(step: int) -> None:
    st.session_state.step = step


def key_moments(log, limit: int = 4) -> list[tuple[int, str]]:
    """Steps worth showing in a demo: every non-trivial decision, then the finish."""
    moments = []
    for i, s in enumerate(log, 1):
        if s.alert or s.outcome != "moved":
            moments.append((i, f"t={s.tick} · {s.action}"
                               + (" 💥" if s.outcome == "collision" else "")))
    moments = moments[:limit]
    moments.append((len(log), "🏁 Finish"))
    return moments


with tab_mission:
    ep = cached_episode(map_name, hot, policy, seed, sens, fa, collision, gamma, mixing)
    n = len(ep.log)

    m = st.columns(6)
    m[0].metric("Delivered", "Yes ✅" if ep.delivered else "No ❌")
    m[1].metric("Time (ticks)", ep.ticks)
    m[2].metric("Collisions", ep.collisions)
    m[3].metric("Waits", ep.waits)
    m[4].metric("Reroutes", ep.reroutes)
    m[5].metric("Total cost", f"{ep.total_cost:g}", help="ticks + collision penalty × collisions")

    if n == 0:
        st.info("The robot made no decisions in this episode.")
    else:
        signature = (map_name, hot, policy, seed, sens, fa, collision, gamma, mixing)
        if st.session_state.get("episode_sig") != signature:
            st.session_state.episode_sig = signature
            st.session_state.step = 1
        st.session_state.step = min(max(int(st.session_state.get("step", 1)), 1), n)

        moments = key_moments(ep.log)
        st.markdown("**Key moments** — click one to jump straight to that decision, "
                    "or use **Next ▶** to step through the whole delivery:")
        cols = st.columns(len(moments))
        for col, (step_no, label) in zip(cols, moments):
            col.button(label, key=f"moment_{step_no}", on_click=_jump_to, args=(step_no,),
                       type="primary" if st.session_state.step == step_no else "secondary",
                       width="stretch")

        left, right = st.columns([1.15, 1], gap="large")
        with left:
            if n > 1:
                b1, b2, b3 = st.columns([1, 4, 1], vertical_alignment="bottom")
                b1.button("◀ Prev", key="prev", on_click=_shift_step, args=(-1, n),
                          disabled=st.session_state.step <= 1)
                b2.slider("Timeline — decision step", 1, n, key="step")
                b3.button("Next ▶", key="next", on_click=_shift_step, args=(1, n),
                          disabled=st.session_state.step >= n)
            k = st.session_state.step
            s = ep.log[k - 1]
            route_before = ep.log[k - 2].route if k >= 2 else ep.initial_route
            show_truth = st.toggle("Show ground truth (hidden from the robot)", key="truth")
            st.markdown(grid_html(grid, route=route_before, robot=s.position, target=s.target,
                                  blocked_target=show_truth and s.truly_blocked),
                        unsafe_allow_html=True)
            st.markdown(legend_html([("start", "S start"), ("goal", "G goal"),
                                     ("wall", "building"), ("hotspot", "o hotspot"),
                                     ("route", "planned route"), ("route_hot", "hotspot on route"),
                                     ("target", "cell being sensed")]),
                        unsafe_allow_html=True)

        with right:
            with st.container(border=True):
                kind = ("hotspot" if s.target in grid.hotspots else
                        "goal" if s.target == grid.goal else "ordinary cell")
                st.markdown(f"#### Step {k} of {n} · tick {s.tick}")
                st.caption(f"Robot at {s.position} → next cell {s.target} ({kind})")
                c1, c2 = st.columns(2)
                c1.metric("Sensor", "🔔 ALERT" if s.alert else "🔕 quiet")
                c2.metric("P(blocked)", f"{s.posterior:.2f}",
                          delta=f"{s.posterior - s.prior:+.2f} from prior {s.prior:.2f}",
                          delta_color="inverse")
                st.markdown(
                    f"Decision: {badge(s.action.upper(), ACTION_COLORS[s.action])}"
                    f"&nbsp;&nbsp; → {OUTCOME_TEXT[s.outcome]}", unsafe_allow_html=True)
                st.write("")
                if s.scores:
                    qdf = pd.DataFrame({"action": list(s.scores), "Q": list(s.scores.values())})
                    chart = (alt.Chart(qdf).mark_bar(cornerRadius=4).encode(
                        x=alt.X("Q:Q", title="Expected value Q(b, a)  (higher is better)"),
                        y=alt.Y("action:N", sort=list(s.scores), title=None),
                        color=alt.Color("action:N", legend=None, scale=alt.Scale(
                            domain=list(ACTION_COLORS), range=list(ACTION_COLORS.values()))),
                        tooltip=["action", alt.Tooltip("Q:Q", format=".2f")],
                    ).properties(height=120))
                    st.altair_chart(chart, width="stretch")
                else:
                    st.caption("This baseline does not use the MDP; it follows a fixed rule.")
                if s.detour_extra is not None:
                    st.caption(f"A* detour around {s.target}: +{s.detour_extra} extra steps.")
                else:
                    st.caption(f"No A* detour exists around {s.target}.")
                if show_truth:
                    st.markdown(f"**Ground truth:** the cell was actually "
                                f"**{'BLOCKED 🚧' if s.truly_blocked else 'clear'}**.")
                st.info(explain(s, policy))

        with st.expander("Full decision log"):
            st.dataframe(pd.DataFrame([{
                "tick": x.tick, "at": str(x.position), "next": str(x.target),
                "sensor": "ALERT" if x.alert else "quiet", "prior": round(x.prior, 3),
                "posterior": round(x.posterior, 3), "action": x.action, "outcome": x.outcome,
                "truth": "blocked" if x.truly_blocked else "clear",
            } for x in ep.log]), hide_index=True)

    other = "naive" if policy != "naive" else "bayes_mdp"
    if st.checkbox(f"Compare with “{POLICIES[other]}” on the same world", key="compare_naive"):
        ep2 = cached_episode(map_name, hot, other, seed, sens, fa, collision, gamma, mixing)
        st.dataframe(pd.DataFrame([{
            "policy": POLICIES[e.policy], "delivered": e.delivered, "ticks": e.ticks,
            "collisions": e.collisions, "waits": e.waits, "reroutes": e.reroutes,
            "total cost": e.total_cost} for e in (ep, ep2)]), hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: A* search
# ---------------------------------------------------------------------------

with tab_search:
    st.markdown("A* expands the node with the lowest $f(n) = g(n) + h(n)$: cost so far plus "
                "the Manhattan-distance estimate of the cost to go.")
    st.latex(r"f(n) = g(n) + h(n), \qquad h(n) = |r_n - r_G| + |c_n - c_G|")
    free = grid.free_cells()

    def cell_name(c):
        return f"{c}  (S)" if c == grid.start else f"{c}  (G)" if c == grid.goal else str(c)

    c1, c2, c3 = st.columns([1, 1, 2])
    start = c1.selectbox("Start", free, index=free.index(grid.start), format_func=cell_name,
                         key=f"search_start_{map_name}")
    goal = c2.selectbox("Goal", free, index=free.index(grid.goal), format_func=cell_name,
                        key=f"search_goal_{map_name}")
    shown = c3.radio("Show on map", list(ALGORITHMS), horizontal=True, key="search_alg")

    results = {name: fn(grid, start, goal) for name, fn in ALGORITHMS.items()}
    finite = [r.cost for r in results.values() if r.found]
    best_cost = min(finite) if finite else None
    left, right = st.columns([1.15, 1], gap="large")
    with right:
        st.dataframe(pd.DataFrame([{
            "algorithm": name,
            "path length": int(r.cost) if r.found else None,
            "nodes expanded": r.expanded,
            "optimal": ("✅" if r.found and r.cost == best_cost else "❌") if r.found else "—",
        } for name, r in results.items()]), hide_index=True)
        st.caption("A* and UCS/BFS all find the shortest path, but A* expands far fewer nodes "
                   "thanks to the heuristic. Greedy ignores g(n), so it is not guaranteed "
                   "to be optimal.")
    with left:
        res = results[shown]
        if not res.found:
            st.warning(f"No path from {start} to {goal}: the goal is unreachable.")
        st.markdown(grid_html(grid, route=res.path or [], explored=res.explored, start=start,
                              goal=goal), unsafe_allow_html=True)
        st.markdown(legend_html([("start", "S start"), ("goal", "G goal"), ("wall", "building"),
                                 ("explored", "expanded node"), ("route", "path"),
                                 ("hotspot", "o hotspot")]), unsafe_allow_html=True)
        if res.found:
            st.caption(f"{shown}: path of {int(res.cost)} steps, {res.expanded} nodes expanded.")


# ---------------------------------------------------------------------------
# Tab 3: Bayes filter
# ---------------------------------------------------------------------------

with tab_bayes:
    st.latex(r"P(B \mid e) = \frac{P(e \mid B)\,P(B)}{P(e \mid B)\,P(B) + "
             r"P(e \mid \neg B)\,(1 - P(B))}")
    c1, c2, c3 = st.columns(3)
    b_prior = c1.slider("Prior P(blocked)", 0.0, 1.0, float(hot), 0.01, key="bayes_prior")
    b_sens = c2.slider("P(alert | blocked)", 0.0, 1.0, float(sens), 0.01, key="bayes_sens")
    b_fa = c3.slider("P(alert | clear)", 0.0, 1.0, float(fa), 0.01, key="bayes_fa")
    c4, c5 = st.columns([2, 1])
    text = c4.text_input("Sensor readings (A = alert / beep, Q = quiet; up to 10)", "A A Q",
                         key="bayes_readings")
    use_dyn = c5.toggle("Obstacle may change between readings (time update)",
                        key="bayes_dyn")

    tokens = [t for t in re.split(r"[\s,;]+", text.strip()) if t]
    bad = [t for t in tokens if t.upper() not in {"A", "Q", "ALERT", "QUIET"}]
    if not tokens:
        st.info("Enter at least one reading, e.g. `A Q A`.")
    elif bad:
        st.error(f"Unknown readings: {', '.join(bad)}. Use A (alert) or Q (quiet).")
    elif len(tokens) > 10:
        st.error("Please enter at most 10 readings.")
    else:
        readings = [t.upper().startswith("A") for t in tokens]
        sensor = SensorModel(b_sens, b_fa)
        dyn = ObstacleDynamics.from_risk(b_prior, mixing)
        rows, belief, trace = [], b_prior, [{"step": 0, "belief": b_prior}]
        try:
            for i, alert in enumerate(readings, start=1):
                before = predict(belief, dyn) if (use_dyn and i > 1) else belief
                l_b, l_c = sensor.likelihood(alert, True), sensor.likelihood(alert, False)
                evidence = l_b * before + l_c * (1 - before)
                belief = bayes_update(before, alert, sensor)
                rows.append({"#": i, "reading": "ALERT" if alert else "quiet",
                             "prior": round(before, 4), "P(e|blocked)": l_b,
                             "P(e|clear)": l_c, "P(e)": round(evidence, 4),
                             "posterior": round(belief, 4)})
                trace.append({"step": i, "belief": belief})
        except ValueError as err:
            st.error(f"Impossible observation under this sensor model: {err}")
        if rows:
            left, right = st.columns([1.2, 1], gap="large")
            left.dataframe(pd.DataFrame(rows), hide_index=True)
            r0 = rows[0]
            left.caption(f"First update: {r0['P(e|blocked)']:.2f} × {r0['prior']:.2f} / "
                         f"{r0['P(e)']:.4f} = {r0['posterior']:.3f}")
            chart = (alt.Chart(pd.DataFrame(trace)).mark_line(point=True, color="#2471a3")
                     .encode(x=alt.X("step:O", title="reading #  (0 = prior)"),
                             y=alt.Y("belief:Q", title="P(blocked)",
                                     scale=alt.Scale(domain=[0, 1])),
                             tooltip=["step", alt.Tooltip("belief:Q", format=".3f")])
                     .properties(height=260))
            right.altair_chart(chart, width="stretch")
            if use_dyn:
                st.caption(f"Time update between readings: P(B) ← P(B)(1 − {dyn.p_clear:.2f}) "
                           f"+ (1 − P(B))·{dyn.p_block:.2f}")


# ---------------------------------------------------------------------------
# Tab 4: MDP
# ---------------------------------------------------------------------------

with tab_mdp:
    st.markdown(
        "**States:** the robot's belief P(blocked) = 0.00, 0.01, …, 1.00 (plus terminal "
        "*Passed*).  **Actions:** Move, Wait, Reroute.  \n"
        f"**Rewards:** −1 per tick, −{collision:g} for a collision, −(1 + extra + 1) for a "
        "detour.  **Transitions:** the Bayes filter — after a tick the sensor beeps or not, "
        "which moves the belief.")
    st.latex(r"V(b) \leftarrow \max_a \Big[ R(b,a) + \gamma \sum_{b'} P(b' \mid b, a)\, V(b') "
             r"\Big]")
    c1, c2, c3 = st.columns(3)
    cell_kind = c1.radio("Cell type", ["Hotspot", "Ordinary cell"], horizontal=True,
                         key="mdp_cell")
    extra = c2.selectbox("A* detour", [None, 0, 2, 4, 6], index=2, key="mdp_extra",
                         format_func=lambda e: "no detour" if e is None else f"+{e} steps")
    b_now = c3.slider("Current belief P(blocked)", 0.0, 1.0, 0.66, 0.01, key="mdp_belief")

    risk = grid.hotspot_risk if cell_kind == "Hotspot" else grid.normal_risk
    enc = Encounter(ObstacleDynamics.from_risk(risk, mixing), cfg.sensor, cfg.costs, extra, gamma)
    vi = solve(enc)
    scores = action_values(b_now, enc)
    best = max(scores, key=scores.get)
    threshold = move_threshold(enc)

    m = st.columns(4)
    m[0].metric("Value-iteration sweeps", vi.iterations)
    m[1].metric("Move is best while P(blocked) <", f"{threshold:.3f}")
    m[2].metric(f"Best action at b = {b_now:.2f}", best)
    m[3].metric(f"V(b = {b_now:.2f})", f"{scores[best]:.2f}")

    beliefs = [i / 100 for i in range(101)]
    curve = pd.DataFrame([{"belief": b, "action": a, "Q": q}
                          for b in beliefs for a, q in action_values(b, enc).items()])
    lines = alt.Chart(curve).mark_line(strokeWidth=3).encode(
        x=alt.X("belief:Q", title="belief P(blocked)"),
        y=alt.Y("Q:Q", title="Q(b, a)"),
        color=alt.Color("action:N", scale=alt.Scale(domain=list(ACTION_COLORS),
                                                    range=list(ACTION_COLORS.values()))),
        tooltip=["action", alt.Tooltip("belief:Q", format=".2f"),
                 alt.Tooltip("Q:Q", format=".2f")])
    rule = alt.Chart(pd.DataFrame({"belief": [b_now]})).mark_rule(
        strokeDash=[4, 4], color="#888888").encode(x="belief:Q")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Q-values vs belief** (dashed line = current belief)")
        st.altair_chart((lines + rule).properties(height=300), width="stretch")
    with right:
        st.markdown("**Value-iteration convergence**")
        deltas = pd.DataFrame({"sweep": range(1, len(vi.deltas) + 1), "max |ΔV|": vi.deltas})
        deltas = deltas[deltas["max |ΔV|"] > 0]
        st.altair_chart(alt.Chart(deltas).mark_line(point=True, color="#8e44ad").encode(
            x=alt.X("sweep:Q"), y=alt.Y("max |ΔV|:Q", scale=alt.Scale(type="log")),
            tooltip=["sweep", alt.Tooltip("max |ΔV|:Q", format=".2e")]
        ).properties(height=300), width="stretch")

    # Optimal policy as belief ranges
    best_by_belief = curve.loc[curve.groupby("belief")["Q"].idxmax()].sort_values("belief")
    segments, current, lo, prev_b = [], None, 0.0, 0.0
    for b, a in zip(best_by_belief["belief"], best_by_belief["action"]):
        if a != current:
            if current is not None:
                segments.append((lo, prev_b, current))
            current, lo = a, b
        prev_b = b
    segments.append((lo, 1.0, current))
    st.markdown("**Optimal policy:** " + " · ".join(
        f"{badge(a, ACTION_COLORS[a])} for b ∈ [{a_lo:.2f}, {a_hi:.2f}]"
        for a_lo, a_hi, a in segments), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 5: Experiments
# ---------------------------------------------------------------------------

with tab_exp:
    st.markdown("Every policy is evaluated on **the same random worlds** (same seeds), so the "
                "comparison is fair. **Cost = delivery time + collision penalty × collisions** "
                "(lower is better).")
    episodes = st.slider("Episodes per policy", 50, 1000, 200, 50, key="episodes")
    params = (map_name, hot, sens, fa, collision, gamma, mixing, episodes)

    if st.button("▶ Run policy comparison", type="primary", key="run_compare"):
        with st.spinner(f"Simulating {episodes * len(POLICIES)} deliveries…"):
            st.session_state.compare = (params, cached_compare(*params))
    if "compare" in st.session_state:
        saved_params, rows = st.session_state.compare
        if saved_params != params:
            st.caption("Settings changed since this run — press the button to refresh.")
        df = pd.DataFrame([{
            "policy": POLICIES[r["policy"]],
            "avg cost": round(r["avg_cost"], 2),
            "avg ticks": round(r["avg_ticks"], 2),
            "avg collisions": round(r["avg_collisions"], 3),
            "collision-free %": round(100 * r["collision_free_rate"], 1),
            "avg waits": round(r["avg_waits"], 2),
            "avg reroutes": round(r["avg_reroutes"], 2),
            "delivered %": round(100 * r["success_rate"], 1),
        } for r in rows])
        left, right = st.columns([1.3, 1], gap="large")
        left.dataframe(df, hide_index=True)
        best_row = df.loc[df["avg cost"].idxmin()]
        left.success(f"Lowest cost: **{best_row['policy']}** ({best_row['avg cost']:.2f})")
        right.altair_chart(alt.Chart(df).mark_bar(cornerRadius=4).encode(
            x=alt.X("avg cost:Q", title="average cost (lower is better)"),
            y=alt.Y("policy:N", sort=None, title=None),
            color=alt.condition(alt.datum.policy == POLICIES["bayes_mdp"],
                                alt.value("#2471a3"), alt.value("#95a5a6")),
            tooltip=["policy", "avg cost", "avg collisions", "avg ticks"],
        ).properties(height=220), width="stretch")

    st.divider()
    st.markdown("**Sensitivity to the collision penalty** — which policy is best when "
                "collisions are cheap or very expensive?")
    penalties = (5, 20, 50, 100)
    sens_params = (map_name, hot, sens, fa, gamma, mixing, episodes, penalties)
    if st.button("▶ Run penalty sensitivity", key="run_sens",
                 help=f"{len(penalties) * len(POLICIES) * episodes} simulated deliveries"):
        with st.spinner("Running sensitivity study…"):
            st.session_state.sensitivity = (sens_params, cached_sensitivity(*sens_params))
    if "sensitivity" in st.session_state:
        saved_params, rows = st.session_state.sensitivity
        if saved_params != sens_params:
            st.caption("Settings changed since this run — press the button to refresh.")
        table = pd.DataFrame([{
            "collision penalty": r["penalty"],
            **{POLICIES[p]: round(r[p], 2) for p in POLICIES},
            "best policy": POLICIES[min(POLICIES, key=lambda p: r[p])],
        } for r in rows])
        st.dataframe(table, hide_index=True)
        long = pd.DataFrame([{"penalty": r["penalty"], "policy": POLICIES[p], "avg cost": r[p]}
                             for r in rows for p in POLICIES])
        st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("penalty:Q", title="collision penalty"),
            y=alt.Y("avg cost:Q", title="average cost"),
            color=alt.Color("policy:N", title=None),
            tooltip=["policy", "penalty", alt.Tooltip("avg cost:Q", format=".2f")],
        ).properties(height=300), width="stretch")
