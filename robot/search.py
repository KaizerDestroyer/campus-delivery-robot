"""Topic 1 -- Search: A* route planning (with BFS, UCS and Greedy for comparison).

A* expands the frontier node with the lowest

    f(n) = g(n) + h(n)

where g(n) is the cost already paid from the start and h(n) is a heuristic
estimate of the remaining cost. With the Manhattan distance on a
4-connected grid with unit step costs, h never overestimates (admissible)
and h(n) <= 1 + h(n') for neighbours (consistent), so A* returns an
optimal path and never needs to re-open a closed node.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import inf
from typing import Callable

from .grid import Cell, Grid

Blocked = frozenset[Cell] | set[Cell]


@dataclass
class SearchResult:
    algorithm: str
    path: list[Cell] | None          # start ... goal, or None if unreachable
    cost: float                      # number of steps (inf if unreachable)
    expanded: int                    # nodes taken off the frontier and expanded
    explored: list[Cell] = field(default_factory=list)  # expansion order (for the UI)

    @property
    def found(self) -> bool:
        return self.path is not None


def manhattan(a: Cell, b: Cell) -> int:
    """|dr| + |dc| -- admissible and consistent for 4-connected unit-cost moves."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct(came_from: dict[Cell, Cell | None], goal: Cell) -> list[Cell]:
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    return path[::-1]


def _best_first(name: str, grid: Grid, start: Cell, goal: Cell, blocked: Blocked,
                priority: Callable[[int, int], int]) -> SearchResult:
    """Generic best-first graph search; `priority(g, h)` decides the strategy."""
    if not grid.passable(start, blocked) or not grid.passable(goal, blocked):
        return SearchResult(name, None, inf, 0)
    tie = itertools.count()  # FIFO tie-break so heap never compares cells
    h0 = manhattan(start, goal)
    frontier = [(priority(0, h0), h0, next(tie), 0, start)]
    came_from: dict[Cell, Cell | None] = {start: None}
    best_g = {start: 0}
    closed: set[Cell] = set()
    explored: list[Cell] = []

    while frontier:
        _, _, _, g, node = heappop(frontier)
        if node in closed:
            continue  # stale entry: a cheaper copy was already expanded
        closed.add(node)
        explored.append(node)
        if node == goal:
            return SearchResult(name, _reconstruct(came_from, goal), g, len(explored), explored)
        for nxt in grid.neighbors(node, blocked):
            ng = g + 1
            if nxt not in closed and ng < best_g.get(nxt, inf):
                best_g[nxt] = ng
                came_from[nxt] = node
                h = manhattan(nxt, goal)
                heappush(frontier, (priority(ng, h), h, next(tie), ng, nxt))
    return SearchResult(name, None, inf, len(explored), explored)


def astar(grid: Grid, start: Cell | None = None, goal: Cell | None = None,
          blocked: Blocked = frozenset()) -> SearchResult:
    """A*: f = g + h. Optimal with an admissible heuristic."""
    return _best_first("A*", grid, start or grid.start, goal or grid.goal, blocked,
                       lambda g, h: g + h)


def ucs(grid: Grid, start: Cell | None = None, goal: Cell | None = None,
        blocked: Blocked = frozenset()) -> SearchResult:
    """Uniform-cost search (Dijkstra): f = g. Optimal but uninformed."""
    return _best_first("UCS (Dijkstra)", grid, start or grid.start, goal or grid.goal,
                       blocked, lambda g, h: g)


def greedy(grid: Grid, start: Cell | None = None, goal: Cell | None = None,
           blocked: Blocked = frozenset()) -> SearchResult:
    """Greedy best-first: f = h. Fast but NOT guaranteed optimal."""
    return _best_first("Greedy", grid, start or grid.start, goal or grid.goal, blocked,
                       lambda g, h: h)


def bfs(grid: Grid, start: Cell | None = None, goal: Cell | None = None,
        blocked: Blocked = frozenset()) -> SearchResult:
    """Breadth-first search: optimal for unit step costs, explores in rings."""
    start, goal = start or grid.start, goal or grid.goal
    if not grid.passable(start, blocked) or not grid.passable(goal, blocked):
        return SearchResult("BFS", None, inf, 0)
    came_from: dict[Cell, Cell | None] = {start: None}
    queue, explored = deque([start]), []
    while queue:
        node = queue.popleft()
        explored.append(node)
        if node == goal:
            path = _reconstruct(came_from, goal)
            return SearchResult("BFS", path, len(path) - 1, len(explored), explored)
        for nxt in grid.neighbors(node, blocked):
            if nxt not in came_from:
                came_from[nxt] = node
                queue.append(nxt)
    return SearchResult("BFS", None, inf, len(explored), explored)


ALGORITHMS: dict[str, Callable[..., SearchResult]] = {
    "A*": astar,
    "UCS (Dijkstra)": ucs,
    "BFS": bfs,
    "Greedy": greedy,
}
