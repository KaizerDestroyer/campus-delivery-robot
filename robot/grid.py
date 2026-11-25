"""Campus map: a 4-connected grid parsed from ASCII art.

Legend
------
    S  start (main gate)          #  wall / building (never passable)
    G  goal (dormitory)           o  hotspot: busy area where temporary
    .  free path                     obstacles (people, carts) are likely
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

Cell = tuple[int, int]

WALL, FREE, HOTSPOT, START, GOAL = "#", ".", "o", "S", "G"
_ALLOWED = {WALL, FREE, HOTSPOT, START, GOAL}

MOVES: tuple[Cell, ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))  # right, down, left, up


@dataclass(frozen=True)
class Grid:
    """Static campus map plus the prior risk of temporary obstacles."""

    rows: int
    cols: int
    walls: frozenset[Cell]
    hotspots: frozenset[Cell]
    start: Cell
    goal: Cell
    hotspot_risk: float = 0.30  # P(blocked) for a hotspot cell
    normal_risk: float = 0.02   # P(blocked) for an ordinary free cell

    @classmethod
    def from_ascii(cls, text: str, **kwargs: float) -> "Grid":
        """Parse a map written with the legend above (spaces are ignored)."""
        lines = [line.replace(" ", "") for line in text.strip().splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            raise ValueError("map is empty")
        width = len(lines[0])
        if any(len(line) != width for line in lines):
            raise ValueError("map must be rectangular")

        walls, hotspots, starts, goals = set(), set(), [], []
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch not in _ALLOWED:
                    raise ValueError(f"unknown map symbol {ch!r} at {(r, c)}")
                if ch == WALL:
                    walls.add((r, c))
                elif ch == HOTSPOT:
                    hotspots.add((r, c))
                elif ch == START:
                    starts.append((r, c))
                elif ch == GOAL:
                    goals.append((r, c))
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("map needs exactly one S and one G")
        return cls(len(lines), width, frozenset(walls), frozenset(hotspots),
                   starts[0], goals[0], **kwargs)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, cell: Cell, blocked: frozenset[Cell] | set[Cell] = frozenset()) -> bool:
        return self.in_bounds(cell) and cell not in self.walls and cell not in blocked

    def neighbors(self, cell: Cell,
                  blocked: frozenset[Cell] | set[Cell] = frozenset()) -> Iterator[Cell]:
        """Up to four passable neighbours (no diagonal moves)."""
        r, c = cell
        for dr, dc in MOVES:
            nxt = (r + dr, c + dc)
            if self.passable(nxt, blocked):
                yield nxt

    def risk(self, cell: Cell) -> float:
        """Prior probability that `cell` is blocked by a temporary obstacle."""
        if cell in (self.start, self.goal) or cell in self.walls:
            return 0.0
        return self.hotspot_risk if cell in self.hotspots else self.normal_risk

    def free_cells(self) -> list[Cell]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in self.walls]

    def render(self, path: list[Cell] | None = None, robot: Cell | None = None) -> str:
        """Plain-text picture of the map (used by the CLI)."""
        on_path = set(path or [])
        out = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cell = (r, c)
                if cell == robot:
                    ch = "R"
                elif cell in self.walls:
                    ch = "#"
                elif cell == self.start:
                    ch = "S"
                elif cell == self.goal:
                    ch = "G"
                elif cell in on_path:
                    ch = "*"
                elif cell in self.hotspots:
                    ch = "o"
                else:
                    ch = "."
                row.append(ch)
            out.append(" ".join(row))
        return "\n".join(out)


# The original 5x5 map from version 1 of the project, plus one hotspot on the route.
CLASSIC_MAP = """
S . . . .
. # # . .
. . . . o
. . . # .
. . . . G
"""

# A bigger campus: buildings (#), busy hotspots (o) near the cafeteria,
# library and sports hall, main gate S and dormitory G.
CAMPUS_MAP = """
S . . . . # . . . . . .
. # # # . # . # # # # .
. # # # . . o . . . # .
. . . o . # . # # . . .
# # . # . # . # # o # #
. . . . o . . . . . . .
. # # # . # # # # . # .
. # # # . . . o . . # .
. . . . . # # . # . o G
"""

MAPS: dict[str, str] = {
    "Campus 9x12": CAMPUS_MAP,
    "Classic 5x5": CLASSIC_MAP,
}


def load_map(name: str, **kwargs: float) -> Grid:
    """Load one of the built-in maps by name."""
    if name not in MAPS:
        raise KeyError(f"unknown map {name!r}; choose from {list(MAPS)}")
    return Grid.from_ascii(MAPS[name], **kwargs)


def random_grid(rows: int, cols: int, wall_prob: float, seed: int,
                hotspot_prob: float = 0.1) -> Grid:
    """Random map with S in the top-left and G in the bottom-right corner.

    The goal may be unreachable -- callers that need a path should check.
    Used by tests and experiments.
    """
    import random

    rng = random.Random(seed)
    start, goal = (0, 0), (rows - 1, cols - 1)
    walls, hotspots = set(), set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in (start, goal):
                continue
            x = rng.random()
            if x < wall_prob:
                walls.add((r, c))
            elif x < wall_prob + hotspot_prob:
                hotspots.add((r, c))
    return Grid(rows, cols, frozenset(walls), frozenset(hotspots), start, goal)
