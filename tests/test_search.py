"""Tests for topic 1: A* search (and the baseline searches)."""

import pytest

from robot.grid import Grid, load_map, random_grid
from robot.search import ALGORITHMS, astar, bfs, greedy, manhattan, ucs


def assert_valid_path(grid, path, start, goal, blocked=frozenset()):
    assert path[0] == start and path[-1] == goal
    for a, b in zip(path, path[1:]):
        assert manhattan(a, b) == 1, f"non-adjacent step {a} -> {b}"
        assert grid.passable(b, blocked), f"path goes through blocked cell {b}"
    assert len(set(path)) == len(path), "path revisits a cell"


def test_classic_map_reproduces_version_1_result():
    grid = load_map("Classic 5x5")
    result = astar(grid)
    assert result.cost == 8                      # = Manhattan lower bound, so optimal
    assert result.path[:5] == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    assert_valid_path(grid, result.path, grid.start, grid.goal)


@pytest.mark.parametrize("name", ["Classic 5x5", "Campus 9x12"])
def test_all_algorithms_return_valid_paths(name):
    grid = load_map(name)
    for algorithm in ALGORITHMS.values():
        r = algorithm(grid)
        assert r.found
        assert_valid_path(grid, r.path, grid.start, grid.goal)
        assert r.cost == len(r.path) - 1


@pytest.mark.parametrize("seed", range(150))
def test_astar_is_optimal_on_random_grids(seed):
    """A* must match BFS (exact shortest path for unit costs) on every map."""
    grid = random_grid(12, 12, wall_prob=0.28, seed=seed)
    a, b = astar(grid), bfs(grid)
    assert a.found == b.found
    if a.found:
        assert a.cost == b.cost
        assert_valid_path(grid, a.path, grid.start, grid.goal)
        assert a.expanded <= ucs(grid).expanded   # the heuristic only ever helps


def test_greedy_can_be_suboptimal():
    """Greedy ignores g(n) and follows h(n) into a longer route; A* does not."""
    grid = Grid.from_ascii("""
        S . . . # # .
        . . . # . . #
        . . # . . . .
        . # . . # . .
        . . . # . . .
        . . . # . # .
        # . # . . . G
    """)
    assert astar(grid).cost == 16
    assert greedy(grid).cost == 18


def test_heuristic_is_admissible_everywhere():
    """h(n) <= true distance for every reachable cell (checked with BFS)."""
    grid = load_map("Campus 9x12")
    for cell in grid.free_cells():
        true = bfs(grid, cell, grid.goal)
        if true.found:
            assert manhattan(cell, grid.goal) <= true.cost


def test_unreachable_goal():
    grid = Grid.from_ascii("""
        S . #
        . # .
        # . G
    """)
    for algorithm in ALGORITHMS.values():
        r = algorithm(grid)
        assert not r.found and r.path is None and r.cost == float("inf")


def test_blocked_cells_are_avoided_and_detour_is_longer():
    grid = load_map("Classic 5x5")
    direct = astar(grid)
    suspicious = direct.path[4]
    detour = astar(grid, blocked={suspicious})
    assert suspicious not in detour.path
    assert detour.cost >= direct.cost
    assert_valid_path(grid, detour.path, grid.start, grid.goal, {suspicious})


def test_start_equals_goal():
    grid = load_map("Classic 5x5")
    r = astar(grid, start=(2, 2), goal=(2, 2))
    assert r.path == [(2, 2)] and r.cost == 0
