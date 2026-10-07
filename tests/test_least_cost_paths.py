"""Integration tests for find_least_action_path and find_minimum_ascent_path (compiled extension)."""

from __future__ import annotations

import numpy as np
import pytest

from pes_analyzer.grid import path_length
from pes_analyzer.topology import (
    find_least_action_path,
    find_minimax_path,
    find_minimum_ascent_path,
    find_steepest_descent_path,
)

nan = np.nan


def ascent_along(E, path):
    e = np.array([E[tuple(p)] for p in path])
    return float(np.clip(np.diff(e), 0, None).sum())


def action_along(C, path, axes=None):
    c = np.array([C[tuple(p)] for p in path])
    d = np.diff(path_length(np.asarray(path), axes))
    return float((0.5 * (c[1:] + c[:-1]) * d).sum())


# -------- least action ------------------------------------------------------


def test_least_action_takes_the_cheap_detour_and_output_contract():
    cost = np.array([[1.0, 1.0, 1.0], [1.0, 100.0, 1.0], [2.0, 2.0, 2.0]])
    idx, action = find_least_action_path(cost, (1, 0), (1, 2), neighborhood="von_neumann")
    assert idx.dtype == np.int64 and action.dtype == np.float64
    assert idx.tolist() == [[1, 0], [0, 0], [0, 1], [0, 2], [1, 2]]
    assert action.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_unit_cost_measures_length_in_both_stencils_and_with_axes():
    ones = np.ones((3, 4))
    idx, action = find_least_action_path(ones, (0, 0), (2, 3), neighborhood="von_neumann")
    assert action[-1] == 5.0 and len(idx) == 6
    idx, action = find_least_action_path(ones, (0, 0), (2, 3))          # Moore default
    assert action[-1] == pytest.approx(2 * 2**0.5 + 1) and len(idx) == 4
    axes = [np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0, 3.0, 7.0])]
    idx, action = find_least_action_path(ones, (0, 0), (2, 3), axes=axes, neighborhood="von_neumann")
    assert action[-1] == pytest.approx(10.0)                              # Manhattan distance in coordinates
    assert action[-1] == pytest.approx(path_length(idx, axes)[-1])


def test_mask_targets_and_end_forms():
    chain = np.ones((1, 6))
    mask = np.zeros((1, 6), dtype=bool)
    mask[0, 0] = mask[0, 5] = True
    idx, action = find_least_action_path(chain, (0, 2), mask, neighborhood="von_neumann")
    assert idx.tolist() == [[0, 2], [0, 1], [0, 0]] and action.tolist() == [0.0, 1.0, 2.0]
    only = np.zeros((1, 6), dtype=bool)
    only[0, 2] = True
    idx, action = find_least_action_path(chain, (0, 2), only)
    assert idx.tolist() == [[0, 2]] and action.tolist() == [0.0]
    # an index given as a list or an integer ndarray is an index, not a mask
    assert find_least_action_path(chain, (0, 2), [0, 5])[0].tolist()[-1] == [0, 5]
    assert find_least_action_path(chain, (0, 2), np.array([0, 5]))[0].tolist()[-1] == [0, 5]
    # start equal to a single-cell end
    assert find_least_action_path(chain, (0, 2), (0, 2))[1].tolist() == [0.0]


def test_unreachable_target_returns_none():
    wall = np.array([[1.0, nan, 1.0], [1.0, nan, 1.0]])
    assert find_least_action_path(wall, (0, 0), (0, 2)) is None
    mask = np.zeros((2, 3), dtype=bool)
    mask[:, 2] = True
    assert find_least_action_path(wall, (0, 0), mask) is None


# -------- minimum ascent ----------------------------------------------------


def test_ascent_counts_climbs_only():
    e = np.array([[0.0, 3.0, 1.0, 2.0, 0.0]])
    idx, climb = find_minimum_ascent_path(e, (0, 0), (0, 4))
    assert idx[:, 1].tolist() == [0, 1, 2, 3, 4] and climb.tolist() == [0.0, 3.0, 3.0, 4.0, 4.0]
    d = np.array([[0.0, 5.0, 0.0], [0.0, 1.0, 0.0]])
    idx, climb = find_minimum_ascent_path(d, (0, 0), (0, 2))
    assert idx.tolist() == [[0, 0], [1, 0], [1, 1], [1, 2], [0, 2]] and climb.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_zero_ascent_region_returns_the_shortest_path():
    flat = np.zeros((5, 5))
    idx, climb = find_minimum_ascent_path(flat, (2, 0), (2, 4))
    assert idx.tolist() == [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]] and climb[-1] == 0.0
    idx, _ = find_minimum_ascent_path(np.zeros((3, 3)), (0, 0), (2, 2), neighborhood="moore")
    assert idx.tolist() == [[0, 0], [1, 1], [2, 2]]


def test_rounding_limit_of_the_tie_break_is_as_documented():
    e = np.array([[0.0, 0.0, 0.0, nan], [0.0, 2.0**-54, 0.0, 1.0]])
    idx, climb = find_minimum_ascent_path(e, (1, 0), (1, 3))
    assert len(idx) == 6 and climb[-1] == 1.0


def test_relations_between_the_path_kinds_on_a_random_grid():
    rng = np.random.default_rng(7)
    G = rng.random((12, 12)) * 10
    for a, b in [((0, 0), (11, 11)), ((3, 9), (10, 1)), ((5, 5), (0, 11))]:
        p_mm, e_mm = find_minimax_path(G, a, b)
        level = e_mm.max()
        p_asc, c_asc = find_minimum_ascent_path(G, a, b)
        p_back, c_back = find_minimum_ascent_path(G, b, a)
        p_la, c_la = find_least_action_path(G, a, b, neighborhood="von_neumann")
        # 1. no path crosses lower than the minimax level
        assert max(G[tuple(p)] for p in p_asc) >= level - 1e-12
        assert max(G[tuple(p)] for p in p_la) >= level - 1e-12
        # 2. the minimum-ascent path climbs no more than the minimax path
        assert c_asc[-1] <= ascent_along(G, p_mm) + 1e-9
        # 3. the least-action path costs no more than the other two, for the same cost
        assert c_la[-1] <= action_along(G, p_mm) + 1e-9 and c_la[-1] <= action_along(G, p_asc) + 1e-9
        # 4. forward minus backward ascent equals the energy difference
        assert c_asc[-1] - c_back[-1] == pytest.approx(G[b] - G[a])
        # lower bounds on the total ascent
        assert c_asc[-1] >= max(G[b] - G[a], 0) - 1e-12 and c_asc[-1] >= level - G[a] - 1e-12
    # 5. the steepest-descent terminus is reached with zero ascent
    p_sd, _ = find_steepest_descent_path(G, (6, 6))
    _, climb = find_minimum_ascent_path(G, (6, 6), tuple(p_sd[-1]), neighborhood="moore")
    assert climb[-1] == 0.0


def test_float32_matches_float64_and_length_one_axis():
    d = np.array([[0.0, 5.0, 0.0], [0.0, 1.0, 0.0]])
    r32 = find_minimum_ascent_path(d.astype(np.float32), (0, 0), (0, 2))
    r64 = find_minimum_ascent_path(d, (0, 0), (0, 2))
    np.testing.assert_array_equal(r32[0], r64[0])
    np.testing.assert_array_equal(r32[1], r64[1])
    idx, action = find_least_action_path(np.ones((1, 4)), (0, 0), (0, 3), axes=[[2.0], [0.0, 1.0, 2.0, 3.0]])
    assert action[-1] == 3.0


# -------- error contract (spec 6.4) -----------------------------------------


def test_errors():
    c = np.ones((4, 4))
    with pytest.raises(ValueError, match="C-contiguous"):
        find_least_action_path(c.T.copy().T, (0, 0), (3, 3))
    with pytest.raises(ValueError, match="ndim"):
        find_least_action_path(np.ones(4), (0,), (3,))
    with pytest.raises(ValueError):
        find_least_action_path(np.ones((4, 4), dtype=np.int32), (0, 0), (3, 3))
    with pytest.raises(ValueError):
        find_least_action_path(c, (0,), (3, 3))
    with pytest.raises(ValueError):
        find_least_action_path(c, (0, 0), (3,))
    with pytest.raises(IndexError):
        find_least_action_path(c, (4, 0), (3, 3))
    with pytest.raises(IndexError):
        find_least_action_path(c, (0, 0), (-1, 3))
    nan_cell = c.copy()
    nan_cell[0, 0] = nan
    with pytest.raises(ValueError, match="NaN"):
        find_least_action_path(nan_cell, (0, 0), (3, 3))
    with pytest.raises(ValueError, match="NaN"):
        find_least_action_path(nan_cell, (3, 3), (0, 0))
    with pytest.raises(ValueError, match="shape"):
        find_least_action_path(c, (0, 0), np.zeros((4, 5), dtype=bool))
    with pytest.raises(ValueError, match="True"):
        find_least_action_path(c, (0, 0), np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="boolean mask"):
        find_least_action_path(c, (0, 0), np.ones((4, 4), dtype=np.int64))   # 0/1 ints are not a mask
    with pytest.raises(ValueError, match="C-contiguous"):
        find_least_action_path(c, (0, 0), np.ones((4, 4), dtype=bool).T.copy().T)
    inf_cell = c.copy()
    inf_cell[1, 1] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        find_least_action_path(inf_cell, (0, 0), (3, 3))
    with pytest.raises(ValueError, match="infinite"):
        find_minimum_ascent_path(inf_cell, (0, 0), (3, 3))
    neg = c.copy()
    neg[1, 1] = -1.0
    with pytest.raises(ValueError, match="negative"):
        find_least_action_path(neg, (0, 0), (3, 3))
    assert find_minimum_ascent_path(neg, (0, 0), (3, 3)) is not None          # negative energies are fine
    with pytest.raises(ValueError):
        find_least_action_path(c, (0, 0), (3, 3), axes=[[0.0, 1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        find_least_action_path(c, (0, 0), (3, 3), axes=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        find_least_action_path(c, (0, 0), (3, 3), axes=[[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        find_least_action_path(c, (0, 0), (3, 3), neighborhood="king")


def test_float_indices_are_rejected_not_truncated():
    # coordinates passed by mistake as an index must not silently become a cell
    c = np.ones((4, 4))
    with pytest.raises(TypeError):
        find_steepest_descent_path(c, (0.9, 1.7))
    with pytest.raises(TypeError):
        find_least_action_path(c, (0.5, 0), (3, 3))
    with pytest.raises(TypeError):
        find_least_action_path(c, (0, 0), np.array([2.6, 3.4]))
    with pytest.raises(TypeError):
        find_minimum_ascent_path(c, (0, 0), (3.0, 3.0))
    # NumPy integers are indices
    assert find_minimum_ascent_path(c, (np.int64(0), np.int32(0)), np.array([3, 3]))[0][-1].tolist() == [3, 3]
