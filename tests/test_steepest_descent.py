"""Integration tests for find_steepest_descent_path (compiled extension)."""

from __future__ import annotations

import numpy as np
import pytest

from pes_analyzer.extrema import find_minima_grid
from pes_analyzer.synthetic import muller_brown
from pes_analyzer.topology import find_steepest_descent_path


def test_chain_descends_to_the_end_and_output_contract():
    e = np.array([[4.0, 3.0, 2.0, 1.0, 0.0]])
    idx, prof = find_steepest_descent_path(e, (0, 0))
    assert idx.dtype == np.int64 and idx.shape == (5, 2)
    assert prof.dtype == np.float64 and prof.tolist() == [4.0, 3.0, 2.0, 1.0, 0.0]
    np.testing.assert_array_equal(idx[:, 1], np.arange(5))
    idx, prof = find_steepest_descent_path(e, (0, 4))
    assert idx.tolist() == [[0, 4]] and prof.tolist() == [0.0]


def test_equal_slopes_go_to_the_smallest_linear_index():
    e = np.array([[5.0, 0.0, 5.0], [0.0, 1.0, 0.0], [5.0, 0.0, 5.0]])
    idx, _ = find_steepest_descent_path(e, (1, 1), neighborhood="von_neumann")
    assert idx.tolist() == [[1, 1], [0, 1]]


def test_axes_change_the_choice():
    e = np.full((3, 3), 20.0)
    e[1, 1], e[1, 2], e[2, 1] = 10.0, 8.0, 9.0
    idx, prof = find_steepest_descent_path(e, (1, 1), neighborhood="von_neumann")
    assert idx.tolist() == [[1, 1], [1, 2]] and prof.tolist() == [10.0, 8.0]
    axes = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 4.0, 8.0])]
    idx, prof = find_steepest_descent_path(e, (1, 1), axes=axes, neighborhood="von_neumann")
    assert idx.tolist() == [[1, 1], [2, 1]] and prof.tolist() == [10.0, 9.0]
    idx2, _ = find_steepest_descent_path(e, (1, 1), axes={"x": axes[0], "y": axes[1]}, neighborhood="von_neumann")
    assert idx2.tolist() == idx.tolist()


def test_moore_default_takes_the_steeper_diagonal():
    e = np.array([[10.0, 9.0], [20.0, 8.5]])
    assert find_steepest_descent_path(e, (0, 0))[0].tolist() == [[0, 0], [1, 1]]
    assert find_steepest_descent_path(e, (0, 0), neighborhood="von_neumann")[0].tolist() == [[0, 0], [0, 1], [1, 1]]


def test_plateau_nan_neighbours_and_length_one_axis():
    assert find_steepest_descent_path(np.array([[1.0, 1.0, 0.0]]), (0, 0))[0].tolist() == [[0, 0]]
    walled = np.full((3, 3), np.nan)
    walled[1, 1] = 3.0
    assert find_steepest_descent_path(walled, (1, 1))[0].tolist() == [[1, 1]]
    one = np.array([[3.0, 2.0, 1.0]])            # shape (1, 3): no step along axis 0
    idx, _ = find_steepest_descent_path(one, (0, 0), axes=[[5.0], [0.0, 1.0, 2.0]])
    assert idx.tolist() == [[0, 0], [0, 1], [0, 2]]


def test_end_cell_is_a_moore_minimum_on_muller_brown():
    surf = muller_brown()
    axes = [np.linspace(-1.5, 1.2, 109), np.linspace(-0.3, 2.0, 93)]
    E = surf.sample(axes)
    minima = {tuple(i) for i, _ in find_minima_grid(E)}
    rng = np.random.default_rng(1)
    for _ in range(20):
        start = (int(rng.integers(0, 109)), int(rng.integers(0, 93)))
        idx, prof = find_steepest_descent_path(E, start, axes=axes)
        assert tuple(idx[0]) == start
        assert (np.diff(prof) < 0).all()
        assert tuple(idx[-1]) in minima


def test_float32_matches_float64():
    e64 = np.array([[4.0, 3.0, 2.5], [2.0, 1.0, 0.0]])
    r32 = find_steepest_descent_path(e64.astype(np.float32), (0, 0))
    r64 = find_steepest_descent_path(e64, (0, 0))
    np.testing.assert_array_equal(r32[0], r64[0])
    np.testing.assert_array_equal(r32[1], r64[1])


def test_errors():
    e = np.zeros((4, 4))
    with pytest.raises(ValueError, match="C-contiguous"):
        find_steepest_descent_path(e.T.copy().T, (0, 0))
    with pytest.raises(ValueError, match="ndim"):
        find_steepest_descent_path(np.zeros(4), (0,))
    with pytest.raises(ValueError):
        find_steepest_descent_path(np.zeros((4, 4), dtype=np.int32), (0, 0))
    with pytest.raises(ValueError):
        find_steepest_descent_path(e, (0,))
    with pytest.raises(IndexError):
        find_steepest_descent_path(e, (4, 0))
    with pytest.raises(IndexError):
        find_steepest_descent_path(e, (-1, 0))
    nan_start = e.copy()
    nan_start[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        find_steepest_descent_path(nan_start, (0, 0))
    with pytest.raises(ValueError):
        find_steepest_descent_path(e, (0, 0), neighborhood="king")
    with pytest.raises(ValueError):
        find_steepest_descent_path(e, (0, 0), axes=[[0.0, 1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        find_steepest_descent_path(e, (0, 0), axes=[[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
