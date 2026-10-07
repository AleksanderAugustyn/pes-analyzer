"""The code blocks of the new API.md sections, with their printed values asserted."""

from __future__ import annotations

import numpy as np

from pes_analyzer.grid import index_to_coords, jump_map, minimize_grid, path_length
from pes_analyzer.synthetic import hidden_barrier
from pes_analyzer.topology import find_least_action_path, find_minimum_ascent_path, find_steepest_descent_path


def test_minimize_grid_example():
    energies = np.array([[[3.0, 1.0, 2.0], [0.5, 4.0, 4.0]], [[2.0, 2.0, 0.0], [np.nan, np.nan, np.nan]]])
    minimum, index = minimize_grid(energies, keep=(0, 1))
    np.testing.assert_array_equal(minimum, [[1.0, 0.5], [0.0, np.nan]])
    assert index[0, 0].tolist() == [0, 0, 1] and index[1, 1].tolist() == [-1, -1, -1]
    np.testing.assert_array_equal(jump_map(index, keep=(0, 1)), [[1.0, 1.0], [1.0, np.nan]])
    valid = index[..., 0] >= 0
    np.testing.assert_array_equal(energies[tuple(np.moveaxis(index[valid], -1, 0))], [1.0, 0.5, 0.0])


def test_coordinates_example():
    axes = {"x": np.array([0.0, 1.0, 3.0]), "y": np.array([10.0, 20.0, 40.0, 80.0])}
    path = np.array([[0, 0], [1, 1], [2, 3]])
    np.testing.assert_array_equal(index_to_coords(path, axes), [[0.0, 10.0], [1.0, 20.0], [3.0, 80.0]])
    np.testing.assert_allclose(path_length(path, axes), [0.0, 10.04987562, 70.0831997])
    np.testing.assert_allclose(path_length(path), [0.0, 1.41421356, 3.65028154])


def test_steepest_descent_example():
    energies = np.full((3, 3), 20.0)
    energies[1, 1], energies[1, 2], energies[2, 1] = 10.0, 8.0, 9.0
    idx, prof = find_steepest_descent_path(energies, (1, 1), neighborhood="von_neumann")
    assert idx.tolist() == [[1, 1], [1, 2]] and prof.tolist() == [10.0, 8.0]
    axes = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 4.0, 8.0])]
    idx, prof = find_steepest_descent_path(energies, (1, 1), axes=axes, neighborhood="von_neumann")
    assert idx.tolist() == [[1, 1], [2, 1]] and prof.tolist() == [10.0, 9.0]


def test_least_action_example():
    cost = np.array([[1.0, 1.0, 1.0], [1.0, 100.0, 1.0], [2.0, 2.0, 2.0]])
    idx, action = find_least_action_path(cost, (1, 0), (1, 2), neighborhood="von_neumann")
    assert idx.tolist() == [[1, 0], [0, 0], [0, 1], [0, 2], [1, 2]] and action.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    exit_mask = np.zeros((3, 3), dtype=bool)
    exit_mask[2, :] = True
    idx, action = find_least_action_path(cost, (1, 0), exit_mask, neighborhood="von_neumann")
    assert idx.tolist() == [[1, 0], [2, 0]] and action.tolist() == [0.0, 1.5]


def test_minimum_ascent_example():
    energies = np.array([[0.0, 5.0, 0.0], [0.0, 1.0, 0.0]])
    idx, climb = find_minimum_ascent_path(energies, (0, 0), (0, 2))
    assert idx.tolist() == [[0, 0], [1, 0], [1, 1], [1, 2], [0, 2]] and climb.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_synthetic_example():
    surf = hidden_barrier()
    assert surf.minima[0] == ((-1.0, 0.0, -1.0), -0.5)
    assert surf.saddles[-1][0] == (1.0, 0.0, -0.0375) and round(surf.saddles[-1][1], 6) == 5.014059
    assert surf.saddles[-1][2] == (1, 3)
    E = surf.sample({"x": np.linspace(-1.5, 1.5, 61), "y": np.linspace(-0.5, 0.5, 21), "z": np.linspace(-1.5, 1.5, 61)})
    minimum, index = minimize_grid(E, keep=(0, 1))
    assert round(float(minimum[30, 10] - minimum[10, 10]), 6) == 1.5
