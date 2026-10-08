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


from matplotlib.figure import Figure

from pes_analyzer.plot import merge_tree_layout, plot_map, plot_merge_tree
from pes_analyzer.tables import basins_table, minima_table, path_table, read_csv, write_csv
from pes_analyzer.topology import MergeTree, find_watershed_segmentation


def test_tables_example(tmp_path):
    axes = {"x": np.array([0.0, 1.0, 3.0]), "y": np.array([10.0, 20.0, 40.0, 80.0])}
    table = minima_table([((0, 1), 2.5), ((2, 3), -1.0)], axes)
    assert list(table) == ["index_0", "index_1", "x", "y", "energy"]
    write_csv(tmp_path / "minima.csv", table)
    assert (tmp_path / "minima.csv").read_text() == "index_0,index_1,x,y,energy\n0,1,0.0,20.0,2.5\n2,3,3.0,80.0,-1.0\n"
    back = read_csv(tmp_path / "minima.csv")
    assert back["index_0"].dtype == np.int64 and back["x"].tolist() == [0.0, 3.0]
    path = path_table(np.array([[0, 0], [1, 1], [2, 3]]), [1.0, 2.0, 0.5], axes)
    assert list(path) == ["step", "index_0", "index_1", "x", "y", "length", "energy"]
    np.testing.assert_allclose(path["length"], [0.0, 10.04987562, 70.0831997])


def test_basins_table_example():
    energies = np.array([[0.0, 3.0, 8.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]])
    tree = MergeTree(find_watershed_segmentation(energies))
    table = basins_table(tree)
    assert table["basin"].tolist() == [0, 1, 2] and table["parent"].tolist() == [-1, 0, 1]
    assert table["saddle_index_1"].tolist() == [-1, 2, 6]
    assert np.isnan(table["saddle_energy"][0]) and table["saddle_energy"][1:].tolist() == [8.0, 5.0]
    assert np.isinf(table["persistence"][0]) and table["persistence"][1:].tolist() == [7.0, 3.0]
    assert basins_table(tree, min_persistence=4.0)["basin"].tolist() == [0, 1]


def test_merge_tree_layout_example():
    energies = np.array([[0.0, 3.0, 8.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]])
    layout = merge_tree_layout(find_watershed_segmentation(energies))
    assert layout.x == {0: 0, 1: 1, 2: 2}
    assert layout.connectors == [(2, 1, 5.0, 2), (1, 0, 8.0, 1)]
    assert layout.branches == {0: (0, 0.0, 8.4), 1: (1, 1.0, 8.0), 2: (2, 2.0, 5.0)}
    assert layout.top == 8.4


def test_plot_example():
    surf = hidden_barrier()
    axes = {"x": np.linspace(-1.5, 1.5, 61), "y": np.linspace(-0.5, 0.5, 21), "z": np.linspace(-1.5, 1.5, 61)}
    energies = surf.sample(axes)
    minimum, index = minimize_grid(energies, keep=(0, 1))
    jumps = jump_map(index, keep=(0, 1))
    fig = Figure(figsize=(10, 4))
    ax_map, ax_tree = fig.subplots(1, 2)
    out = plot_map(minimum, [axes["x"], axes["y"]], ax=ax_map, levels=np.arange(-1.0, 7.0), mask=jumps >= 5)
    assert out is ax_map and len(fig.axes) == 3                     # two panels plus the colour bar
    tree = MergeTree(find_watershed_segmentation(energies))
    plot_merge_tree(tree, ax=ax_tree, min_persistence=0.1, labels={0: "A"})
    assert len(ax_tree.texts) == 1
