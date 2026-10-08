"""The draw helpers return the Axes they drew on, with the artists the spec names (Agg; no rebuild)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.contour import ContourSet  # noqa: E402

from pes_analyzer.grid import index_to_coords  # noqa: E402
from pes_analyzer.plot import plot_map, plot_path  # noqa: E402
from pes_analyzer.plot._map import _cell_edges  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _mesh(ax) -> QuadMesh:
    meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
    assert len(meshes) == 1
    return meshes[0]


def test_cell_edges_are_midpoints_with_half_steps_outside():
    np.testing.assert_allclose(_cell_edges(np.array([0.0, 1.0, 3.0])), [-0.5, 0.5, 2.0, 4.0])
    np.testing.assert_allclose(_cell_edges(np.array([10.0, 20.0])), [5.0, 15.0, 25.0])


def test_plot_map_returns_the_axes_and_masks_nan():
    values = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]])
    fig, ax = plt.subplots()
    out = plot_map(values, ax=ax)
    assert out is ax
    mesh = _mesh(ax)
    assert mesh.get_array().mask.sum() == 1 and mesh.get_rasterized()
    assert ax.get_xlabel() == "index 0" and ax.get_ylabel() == "index 1"
    assert len(fig.axes) == 2                                   # the colour bar
    x0, x1 = ax.get_xlim()
    assert x0 == pytest.approx(-0.5) and x1 == pytest.approx(1.5)   # axis 0 (2 cells) is horizontal
    plt.figure()
    ax2 = plot_map(values, colorbar=False)                      # ax=None draws on the current axes
    assert ax2 is plt.gca() and len(ax2.figure.axes) == 1


def test_plot_map_cell_edges_follow_the_axes():
    values = np.zeros((3, 2))
    axes = {"c": np.array([1.0, 1.1, 1.5]), "a4": np.array([0.0, 0.2])}
    _fig, ax = plt.subplots()
    plot_map(values, axes, ax=ax, colorbar=False)
    coords = _mesh(ax).get_coordinates()                       # (ny+1, nx+1, 2)
    np.testing.assert_allclose(coords[0, :, 0], [0.95, 1.05, 1.3, 1.7])
    np.testing.assert_allclose(coords[:, 0, 1], [-0.1, 0.1, 0.3])
    assert ax.get_xlabel() == "c" and ax.get_ylabel() == "a4"
    _fig, ax = plt.subplots()
    plot_map(values, [axes["c"], axes["a4"]], ax=ax, colorbar=False)
    assert ax.get_xlabel() == "" and ax.get_ylabel() == ""


def test_plot_map_levels_contours_mask():
    values = np.add.outer(np.arange(4.0), np.arange(5.0))
    _fig, ax = plt.subplots()
    plot_map(values, ax=ax, levels=[0, 2, 4, 6], contours=[1.5, 3.5], mask=values > 5, colorbar=False)
    mesh = _mesh(ax)
    assert isinstance(mesh.norm, BoundaryNorm) and mesh.norm.extend == "both"
    assert any(isinstance(c, ContourSet) for c in ax.collections) or len(ax.texts) > 0
    hatched = [c for c in ax.collections if c.get_hatch() == "///"]
    assert len(hatched) == 1
    with pytest.raises(ValueError):
        plot_map(values, levels=[0, 1], norm=BoundaryNorm([0, 1], 2))
    with pytest.raises(ValueError):
        plot_map(values, levels=[3, 1])
    with pytest.raises(ValueError):
        plot_map(values, mask=np.ones((2, 2), bool))


def test_plot_map_passes_mesh_kwargs():
    values = np.arange(6.0).reshape(2, 3)
    _fig, ax = plt.subplots()
    plot_map(values, ax=ax, vmin=-1.0, vmax=10.0, colorbar=False)
    assert _mesh(ax).norm.vmin == -1.0 and _mesh(ax).norm.vmax == 10.0


def test_plot_map_rejects_bad_shapes():
    with pytest.raises(ValueError):
        plot_map(np.arange(5.0))
    with pytest.raises(ValueError):
        plot_map(np.zeros((1, 5)))
    with pytest.raises(ValueError):
        plot_map(np.zeros((3, 3)), axes=[np.arange(3.0), np.arange(4.0)])


def test_plot_path_projects_onto_the_kept_axes():
    axes = {"x": np.array([0.0, 1.0, 3.0]), "y": np.array([10.0, 20.0, 40.0, 80.0]), "z": np.array([5.0, 6.0])}
    path = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 1]])
    _fig, ax = plt.subplots()
    out = plot_path(path, axes, keep=(0, 1), ax=ax, color="red")
    assert out is ax and len(ax.lines) == 1
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xydata(), index_to_coords(path, axes)[:, [0, 1]])
    assert line.get_marker() == "." and line.get_color() == "red"
    plot_path(path, keep=(2, 0), ax=ax)
    np.testing.assert_allclose(ax.lines[1].get_xydata(), path[:, [2, 0]])
    for bad in ((0, 0), (0, 3), (0,), (0, 1, 2)):
        with pytest.raises(ValueError):
            plot_path(path, axes, keep=bad)
    with pytest.raises(TypeError):
        plot_path(path, axes, keep=(0.8, 1.8))


def test_plot_path_points_and_float_indices():
    points = np.array([[0, 1], [2, 3], [1, 0]])
    _fig, ax = plt.subplots()
    plot_path(points, ax=ax, linestyle="none", marker="o")
    assert ax.lines[0].get_linestyle() == "None" and ax.lines[0].get_marker() == "o"
    assert len(ax.lines[0].get_xydata()) == 3
    with pytest.raises(ValueError):
        plot_path(points.astype(float), ax=ax)


from pes_analyzer.grid import path_length  # noqa: E402
from pes_analyzer.plot import plot_profile  # noqa: E402
from pes_analyzer.topology import analyze_path_profile  # noqa: E402


def test_plot_profile_x_axis_and_markers():
    energies = np.array([0.0, 2.0, 1.0, 3.0, 0.5])
    path = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2]])
    axes = [np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0, 4.0])]
    profile = analyze_path_profile(energies)
    _fig, ax = plt.subplots()
    out = plot_profile(energies, indices=path, axes=axes, profile=profile, ax=ax)
    assert out is ax
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), path_length(path, axes))
    np.testing.assert_allclose(line.get_ydata(), energies)
    assert line.get_marker() == "."
    markers = {ln.get_marker(): len(ln.get_xydata()) for ln in ax.lines[1:]}
    assert markers == {"o": len(profile.minima), "s": len(profile.saddles)}
    assert all(ln.get_markerfacecolor() == line.get_color() for ln in ax.lines[1:])
    assert ax.get_xlabel() == "length" and ax.get_ylabel() == "energy"
    _fig, ax = plt.subplots()
    plot_profile(energies, indices=path, ax=ax)
    assert ax.get_xlabel() == "length (cells)"
    _fig, ax = plt.subplots()
    plot_profile(energies, ax=ax, label="climb")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), np.arange(5))
    assert ax.get_xlabel() == "step" and ax.get_ylabel() == "climb"


def test_plot_profile_draws_a_cumulative_climb_unchanged():
    climb = np.array([0.0, 3.0, 3.0, 4.0, 4.0])
    _fig, ax = plt.subplots()
    plot_profile(climb, ax=ax, label="climb")
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), climb)


def test_plot_profile_validation():
    energies = np.array([0.0, 2.0, 1.0])
    with pytest.raises(ValueError):
        plot_profile(energies, indices=np.zeros((2, 2), int))
    with pytest.raises(ValueError):
        plot_profile(np.zeros((2, 2)))
    bad = analyze_path_profile(np.array([0.0, 2.0, 1.0, 3.0, 0.5]))        # indices up to 4
    with pytest.raises(ValueError):
        plot_profile(energies, profile=bad)


from matplotlib.collections import LineCollection  # noqa: E402

from pes_analyzer.plot import merge_tree_layout, plot_merge_tree  # noqa: E402
from pes_analyzer.topology import MergeTree, Watershed  # noqa: E402


def _fan_ws():
    basins = [((0, 0), 0.0), ((0, 2), 1.0), ((0, 4), 0.5), ((0, 6), 1.5), ((0, 8), 1.5)]
    merges = [((0, 7), 2.0, 1, 4), ((0, 1), 3.0, 0, 1), ((0, 5), 4.0, 0, 3), ((0, 3), 5.0, 0, 2)]
    return Watershed(labels=None, basins=basins, merges=merges, neighborhood="von_neumann",
                     parents=None, merge_table=None, dtype=np.dtype("float64"), fingerprint=b"\x00" * 16)


def test_plot_merge_tree_segments_and_labels():
    ws = _fan_ws()
    layout = merge_tree_layout(ws)
    _fig, ax = plt.subplots()
    out = plot_merge_tree(ws, ax=ax, labels={0: "deepest", 4: "leaf"}, saddle_labels={2: "highest saddle"}, label="E")
    assert out is ax
    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lines) == 1 and len(lines[0].get_segments()) == len(layout.branches) + len(layout.connectors)
    texts = {t.get_text(): t.xy for t in ax.texts}
    assert texts["deepest"] == (layout.x[0], 0.0) and texts["leaf"] == (layout.x[4], 1.5)
    x_c, x_p, e_s, _ = next(c for c in layout.connectors if c[3] == 2)
    assert texts["highest saddle"] == (0.5 * (x_c + x_p), e_s)
    markers = sorted(ln.get_marker() for ln in ax.lines)
    assert markers == ["o", "o", "s"]
    assert ax.get_ylabel() == "E" and ax.get_xticks().size == 0
    assert ax.get_xlim() == (-1.0, 5.0)
    lo, hi = ax.get_ylim()
    assert lo < 0.0 and hi > layout.top
    _fig, ax = plt.subplots()
    plot_merge_tree(MergeTree(ws), ax=ax, min_persistence=1.0)
    assert len(ax.collections[0].get_segments()) == 4 + 3


def test_plot_merge_tree_rejects_undrawn_ids():
    ws = _fan_ws()
    with pytest.raises(ValueError, match="4"):
        plot_merge_tree(ws, min_persistence=1.0, labels={4: "pruned away"})
    with pytest.raises(ValueError, match="1"):
        plot_merge_tree(ws, max_basins=3, saddle_labels={1: "capped away"})
    with pytest.raises(ValueError, match="root"):
        plot_merge_tree(ws, saddle_labels={0: "no saddle"})
