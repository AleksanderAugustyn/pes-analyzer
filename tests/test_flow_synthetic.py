"""The whole flow on an analytic surface: sample, analyse, minimise, plot, save, read back."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pes_analyzer.extrema import find_minima_grid  # noqa: E402
from pes_analyzer.grid import jump_map, minimize_grid  # noqa: E402
from pes_analyzer.plot import plot_map, plot_merge_tree, plot_path, plot_profile  # noqa: E402
from pes_analyzer.synthetic import hidden_barrier  # noqa: E402
from pes_analyzer.tables import basins_table, minima_table, path_table, read_csv, write_csv  # noqa: E402
from pes_analyzer.topology import MergeTree, analyze_path_profile, find_minimax_path, find_watershed_segmentation  # noqa: E402


def test_flow_on_hidden_barrier(tmp_path):
    surf = hidden_barrier()
    axes = {"x": np.linspace(-1.5, 1.5, 61), "y": np.linspace(-0.5, 0.5, 21), "z": np.linspace(-1.5, 1.5, 61)}
    energies = surf.sample(axes)

    # analyse
    minima = find_minima_grid(energies)
    ws = find_watershed_segmentation(energies, parents=True)
    tree = MergeTree(ws)
    assert len(minima) == 4 and len(ws.basins) >= 4            # extra basins, if any, are below 0.1 persistence
    start, end = ws.basins[0][0], ws.basins[1][0]
    path_idx, path_e = find_minimax_path(energies, start, end, tree=tree)
    profile = analyze_path_profile(path_e)

    # minimise
    minimum, index = minimize_grid(energies, keep=(0, 1))
    jumps = jump_map(index, keep=(0, 1))
    map_axes = [axes["x"], axes["y"]]

    # plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plot_map(minimum, map_axes, ax=axs[0, 0], mask=jumps >= 5, contours=np.arange(0.0, 6.0))
    plot_path(path_idx, axes, keep=(0, 1), ax=axs[0, 0], color="red")
    plot_map(jumps, map_axes, ax=axs[0, 1], label="jump (cells)")
    plot_merge_tree(tree, ax=axs[1, 0], min_persistence=0.1, labels={0: "A", 1: "B"}, saddle_labels={1: "A|B"})
    plot_profile(path_e, indices=path_idx, axes=axes, profile=profile, ax=axs[1, 1])
    fig.savefig(tmp_path / "flow.pdf")
    plt.close(fig)
    assert (tmp_path / "flow.pdf").stat().st_size > 0
    assert (jumps >= 5).any()                                       # the map really hides a valley switch

    # save and read back
    tables = {
        "minima": minima_table(minima, axes),
        "basins": basins_table(tree, axes, min_persistence=0.1),
        "path": path_table(path_idx, path_e, axes),
    }
    for name, table in tables.items():
        write_csv(tmp_path / f"{name}.csv", table)
        back = read_csv(tmp_path / f"{name}.csv")
        assert list(back) == list(table)
        for column in table:
            assert back[column].dtype == table[column].dtype
            np.testing.assert_array_equal(back[column], table[column])
    assert tables["basins"]["basin"].tolist() == [0, 1, 2, 3]
    assert tables["path"]["energy"].max() == path_e.max()
