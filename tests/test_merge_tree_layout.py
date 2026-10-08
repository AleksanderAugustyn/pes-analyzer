"""Pure layout of the merge-tree dendrogram: selection, contiguity, no crossings (no rebuild)."""
from __future__ import annotations

import numpy as np
import pytest

from pes_analyzer.plot import MergeTreeLayout, merge_tree_layout
from pes_analyzer.synthetic import separable_wells
from pes_analyzer.topology import MergeTree, Watershed, find_watershed_segmentation


def _ws(basins, merges):
    return Watershed(labels=None, basins=basins, merges=merges, neighborhood="von_neumann",
                     parents=None, merge_table=None, dtype=np.dtype("float64"), fingerprint=b"\x00" * 16)


def _fan():
    """Root 0; children 1 (saddle 3), 2 (saddle 5), 3 (saddle 4); grandchild 4 under 1 (saddle 2)."""
    basins = [((0, 0), 0.0), ((0, 2), 1.0), ((0, 4), 0.5), ((0, 6), 1.5), ((0, 8), 1.5)]
    merges = [((0, 7), 2.0, 1, 4), ((0, 1), 3.0, 0, 1), ((0, 5), 4.0, 0, 3), ((0, 3), 5.0, 0, 2)]
    return _ws(basins, merges)


def _forest():
    basins = [((0, 0), 0.0), ((0, 3), 1.0), ((1, 0), 0.5), ((1, 3), 2.0)]
    merges = [((0, 1), 3.0, 0, 1), ((1, 1), 4.0, 2, 3)]
    return _ws(basins, merges)


def _check_invariants(layout: MergeTreeLayout, tree: MergeTree) -> None:
    ids = layout.basin_ids
    assert ids == sorted(ids) and set(layout.x) == set(ids) == set(layout.branches)
    assert sorted(layout.x.values()) == list(range(len(ids)))                      # a permutation of the slots
    drawn = set(ids)
    for b in ids:
        p = tree.node(b).parent
        assert p is None or p in drawn                                           # closed under parent
        x, e_min, e_top = layout.branches[b]
        assert x == layout.x[b] and e_min <= e_top
        assert e_top == (layout.top if p is None else tree.node(b).saddle_to_parent[1])
    # every subtree occupies a contiguous block of slots
    subtree = {b: {b} for b in ids}
    for b in ids:
        p = tree.node(b).parent
        while p is not None:
            subtree[p].add(b)
            p = tree.node(p).parent
    for b, members in subtree.items():
        slots = sorted(layout.x[m] for m in members)
        assert slots == list(range(slots[0], slots[0] + len(slots)))
    # no connector crosses a branch: everything strictly between its ends tops out at or below it
    assert layout.connectors == sorted(layout.connectors, key=lambda c: (c[2], c[3]))
    for x_c, x_p, e_s, child in layout.connectors:
        assert layout.x[child] == x_c and layout.x[tree.node(child).parent] == x_p
        lo, hi = sorted((x_c, x_p))
        for b, (xb, _e_min, e_top) in layout.branches.items():
            if lo < xb < hi:
                assert e_top <= e_s


def test_fan_layout_is_exact():
    ws = _fan()
    layout = merge_tree_layout(ws)
    assert layout.basin_ids == [0, 1, 2, 3, 4]
    assert layout.x == {3: 0, 0: 1, 1: 2, 4: 3, 2: 4}                 # left child 3, root, then 1 (with 4), 2
    assert layout.connectors == [(3, 2, 2.0, 4), (2, 1, 3.0, 1), (0, 1, 4.0, 3), (4, 1, 5.0, 2)]
    assert layout.top == pytest.approx(5.0 + 0.05 * 5.0)
    assert layout.branches[0] == (1, 0.0, layout.top) and layout.branches[2] == (4, 0.5, 5.0)
    _check_invariants(layout, MergeTree(ws))


def test_fan_selection_arguments():
    ws = _fan()
    tree = MergeTree(ws)
    pruned = merge_tree_layout(ws, min_persistence=1.0)
    assert pruned.basin_ids == [0, 1, 2, 3] and 4 not in pruned.x
    _check_invariants(pruned, tree)
    capped = merge_tree_layout(tree, max_basins=3)
    assert capped.basin_ids == [0, 2, 3]
    _check_invariants(capped, tree)
    assert merge_tree_layout(ws, max_basins=None).basin_ids == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        merge_tree_layout(ws, max_basins=0)


def test_forest_draws_every_root():
    ws = _forest()
    layout = merge_tree_layout(ws)
    assert layout.basin_ids == [0, 1, 2, 3]
    assert layout.x == {0: 0, 1: 1, 2: 2, 3: 3}
    assert layout.top == pytest.approx(4.0 + 0.05 * 4.0)
    assert layout.branches[0][2] == layout.top and layout.branches[2][2] == layout.top
    _check_invariants(layout, MergeTree(ws))
    roots_only = merge_tree_layout(ws, min_persistence=2.5)
    assert roots_only.basin_ids == [0, 2] and roots_only.connectors == []
    assert roots_only.top == pytest.approx(0.5 + 0.05 * 0.5)
    assert merge_tree_layout(ws, max_basins=1).basin_ids == [0, 2]


def test_nan_wall_forest_from_the_flood():
    nan = np.nan
    E = np.array([[5.0, 4.0, nan, 6.0, 7.0], [4.0, 0.0, nan, 1.0, 6.0], [5.0, 4.0, nan, 6.0, 7.0]])
    ws = find_watershed_segmentation(E)
    assert len(ws.basins) == 2 and ws.merges == []
    layout = merge_tree_layout(ws)
    assert layout.basin_ids == [0, 1] and layout.connectors == []
    assert layout.top == pytest.approx(1.0 + 0.05 * 1.0)
    _check_invariants(layout, MergeTree(ws))


def test_elevated_root_and_single_basin():
    basins = [((0, 0), 0.0), ((0, 2), 1.0), ((1, 0), 100.0)]          # the second root sits far above the only saddle
    merges = [((0, 1), 2.0, 0, 1)]
    ws = _ws(basins, merges)
    layout = merge_tree_layout(ws)
    assert layout.top == pytest.approx(100.0 + 0.05 * 100.0)
    assert layout.branches[2] == (2, 100.0, layout.top)
    _check_invariants(layout, MergeTree(ws))
    single = merge_tree_layout(_ws([((0, 0), 3.0)], []))
    assert single.x == {0: 0} and single.top == pytest.approx(4.0) and single.connectors == []


def test_empty_tree_raises():
    with pytest.raises(ValueError):
        merge_tree_layout(_ws([], []))


@pytest.mark.parametrize("ndim, n", [(3, 31), (4, 13)])
def test_separable_wells_invariants(ndim, n):
    surf = separable_wells(ndim)
    axes = [np.linspace(-1.5, 1.5, n)] * ndim
    ws = find_watershed_segmentation(surf.sample(axes))
    tree = MergeTree(ws)
    assert len(ws.basins) == 2**ndim
    for kwargs in ({}, {"min_persistence": 0.5}, {"max_basins": 5}, {"min_persistence": 0.2, "max_basins": 3}):
        _check_invariants(merge_tree_layout(tree, **kwargs), tree)
