"""Merge-tree dendrogram: crossing-free layout (pure Python) and its matplotlib drawing."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..topology._tree import _select_basins
from ..topology.merge_tree import MergeTree, _as_tree

__all__ = ["MergeTreeLayout", "merge_tree_layout", "plot_merge_tree"]


@dataclass(frozen=True)
class MergeTreeLayout:
    """Draw-ready positions of a (pruned) merge tree. See ``_docs/API.md``.

    ``x`` maps a basin id to its integer slot; ``branches`` to ``(x, e_min, e_top)``;
    ``connectors`` holds ``(x_child, x_parent, e_saddle, child_id)`` in ascending
    saddle order; every root's branch ends at ``top``.
    """

    basin_ids: list[int]
    x: dict[int, int]
    branches: dict[int, tuple[int, float, float]]
    connectors: list[tuple[int, int, float, int]]
    top: float


def merge_tree_layout(tree, *, min_persistence: float = 0.0, max_basins: int | None = 2000) -> MergeTreeLayout:
    """Crossing-free dendrogram layout of the basins selected by ``min_persistence`` and ``max_basins``.

    Each subtree owns a contiguous block of slots. A basin's children, sorted
    by saddle energy, take sides alternately right, left, right, … so the
    lowest saddle sits nearest. Every basin strictly between a connector's
    ends merged at or below that connector, so no connector crosses a branch.
    """
    tree = _as_tree(tree)
    ws = tree.ws
    if not ws.basins:
        raise ValueError("the tree has no basins")
    ids = _select_basins(ws.basins, ws.merges, min_persistence, max_basins)
    children: dict[int, list[int]] = {b: [] for b in ids}
    roots: list[int] = []
    for b in ids:
        parent = tree.node(b).parent
        if parent is None:
            roots.append(b)
        else:
            children[parent].append(b)        # the parent is drawn: the selection is closed under parent
    for b in ids:
        children[b].sort(key=lambda c: (tree.node(c).saddle_to_parent[1], c))

    # Left-to-right order of the slots, iteratively (a tree can be thousands deep).
    order: list[int] = []
    stack: list[tuple[bool, int]] = [(False, r) for r in reversed(roots)]
    while stack:
        is_leaf, b = stack.pop()
        if is_leaf:
            order.append(b)
            continue
        kids = children[b]
        right, left = kids[0::2], kids[1::2]                     # nearest first on each side
        sequence = [(False, c) for c in reversed(left)] + [(True, b)] + [(False, c) for c in right]
        stack.extend(reversed(sequence))
    x = {b: i for i, b in enumerate(order)}

    e_min = {b: tree.node(b).minimum_energy for b in ids}
    connectors = sorted(
        ((x[b], x[tree.node(b).parent], tree.node(b).saddle_to_parent[1], b) for b in ids if tree.node(b).parent is not None),
        key=lambda c: (c[2], c[3]),
    )
    e_lo, e_hi = min(e_min.values()), max(e_min.values())        # a root may sit above every saddle (forests)
    if connectors:
        e_hi = max(e_hi, connectors[-1][2])
    top = e_hi + 0.05 * (e_hi - e_lo) if e_hi > e_lo else e_hi + 1.0
    branches = {
        b: (x[b], e_min[b], top if tree.node(b).parent is None else tree.node(b).saddle_to_parent[1]) for b in ids
    }
    return MergeTreeLayout(ids, x, branches, connectors, top)
