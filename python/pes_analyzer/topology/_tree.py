"""Persistence-pruned merge-tree analysis for the watershed segmentation.

These helpers are pure Python; the heavy lifting (the flood) is in the
Rust kernel ``find_watershed_segmentation``. See ``_docs/API.md`` for the
full contract.
"""

from __future__ import annotations

import heapq

import numpy as np
import numpy.typing as npt

__all__ = [
    "compute_persistence",
    "prune_merge_tree",
]


def compute_persistence(
    basins: list[tuple[tuple[int, ...], float]],
    merges: list[tuple[tuple[int, ...], float, int, int]],
) -> npt.NDArray[np.float64]:
    """Per-basin topological persistence.

    The deepest basin (``basins[0]``) never dies; its persistence is
    ``+inf``. Every other basin appears as the ``shallower`` end of
    exactly one merge event, and its persistence is the energy gap
    between that saddle and the basin's own minimum.

    Parameters
    ----------
    basins
        Output of :func:`find_watershed_segmentation`.
    merges
        Output of :func:`find_watershed_segmentation`.

    Returns
    -------
    persistence
        ``float64`` array of length ``len(basins)``; ``persistence[i]``
        is the persistence of basin ``i``.
    """
    persistence = np.full(len(basins), np.inf, dtype=np.float64)
    for _saddle_idx, saddle_e, _deeper, shallower in merges:
        persistence[shallower] = saddle_e - basins[shallower][1]
    return persistence


def prune_merge_tree(
    basins: list[tuple[tuple[int, ...], float]],
    merges: list[tuple[tuple[int, ...], float, int, int]],
    threshold: float,
) -> tuple[list[int], list[tuple[tuple[int, ...], float, int, int]]]:
    """Drop basins with persistence below ``threshold`` and their death events.

    Parameters
    ----------
    basins, merges
        Outputs of :func:`find_watershed_segmentation`.
    threshold
        Minimum persistence (same units as energy) for a basin to survive.

    Returns
    -------
    surviving
        Basin IDs (indices into ``basins``) that pass the threshold, in
        ascending order. Always includes basin 0 (its persistence is
        ``+inf``).
    kept
        The subset of ``merges`` where the ``shallower`` basin survives.
        Input order (= ascending saddle energy) is preserved.
    """
    persistence = compute_persistence(basins, merges)
    surviving = [i for i, p in enumerate(persistence) if p >= threshold]
    survivor_set = set(surviving)
    kept = [m for m in merges if m[3] in survivor_set]
    return surviving, kept




def _select_basins(
    basins: list[tuple[tuple[int, ...], float]],
    merges: list[tuple[tuple[int, ...], float, int, int]],
    min_persistence: float = 0.0,
    max_basins: int | None = None,
) -> list[int]:
    """Basin ids to draw or tabulate: survivors of ``min_persistence``, at most
    ``max_basins`` of them (the most persistent, ties to the lower id), closed
    under parent. Roots have infinite persistence, so every root survives both
    rules. The closure walk is a safety net: a parent is at least as persistent
    as each child, so the two rules already give a parent-closed set.
    """
    if max_basins is not None and max_basins < 1:
        raise ValueError("max_basins must be at least 1, or None")
    persistence = compute_persistence(basins, merges)
    surviving = [i for i, p in enumerate(persistence) if p >= min_persistence]
    if max_basins is not None and len(surviving) > max_basins:
        # roots are exempt from the cap (every root is always drawn); the budget goes to the rest,
        # picked with a heap: O(B log n) for B survivors and n drawn, not a full sort of all survivors
        roots = [b for b in surviving if not np.isfinite(persistence[b])]
        budget = max(0, max_basins - len(roots))
        rest = heapq.nsmallest(budget, (b for b in surviving if np.isfinite(persistence[b])),
                               key=lambda b: (-persistence[b], b))
        surviving = roots + rest
    parent = {shallower: deeper for _idx, _e, deeper, shallower in merges}
    chosen = set(surviving)
    for b in list(chosen):
        p = parent.get(b)
        while p is not None and p not in chosen:
            chosen.add(p)
            p = parent.get(p)
    return sorted(chosen)
