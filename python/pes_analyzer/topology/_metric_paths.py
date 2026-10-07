"""Paths whose result depends on step lengths or on a search over the grid.

Thin wrappers over the Rust kernels ``topology::steepest`` and
``topology::dijkstra``. Contracts in ``_docs/API.md``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pes_analyzer._native import topology as _native_topology
from pes_analyzer.grid import Axes, _normalize_axes

__all__ = ["find_least_action_path", "find_minimum_ascent_path", "find_steepest_descent_path"]

PathResult = tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]


def _axes_arg(axes: Axes, shape: tuple[int, ...]) -> list[list[float]] | None:
    ax = _normalize_axes(axes, shape)
    return None if ax is None else [a.tolist() for a in ax]


def find_steepest_descent_path(
    energies: npt.ArrayLike,
    start: tuple[int, ...],
    *,
    axes: Axes = None,
    neighborhood: str = "moore",
) -> PathResult:
    """Follow the largest downward slope from ``start`` until no neighbour is lower.

    Returns ``(path_indices, path_energies)``; row 0 is ``start``. With
    ``axes`` the slope is ΔE over the physical step length. Equal slopes go
    to the smaller linear index. See ``_docs/API.md``.
    """
    energies = np.asarray(energies)
    return _native_topology.find_steepest_descent_path(
        energies, tuple(int(i) for i in start), _axes_arg(axes, energies.shape), neighborhood
    )


def _split_end(end, shape: tuple[int, ...]):
    """``(index, None)`` for an index tuple, ``(None, mask)`` for a boolean mask of the grid shape."""
    if isinstance(end, np.ndarray) and end.dtype == np.bool_:
        if end.shape != shape:
            raise ValueError(f"end mask shape {end.shape} does not match the grid shape {shape}")
        if not end.flags["C_CONTIGUOUS"]:
            raise ValueError("end mask must be C-contiguous; call np.ascontiguousarray(mask) if you intend a copy")
        if not end.any():
            raise ValueError("end mask has no True cell")
        return None, end
    if isinstance(end, np.ndarray) and end.ndim != 1:
        raise ValueError("end must be an index tuple or a boolean mask with the shape of the grid")
    return tuple(int(i) for i in end), None


def _search(field, start, end, rule: str, axes: Axes, neighborhood: str) -> PathResult | None:
    field = np.asarray(field)
    index, mask = _split_end(end, field.shape)
    return _native_topology.find_search_path(
        field, tuple(int(i) for i in start), index, mask, rule, _axes_arg(axes, field.shape), neighborhood
    )


def find_least_action_path(
    cost: npt.ArrayLike,
    start: tuple[int, ...],
    end: tuple[int, ...] | npt.NDArray[np.bool_],
    *,
    axes: Axes = None,
    neighborhood: str = "moore",
) -> PathResult | None:
    """Path minimising ∫ cost ds (trapezoid rule per step) from ``start`` to ``end``.

    ``end`` is one index or a boolean mask of target cells; the search stops
    at the first target reached. Among equal-action paths the shortest wins.
    Returns ``(path_indices, path_action)`` with the cumulative action per
    cell, or ``None`` if no target is reachable. ``cost`` must be
    non-negative; ``NaN`` is a wall. See ``_docs/API.md``.
    """
    return _search(cost, start, end, "cost", axes, neighborhood)


def find_minimum_ascent_path(
    energies: npt.ArrayLike,
    start: tuple[int, ...],
    end: tuple[int, ...] | npt.NDArray[np.bool_],
    *,
    axes: Axes = None,
    neighborhood: str = "von_neumann",
) -> PathResult | None:
    """Path minimising the total climb Σ max(ΔE, 0) from ``start`` to ``end``.

    Descents are free. Among equal-climb paths the shortest wins (``axes``
    enter only there). Returns ``(path_indices, path_ascent)`` with the
    cumulative climb per cell, or ``None`` if no target is reachable. See
    ``_docs/API.md``.
    """
    return _search(energies, start, end, "ascent", axes, neighborhood)
