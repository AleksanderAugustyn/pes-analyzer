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
