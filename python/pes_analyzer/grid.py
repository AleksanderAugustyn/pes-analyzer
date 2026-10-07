"""pes_analyzer.grid: dense N-D ndarray construction from sparse coords.

This is the canonical scatter-to-dense helper paired with the Rust
kernels in ``pes_analyzer.saddle``, ``pes_analyzer.extrema``, and
``pes_analyzer.topology``. Pure NumPy; no polars dependency.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

__all__ = ["build_dense", "index_to_coords", "jump_map", "minimize_grid", "path_length"]

Axes = Mapping[str, npt.ArrayLike] | Sequence[npt.ArrayLike] | None


def build_dense(
    coords: dict[str, npt.NDArray],
    values: npt.NDArray,
    *,
    dtype: npt.DTypeLike | None = None,
) -> tuple[npt.NDArray, dict[str, npt.NDArray]]:
    """Scatter (coord_per_row, value_per_row) data into a dense N-D ndarray.

    Parameters
    ----------
    coords
        Ordered mapping ``{axis_name: coord_per_row_1d_array}``. The
        insertion order of the dict determines the axis order of the
        output ndarray (Python 3.7+ dicts preserve insertion order). Each
        coordinate array must have length equal to ``len(values)``.
    values
        1-D array of scalars (energies or any per-row value).
    dtype
        Output dtype (keyword-only). When ``None`` (default), the dtype of
        ``values`` is preserved if it is floating, else ``np.float64``. An
        explicit ``dtype`` forces the output to that type. Use ``np.float32``
        to halve the memory footprint of large grids.

    Returns
    -------
    dense
        C-contiguous array of shape
        ``(n_unique_axis_0, ..., n_unique_axis_{N-1})`` and dtype per the
        ``dtype`` rule above. Missing cells are ``np.nan``.
    axes
        ``{axis_name: sorted_unique_values_1d_array}``, same key order
        as ``coords``.

    Notes
    -----
    - Axes with only one unique value are NOT squeezed; the caller is
      responsible for filtering active axes before calling.
    - Duplicate ``(coords, ...)`` rows: last-write-wins.

    Raises
    ------
    ValueError
        If any coord array length disagrees with ``len(values)`` or if
        ``coords`` is empty.
    """
    if not coords:
        raise ValueError("coords must contain at least one axis")

    n_rows = len(values)
    for name, arr in coords.items():
        if len(arr) != n_rows:
            raise ValueError(
                f"coord '{name}' has length {len(arr)}, "
                f"but values has length {n_rows}"
            )

    axis_names = list(coords)
    uniques: dict[str, npt.NDArray] = {
        name: np.unique(coords[name]) for name in axis_names
    }
    shape = tuple(len(uniques[name]) for name in axis_names)

    out_dtype = dtype or (
        values.dtype
        if np.issubdtype(np.asarray(values).dtype, np.floating)
        else np.float64
    )
    dense = np.full(shape, np.nan, dtype=out_dtype)
    idx_per_axis = tuple(
        np.searchsorted(uniques[name], coords[name]) for name in axis_names
    )
    dense[idx_per_axis] = np.asarray(values, dtype=out_dtype)
    return dense, uniques


def _normalize_axes(axes: Axes, shape: tuple[int, ...] | None) -> list[npt.NDArray[np.float64]] | None:
    """Validate ``axes`` (mapping or sequence of 1-D arrays) and return float64 copies.

    ``shape=None`` skips the per-axis length check. Every failure is a
    ``ValueError`` so that the Rust wrappers and the pure-Python helpers raise
    the same type for the same mistake.
    """
    if axes is None:
        return None
    values = list(axes.values()) if isinstance(axes, Mapping) else list(axes)
    if shape is not None and len(values) != len(shape):
        raise ValueError(f"axes has {len(values)} entries, but the grid has {len(shape)} axes")
    out: list[npt.NDArray[np.float64]] = []
    for a, vals in enumerate(values):
        arr = np.ascontiguousarray(vals, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"axis {a} must be 1-D, got shape {arr.shape}")
        if shape is not None and arr.shape[0] != shape[a]:
            raise ValueError(
                f"axis {a} has {arr.shape[0]} coordinates, but the grid has {shape[a]} cells along it"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"axis {a} contains a non-finite coordinate")
        if arr.shape[0] > 1 and not (np.diff(arr) > 0).all():
            raise ValueError(f"axis {a} must be strictly increasing")
        out.append(arr)
    return out


def index_to_coords(indices: npt.ArrayLike, axes: Axes) -> npt.NDArray[np.float64]:
    """Physical coordinates of integer grid indices of shape ``(..., N)``.

    Raises ``IndexError`` for an index outside an axis, including the ``-1``
    rows that :func:`minimize_grid` writes where a column has no valid cell.
    """
    idx = np.asarray(indices)
    if idx.ndim < 1 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("indices must be an integer array of shape (..., N)")
    ax = _normalize_axes(axes, None)
    if ax is None or len(ax) != idx.shape[-1]:
        raise ValueError(f"axes must hold {idx.shape[-1]} coordinate arrays")
    out = np.empty(idx.shape, dtype=np.float64)
    for a, vals in enumerate(ax):
        col = idx[..., a]
        if col.size and (col.min() < 0 or col.max() >= vals.shape[0]):
            raise IndexError(f"index out of bounds for axis {a} with size {vals.shape[0]}")
        out[..., a] = vals[col]
    return out


def path_length(indices: npt.ArrayLike, axes: Axes = None) -> npt.NDArray[np.float64]:
    """Cumulative Euclidean length along a ``(K, N)`` index path; ``0`` at row 0.

    Without ``axes`` the coordinates are the indices themselves. Rows need
    not be stencil neighbours.
    """
    idx = np.asarray(indices)
    if idx.ndim != 2 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("indices must be an integer array of shape (K, N)")
    coords = idx.astype(np.float64) if axes is None else index_to_coords(idx, axes)
    out = np.zeros(idx.shape[0], dtype=np.float64)
    if idx.shape[0] > 1:
        np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1), out=out[1:])
    return out
