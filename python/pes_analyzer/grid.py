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


def _normalize_keep(keep, ndim: int) -> tuple[int, ...]:
    if isinstance(keep, (int, np.integer)):
        keep_t = (keep,)
    elif isinstance(keep, (Sequence, np.ndarray)) and not isinstance(keep, str):
        keep_t = tuple(keep)
    else:
        raise ValueError(f"keep must be an axis index or a sequence of axis indices, got {keep!r}")
    if not all(isinstance(k, (int, np.integer)) for k in keep_t):
        raise ValueError("keep must hold integer axis indices")
    keep_t = tuple(int(k) for k in keep_t)
    if not 1 <= len(keep_t) <= ndim - 1:
        raise ValueError(f"keep must name 1 to {ndim - 1} axes, got {len(keep_t)}")
    if any(not 0 <= k < ndim for k in keep_t) or len(set(keep_t)) != len(keep_t):
        raise ValueError(f"keep must hold distinct axis indices in [0, {ndim}), got {keep_t}")
    return keep_t


def minimize_grid(
    energies: npt.ArrayLike,
    keep: int | Sequence[int],
    *,
    threads: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray[np.intp]]:
    """Minimise an N-D grid over every axis not in ``keep``.

    Returns ``(minimum, index)``: ``minimum`` has the kept axes in the order
    of ``keep`` and the dtype of ``energies`` (``NaN`` where a column has no
    valid cell); ``index`` has shape ``minimum.shape + (N,)`` and holds the
    full N-D index of the minimising cell in the original axis order, or
    ``-1`` in every entry where there is none. ``NaN`` cells are ignored, as
    in ``np.nanmin``. Ties go to the first cell in C order over the hidden
    axes. Work is split into slabs
    along the first kept axis on ``threads`` workers (default
    ``os.cpu_count()``); the result does not depend on ``threads``.
    """
    e = np.asarray(energies)
    if not np.issubdtype(e.dtype, np.floating):
        raise ValueError("energies must be a floating-point array")
    n = e.ndim
    if n < 2:
        raise ValueError("energies must have at least 2 axes")
    keep_t = _normalize_keep(keep, n)
    if threads is None:
        threads = os.cpu_count() or 1
    if not isinstance(threads, (int, np.integer)) or threads < 1:
        raise ValueError("threads must be a positive integer")
    hidden = tuple(a for a in range(n) if a not in keep_t)
    kept_shape = tuple(e.shape[k] for k in keep_t)
    hidden_shape = tuple(e.shape[h] for h in hidden)
    moved = np.transpose(e, keep_t + hidden)          # a view: kept axes first, hidden axes in original order
    minimum = np.full(kept_shape, np.nan, dtype=e.dtype)
    index = np.full(kept_shape + (n,), -1, dtype=np.intp)
    rest = np.indices(kept_shape[1:], dtype=np.intp)

    def work(i: int) -> None:
        flat = moved[i].reshape(kept_shape[1:] + (-1,))                       # copies if not contiguous
        isnan = np.isnan(flat)
        arg = np.asarray(np.where(isnan, np.inf, flat).argmin(axis=-1))
        val = np.take_along_axis(flat, arg[..., None], axis=-1)[..., 0]
        # a column whose only valid cells are +inf: the argmin above may have landed on a
        # NaN cell; take its first non-NaN cell instead, as np.nanmin would
        inf_only = np.asarray(np.isnan(val) & ~isnan.all(axis=-1))
        if inf_only.any():
            arg = np.where(inf_only, isnan.argmin(axis=-1), arg)
            val = np.take_along_axis(flat, arg[..., None], axis=-1)[..., 0]
        valid = np.asarray(~np.isnan(val))
        out_idx = index[i]
        out_idx[..., keep_t[0]] = i
        for j, k in enumerate(keep_t[1:]):
            out_idx[..., k] = rest[j]
        for h, hid in zip(hidden, np.unravel_index(arg, hidden_shape)):
            out_idx[..., h] = hid
        out_idx[~valid] = -1
        minimum[i] = val

    with ThreadPoolExecutor(max_workers=max(1, min(int(threads), kept_shape[0]))) as pool:
        list(pool.map(work, range(kept_shape[0])))
    return minimum, index


def jump_map(index: npt.ArrayLike, keep: int | Sequence[int]) -> npt.NDArray[np.float64]:
    """How far the minimiser moves between neighbouring map points, in cells.

    For each map point with a minimiser: the largest Chebyshev distance over
    the hidden axes between its minimiser and that of any von Neumann map
    neighbour with one; ``0.0`` if no neighbour has one; ``NaN`` where the
    point itself has none. A value of at most 1 means the map is continuous
    there to within one hidden step.
    """
    idx = np.asarray(index)
    if idx.ndim < 2 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("index must be an integer array of shape kept_shape + (N,)")
    n = idx.shape[-1]
    keep_t = _normalize_keep(keep, n)
    if idx.ndim != len(keep_t) + 1:
        raise ValueError("index must have shape kept_shape + (N,)")
    hid = idx[..., [a for a in range(n) if a not in keep_t]]
    valid = idx[..., 0] >= 0
    out = np.zeros(valid.shape, dtype=np.float64)
    for ax in range(len(keep_t)):
        lo = [slice(None)] * len(keep_t)
        hi = list(lo)
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        d = np.abs(hid[hi] - hid[lo]).max(axis=-1).astype(np.float64)
        d[~(valid[lo] & valid[hi])] = 0.0
        np.maximum(out[lo], d, out=out[lo])
        np.maximum(out[hi], d, out=out[hi])
    out[~valid] = np.nan
    return out
