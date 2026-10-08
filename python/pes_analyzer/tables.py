"""Tables of minima, basins and paths as plain column dicts, with one CSV writer and reader.

A table is ``dict[str, numpy.ndarray]``: every column 1-D and of one length,
insertion order is column order, so ``pandas.DataFrame(table)`` works for
anyone who wants it. Columns hold integers, booleans and floats only.
See ``_docs/API.md`` section ``tables``.
"""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from os import PathLike

import numpy as np
import numpy.typing as npt

from .grid import Axes, index_to_coords, path_length
from .topology._tree import _select_basins
from .topology.merge_tree import _as_tree

Table = dict[str, np.ndarray]

__all__ = ["Table", "minima_table", "basins_table", "path_table", "write_csv", "read_csv"]


def _axis_names(axes: Axes, ndim: int) -> list[str]:
    names = [str(k) for k in axes] if isinstance(axes, Mapping) else [f"x{a}" for a in range(len(axes))]
    if len(names) != ndim:
        raise ValueError(f"axes has {len(names)} entries, but the indices have {ndim} columns")
    return names


def _index_columns(indices: np.ndarray, prefix: str = "") -> Table:
    return {f"{prefix}index_{a}": indices[:, a].astype(np.int64) for a in range(indices.shape[1])}


def _coordinate_columns(indices: np.ndarray, axes: Axes, prefix: str = "") -> Table:
    """``{prefix + name: coordinates}`` for an ``(M, N)`` index array; empty when ``axes`` is ``None``.

    Rows whose first index is negative (sentinels) get ``NaN`` coordinates.
    """
    if axes is None:
        return {}
    names = _axis_names(axes, indices.shape[1])
    coords = np.full(indices.shape, np.nan)
    valid = indices[:, 0] >= 0
    if valid.any():
        coords[valid] = index_to_coords(indices[valid], axes)
    return {prefix + name: coords[:, a].copy() for a, name in enumerate(names)}


def _assemble(*parts: Table) -> Table:
    out: Table = {}
    for part in parts:
        for name, column in part.items():
            if name in out:
                raise ValueError(f"column {name!r} is defined twice: an axis name collides with a table column")
            out[name] = column
    return out


def minima_table(
    points: Sequence[tuple[Sequence[int], float]],
    axes: Axes = None,
    *,
    ndim: int | None = None,
) -> Table:
    """Any list of ``(index, energy)`` pairs as a table, rows in input order.

    N comes from the first point, else from ``axes``, else from ``ndim``; an
    empty list with neither is a ``ValueError``.
    """
    if len(points):
        n = len(points[0][0])
        if ndim is not None and ndim != n:
            raise ValueError(f"ndim={ndim} but the points have {n} indices")
    elif axes is not None:
        n = len(axes)
    elif ndim is not None:
        n = int(ndim)
    else:
        raise ValueError("minima_table needs axes= or ndim= to size an empty table")
    indices = np.array([tuple(p[0]) for p in points], dtype=np.int64).reshape(len(points), n)
    energies = np.array([float(p[1]) for p in points], dtype=np.float64)
    return _assemble(_index_columns(indices), _coordinate_columns(indices, axes), {"energy": energies})


def path_table(indices: npt.ArrayLike, energies: npt.ArrayLike, axes: Axes = None) -> Table:
    """A ``(K, N)`` path and the energies at its cells as a table with the cumulative length."""
    idx = np.asarray(indices)
    if idx.ndim != 2 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("indices must be an integer array of shape (K, N)")
    e = np.asarray(energies, dtype=np.float64)
    if e.shape != (idx.shape[0],):
        raise ValueError(f"energies has shape {e.shape}, expected ({idx.shape[0]},)")
    return _assemble(
        {"step": np.arange(idx.shape[0], dtype=np.int64)},
        _index_columns(idx),
        _coordinate_columns(idx, axes),
        {"length": path_length(idx, axes), "energy": e.copy()},
    )


def basins_table(tree, axes: Axes = None, *, min_persistence: float = 0.0) -> Table:
    """The merge tree as a table: one row per basin with its minimum, parent, death saddle and persistence.

    Every root has ``parent`` −1, saddle indices −1, saddle coordinates and
    ``saddle_energy`` ``NaN``, ``persistence`` ``inf``. ``min_persistence``
    selects rows as in ``merge_tree_layout``; the parent of a selected basin
    is always selected too.
    """
    tree = _as_tree(tree)
    ws = tree.ws
    if not ws.basins:
        raise ValueError("the tree has no basins")
    ids = _select_basins(ws.basins, ws.merges, min_persistence)
    n, ndim = len(ids), len(ws.basins[0][0])
    min_idx = np.array([ws.basins[b][0] for b in ids], dtype=np.int64).reshape(n, ndim)
    min_e = np.array([ws.basins[b][1] for b in ids], dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    saddle_idx = np.full((n, ndim), -1, dtype=np.int64)
    saddle_e = np.full(n, np.nan)
    persistence = np.array([tree.node(b).persistence for b in ids], dtype=np.float64)
    for r, b in enumerate(ids):
        node = tree.node(b)
        if node.parent is not None:
            parent[r] = node.parent
            saddle_idx[r] = node.saddle_to_parent[0]
            saddle_e[r] = node.saddle_to_parent[1]
    return _assemble(
        {"basin": np.array(ids, dtype=np.int64)},
        _index_columns(min_idx, "min_"),
        _coordinate_columns(min_idx, axes, "min_"),
        {"min_energy": min_e, "parent": parent},
        _index_columns(saddle_idx, "saddle_"),
        _coordinate_columns(saddle_idx, axes, "saddle_"),
        {"saddle_energy": saddle_e, "persistence": persistence},
    )


def _fmt_int(v) -> str:
    return str(int(v))


def _fmt_float(v) -> str:
    return repr(float(v))


def write_csv(path: str | PathLike[str], table: Mapping[str, npt.ArrayLike]) -> None:
    """Write a table as CSV: header row, comma separated, ``\\n`` endings, UTF-8.

    Integers and booleans are written as integers, floats as ``repr`` (the
    shortest round-trip form, ``nan``/``inf``/``-inf`` spelled so). Any other
    dtype is a ``ValueError``: the format is numeric only so that
    :func:`read_csv` can always invert it. An existing file is overwritten.
    """
    columns: list[tuple[np.ndarray, object]] = []
    length: int | None = None
    for name, values in table.items():
        col = np.asarray(values)
        if col.ndim != 1:
            raise ValueError(f"column {name!r} must be 1-D, got shape {col.shape}")
        if length is None:
            length = col.shape[0]
        elif col.shape[0] != length:
            raise ValueError(f"column {name!r} has {col.shape[0]} rows, expected {length}")
        if col.dtype.kind in "iub":
            columns.append((col, _fmt_int))
        elif col.dtype.kind == "f":
            columns.append((col, _fmt_float))
        else:
            raise ValueError(
                f"column {name!r} has dtype {col.dtype}; tables hold integer, boolean and floating columns only"
            )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(list(table))
        for r in range(length or 0):
            writer.writerow([fmt(col[r]) for col, fmt in columns])


def read_csv(path: str | PathLike[str]) -> Table:
    """Read a file written by :func:`write_csv`.

    A column is ``int64`` when every entry parses as an integer, else
    ``float64``. A header-only file carries no type information: its columns
    come back zero-length ``float64``. Not a general CSV reader.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path}: empty file, expected a header row")
    names, body = rows[0], rows[1:]
    out: Table = {}
    for c, name in enumerate(names):
        cells = [row[c] for row in body]
        if not cells:
            out[name] = np.empty(0, dtype=np.float64)
            continue
        try:
            out[name] = np.array([int(v) for v in cells], dtype=np.int64)
        except ValueError:
            out[name] = np.array([float(v) for v in cells], dtype=np.float64)
    return out
