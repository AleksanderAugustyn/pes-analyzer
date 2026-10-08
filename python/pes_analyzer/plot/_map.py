"""2-D maps of raw grid values, and paths projected onto two axes."""
from __future__ import annotations

import operator
from collections.abc import Mapping

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes as MplAxes
from matplotlib.colors import BoundaryNorm

from ..grid import Axes, _normalize_axes, index_to_coords

__all__ = ["plot_map", "plot_path"]


def _cell_edges(a: np.ndarray) -> np.ndarray:
    """Edges around the coordinates ``a``: midpoints inside, half a step beyond the ends."""
    inner = 0.5 * (a[:-1] + a[1:])
    return np.concatenate(([a[0] - 0.5 * (a[1] - a[0])], inner, [a[-1] + 0.5 * (a[-1] - a[-2])]))


def _map_axes(values: np.ndarray, axes: Axes) -> tuple[list[np.ndarray], list[str]]:
    """Coordinate arrays and axis labels of a 2-D map."""
    if axes is None:
        return [np.arange(n, dtype=np.float64) for n in values.shape], ["index 0", "index 1"]
    coords = _normalize_axes(axes, values.shape)
    names = [str(k) for k in axes] if isinstance(axes, Mapping) else ["", ""]
    return coords, names


def plot_map(
    values: npt.ArrayLike,
    axes: Axes = None,
    *,
    ax: MplAxes | None = None,
    cmap=None,
    levels: npt.ArrayLike | None = None,
    contours: npt.ArrayLike | None = None,
    mask: npt.ArrayLike | None = None,
    colorbar: bool = True,
    label: str = "energy",
    **mesh_kw,
) -> MplAxes:
    """Draw a 2-D map with ``pcolormesh`` on cell edges from the axis coordinates.

    ``values[i, j]`` sits at ``x = axes[0][i]``, ``y = axes[1][j]``: axis 0 is
    horizontal, the order of ``keep`` in ``minimize_grid``. Nothing is
    interpolated; ``NaN`` cells stay blank. ``levels`` gives discrete bands
    (``BoundaryNorm``, under/over colours outside); ``contours`` adds black
    labelled contour lines through the cell centres; ``mask`` hatches the
    ``True`` cells. ``mesh_kw`` reaches ``pcolormesh``. Returns ``ax``.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or min(v.shape) < 2:
        raise ValueError(f"values must be a 2-D array with at least two cells per side, got shape {v.shape}")
    coords, names = _map_axes(v, axes)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != v.shape:
            raise ValueError(f"mask has shape {m.shape}, expected {v.shape}")
    if levels is not None and "norm" in mesh_kw:
        raise ValueError("pass either levels= or norm=, not both")
    kw: dict = {"shading": "flat", "rasterized": True, "cmap": cmap}
    if levels is not None:
        lv = np.asarray(levels, dtype=np.float64)
        if lv.ndim != 1 or lv.size < 2 or not (np.diff(lv) > 0).all():
            raise ValueError("levels must be a strictly increasing sequence of at least two values")
        colormap = matplotlib.colormaps.get_cmap(cmap)
        kw["norm"] = BoundaryNorm(lv, colormap.N, extend="both")
        kw["cmap"] = colormap
    kw.update(mesh_kw)
    if ax is None:
        ax = plt.gca()
    edges_x, edges_y = _cell_edges(coords[0]), _cell_edges(coords[1])
    field = np.ma.masked_invalid(v.T)
    mesh = ax.pcolormesh(edges_x, edges_y, field, **kw)
    if contours is not None:
        cs = ax.contour(coords[0], coords[1], field, levels=list(np.asarray(contours, dtype=np.float64)),
                        colors="black", linewidths=0.8)
        ax.clabel(cs, fmt="%g", fontsize=7)
    if mask is not None:
        overlay = np.ma.masked_where(~m.T, np.ones(m.T.shape))
        # keep the masked array: pcolor draws only unmasked quads, and set_array(None) would unmask every one
        ax.pcolor(edges_x, edges_y, overlay, hatch="///", facecolor="none", edgecolor="black", linewidth=0.0, zorder=2)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=label)
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    return ax


def plot_path(
    indices: npt.ArrayLike,
    axes: Axes = None,
    *,
    keep: tuple[int, int] = (0, 1),
    ax: MplAxes | None = None,
    **line_kw,
) -> MplAxes:
    """Draw a ``(K, N)`` path, or ``(M, N)`` points, projected onto the two axes ``keep``.

    Coordinates come from ``index_to_coords`` (the indices themselves when
    ``axes`` is ``None``). Default style: a line with a marker at every cell;
    ``line_kw`` overrides it (``linestyle="none", marker="o"`` for points).
    """
    idx = np.asarray(indices)
    if idx.ndim != 2 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("indices must be an integer array of shape (K, N)")
    n = idx.shape[1]
    kept = tuple(operator.index(a) for a in keep)          # TypeError for 0.8: never a silent truncation
    if len(kept) != 2 or kept[0] == kept[1] or not all(0 <= a < n for a in kept):
        raise ValueError(f"keep must name two distinct axes in [0, {n}), got {keep!r}")
    coords = idx.astype(np.float64) if axes is None else index_to_coords(idx, axes)
    if ax is None:
        ax = plt.gca()
    kw: dict = {"marker": "."}
    kw.update(line_kw)
    ax.plot(coords[:, kept[0]], coords[:, kept[1]], **kw)
    return ax
