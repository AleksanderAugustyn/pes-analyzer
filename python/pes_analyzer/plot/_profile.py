"""A quantity along a path against its cumulative length or step number."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes as MplAxes

from ..grid import Axes, path_length

__all__ = ["plot_profile"]


def plot_profile(
    values: npt.ArrayLike,
    *,
    indices: npt.ArrayLike | None = None,
    axes: Axes = None,
    profile=None,
    ax: MplAxes | None = None,
    label: str = "energy",
    **line_kw,
) -> MplAxes:
    """Draw ``values`` (an energy profile, or a cumulative action or climb) along a path.

    x is ``path_length(indices, axes)`` when ``indices`` is given, else the
    step number. ``profile`` (a ``PathProfile`` of the same values) marks its
    minima with circles and its saddles with squares. Returns ``ax``.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"values must be a non-empty 1-D array, got shape {v.shape}")
    if indices is None:
        x = np.arange(v.size, dtype=np.float64)
        xlabel = "step"
    else:
        idx = np.asarray(indices)
        if idx.ndim != 2 or idx.shape[0] != v.size:
            raise ValueError(f"indices must have shape ({v.size}, N), got {idx.shape}")
        x = path_length(idx, axes)
        xlabel = "length" if axes is not None else "length (cells)"
    if ax is None:
        ax = plt.gca()
    kw: dict = {"marker": "."}
    kw.update(line_kw)
    (line,) = ax.plot(x, v, **kw)
    if profile is not None:
        for pairs, marker in ((profile.minima, "o"), (profile.saddles, "s")):
            steps = [int(k) for k, _e in pairs]
            if any(k < 0 or k >= v.size for k in steps):
                raise ValueError("profile refers to a step outside the path")
            if steps:
                ax.plot(x[steps], v[steps], linestyle="none", marker=marker, markersize=7,
                        markerfacecolor=line.get_color(), markeredgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label)
    return ax
