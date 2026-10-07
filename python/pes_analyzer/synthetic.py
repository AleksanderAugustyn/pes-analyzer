"""pes_analyzer.synthetic: analytic surfaces with known critical points.

Each constructor returns an :class:`AnalyticSurface` whose ``minima`` and
index-1 ``saddles`` are exact (closed form, or Newton-refined to ten
decimals for Müller–Brown). ``sample(axes)`` evaluates it on a grid that the
other functions of the package consume. Pure NumPy.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pes_analyzer.grid import Axes, _normalize_axes

__all__ = ["AnalyticSurface", "hidden_barrier", "muller_brown", "separable_wells"]

Minimum = tuple[tuple[float, ...], float]
Saddle = tuple[tuple[float, ...], float, tuple[int, int]]


@dataclass(frozen=True)
class AnalyticSurface:
    """An analytic surface with its minima and index-1 saddles.

    ``minima`` holds ``(coords, energy)`` ascending by energy; ``saddles``
    holds ``(coords, energy, (i, j))`` ascending by energy, where ``i < j``
    are the positions in ``minima`` of the two minima the saddle joins along
    its unstable direction.
    """

    name: str
    ndim: int
    minima: tuple[Minimum, ...]
    saddles: tuple[Saddle, ...]
    _func: Callable[..., npt.NDArray[np.float64]] = field(repr=False, compare=False)

    def __call__(self, *coords: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate on ``ndim`` broadcastable coordinate arrays."""
        if len(coords) != self.ndim:
            raise ValueError(f"{self.name} takes {self.ndim} coordinates, got {len(coords)}")
        return np.asarray(self._func(*(np.asarray(c, dtype=np.float64) for c in coords)), dtype=np.float64)

    def sample(self, axes: Axes) -> npt.NDArray[np.float64]:
        """Dense C-contiguous float64 grid on ``axes`` (mapping or sequence of 1-D arrays)."""
        ax = _normalize_axes(axes, None)
        if ax is None or len(ax) != self.ndim:
            raise ValueError(f"{self.name} needs {self.ndim} coordinate arrays")
        mesh = np.meshgrid(*ax, indexing="ij", sparse=True)
        shape = tuple(a.shape[0] for a in ax)
        return np.ascontiguousarray(np.broadcast_to(self(*mesh), shape))


def _s(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # s(±1) = ±1 and s'(±1) = 0: a tilt that leaves the wells at ±1 exactly.
    return (3.0 * u - u**3) / 2.0


_MB_A = np.array([-200.0, -100.0, -170.0, 15.0])
_MB_a = np.array([-1.0, -1.0, -6.5, 0.7])
_MB_b = np.array([0.0, 0.0, 11.0, 0.6])
_MB_c = np.array([-10.0, -10.0, -6.5, 0.7])
_MB_X0 = np.array([1.0, 0.0, -0.5, -1.0])
_MB_Y0 = np.array([0.0, 0.5, 1.5, 1.0])


def _muller_brown(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    dx = x[..., None] - _MB_X0
    dy = y[..., None] - _MB_Y0
    return (_MB_A * np.exp(_MB_a * dx * dx + _MB_b * dx * dy + _MB_c * dy * dy)).sum(axis=-1)


def muller_brown() -> AnalyticSurface:
    """The 2-D Müller–Brown surface (1979 parameters); three minima, two saddles."""
    minima = (
        ((-0.5582236346, 1.4417258418), -146.69951721),
        ((0.6234994049, 0.0280377585), -108.16672412),
        ((-0.0500108231, 0.4666941049), -80.76781813),
    )
    saddles = (
        ((0.2124865820, 0.2929883251), -72.24894011, (1, 2)),
        ((-0.8220015587, 0.6243128028), -40.66484351, (0, 2)),
    )
    return AnalyticSurface("muller_brown", 2, minima, saddles, _muller_brown)


def separable_wells(ndim: int, tilts: npt.ArrayLike | None = None) -> AnalyticSurface:
    """Sum of tilted double wells, ``E = Σ (x_a² − 1)² + τ_a s(x_a)``.

    2ᴺ minima at the corners ``x_a = ±1`` with energy ``Σ ±τ_a`` and
    ``N · 2ᴺ⁻¹`` saddles with one axis at ``3τ_a/8``. Default
    ``τ_a = 0.02 · 2ᵃ``; every tilt must lie in ``(0, 8/3)``.
    """
    if ndim < 1:
        raise ValueError("ndim must be at least 1")
    t = 0.02 * 2.0 ** np.arange(ndim) if tilts is None else np.asarray(tilts, dtype=np.float64)
    if t.shape != (ndim,) or not ((t > 0) & (t < 8.0 / 3.0)).all():
        raise ValueError("tilts must hold ndim values in (0, 8/3)")

    def func(*x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return sum((xa * xa - 1.0) ** 2 + ta * _s(xa) for xa, ta in zip(x, t))

    minima = sorted(
        ((signs, float(np.dot(signs, t))) for signs in itertools.product((-1.0, 1.0), repeat=ndim)),
        key=lambda m: (m[1], m[0]),
    )
    position = {coords: i for i, (coords, _) in enumerate(minima)}
    saddles: list[Saddle] = []
    for a in range(ndim):
        u = 3.0 * t[a] / 8.0
        barrier = (u * u - 1.0) ** 2 + t[a] * _s(u)
        for signs in itertools.product((-1.0, 1.0), repeat=ndim - 1):
            rest = float(np.dot(signs, np.delete(t, a)))
            lo = signs[:a] + (-1.0,) + signs[a:]
            hi = signs[:a] + (1.0,) + signs[a:]
            joins = (min(position[lo], position[hi]), max(position[lo], position[hi]))
            saddles.append((signs[:a] + (float(u),) + signs[a:], float(barrier + rest), joins))
    saddles.sort(key=lambda s: (s[1], s[0]))
    return AnalyticSurface(f"separable_wells_{ndim}d", ndim, tuple(minima), tuple(saddles), func)


def hidden_barrier(b: float = 1.0, h: float = 5.0, t: float = 0.5, w: float = 1.0) -> AnalyticSurface:
    """3-D surface whose barrier a map minimised over ``z`` hides.

    ``E = b (x² − 1)² + w y² + h (z² − 1)² − t s(x) s(z)``. Minimised over
    ``z`` the map shows a barrier of ``b + t`` between the two deep minima;
    the real one is ``max(E_x, E_z) + t``. Requires ``b, h, w > 0`` and
    ``0 < t < min(8·min(b, h)/3, 16·sqrt(b·h)/9)``: the first bound keeps the
    saddles on the edges, the second keeps the origin a maximum.
    """
    if min(b, h, w) <= 0:
        raise ValueError("b, h and w must be positive")
    if not 0 < t < min(8.0 * min(b, h) / 3.0, 16.0 / 9.0 * math.sqrt(b * h)):
        raise ValueError("t must lie in (0, min(8*min(b, h)/3, 16*sqrt(b*h)/9))")

    def func(x, y, z):
        return b * (x * x - 1.0) ** 2 + w * y * y + h * (z * z - 1.0) ** 2 - t * _s(x) * _s(z)

    xs, zs = 3.0 * t / (8.0 * b), 3.0 * t / (8.0 * h)
    e_x = b * (xs * xs - 1.0) ** 2 + t * _s(xs)
    e_z = h * (zs * zs - 1.0) ** 2 + t * _s(zs)
    minima = (
        ((-1.0, 0.0, -1.0), -t),
        ((1.0, 0.0, 1.0), -t),
        ((-1.0, 0.0, 1.0), t),
        ((1.0, 0.0, -1.0), t),
    )
    saddles = sorted(
        [
            ((-xs, 0.0, 1.0), e_x, (1, 2)),
            ((xs, 0.0, -1.0), e_x, (0, 3)),
            ((-1.0, 0.0, zs), e_z, (0, 2)),
            ((1.0, 0.0, -zs), e_z, (1, 3)),
        ],
        key=lambda s: (s[1], s[0]),
    )
    return AnalyticSurface("hidden_barrier", 3, minima, tuple(saddles), func)
