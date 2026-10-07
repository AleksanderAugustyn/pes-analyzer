"""The analytic surfaces: every listed critical point is one, with the listed energy and incidence."""

from __future__ import annotations

import numpy as np
import pytest

from pes_analyzer.synthetic import AnalyticSurface, hidden_barrier, muller_brown, separable_wells

SURFACES = [muller_brown(), separable_wells(3), hidden_barrier()]


def num_grad(f, p, h=1e-6):
    p = np.asarray(p, dtype=float)
    g = np.zeros(p.size)
    for i in range(p.size):
        d = np.zeros(p.size)
        d[i] = h
        g[i] = (float(f(*(p + d))) - float(f(*(p - d)))) / (2 * h)
    return g


def num_hess(f, p, h=1e-4):
    p = np.asarray(p, dtype=float)
    H = np.zeros((p.size, p.size))
    for i in range(p.size):
        d = np.zeros(p.size)
        d[i] = h
        H[:, i] = (num_grad(f, p + d, 1e-5) - num_grad(f, p - d, 1e-5)) / (2 * h)
    return (H + H.T) / 2


def flow_to_minimum(surf, p, step=5e-3, radius=0.05, limit=20000):
    """Normalised gradient descent until a listed minimum is within ``radius``."""
    p = np.asarray(p, dtype=float).copy()
    mins = np.array([m[0] for m in surf.minima])
    for _ in range(limit):
        d = np.linalg.norm(mins - p, axis=1)
        if d.min() < radius:
            return int(d.argmin())
        g = num_grad(surf, p)
        p -= step * g / np.linalg.norm(g)
    raise AssertionError("gradient flow did not reach a listed minimum")


@pytest.mark.parametrize("surf", SURFACES, ids=lambda s: s.name)
def test_minima_are_minima_with_the_listed_energy(surf):
    assert isinstance(surf, AnalyticSurface)
    energies = [e for _, e in surf.minima]
    assert energies == sorted(energies)
    for coords, energy in surf.minima:
        assert len(coords) == surf.ndim
        assert abs(float(surf(*coords)) - energy) < 1e-7 * max(1.0, abs(energy))
        assert np.abs(num_grad(surf, coords)).max() < 1e-6
        assert (np.linalg.eigvalsh(num_hess(surf, coords)) > 0).all()


@pytest.mark.parametrize("surf", SURFACES, ids=lambda s: s.name)
def test_saddles_are_index_one_and_join_the_listed_minima(surf):
    energies = [e for _, e, _ in surf.saddles]
    assert energies == sorted(energies)
    for coords, energy, (i, j) in surf.saddles:
        assert i < j < len(surf.minima)
        assert abs(float(surf(*coords)) - energy) < 1e-7 * max(1.0, abs(energy))
        assert np.abs(num_grad(surf, coords)).max() < 1e-6
        w, v = np.linalg.eigh(num_hess(surf, coords))
        assert (w < 0).sum() == 1
        ends = sorted(flow_to_minimum(surf, np.asarray(coords) + s * 2e-2 * v[:, 0]) for s in (1, -1))
        assert tuple(ends) == (i, j)


def test_separable_wells_counts_and_default_tilts():
    for n in (2, 5, 7):
        s = separable_wells(n)
        assert s.ndim == n and len(s.minima) == 2**n and len(s.saddles) == n * 2 ** (n - 1)
    coords, energy = separable_wells(2).minima[0]
    assert coords == (-1.0, -1.0) and energy == pytest.approx(-0.06)
    with pytest.raises(ValueError):
        separable_wells(2, tilts=[0.1, 3.0])
    with pytest.raises(ValueError):
        separable_wells(2, tilts=[0.1])


def test_hidden_barrier_defaults_and_bounds():
    s = hidden_barrier()
    assert s.minima[0] == ((-1.0, 0.0, -1.0), -0.5)
    assert s.saddles[0][0] == (-0.1875, 0.0, 1.0) and abs(s.saddles[0][1] - 1.069901) < 1e-6
    assert s.saddles[-1][0] == (1.0, 0.0, -0.0375) and abs(s.saddles[-1][1] - 5.014059) < 1e-6
    assert s.saddles[-1][2] == (1, 3)
    for kwargs in ({"t": 2.0, "h": 1.0}, {"t": 1.8, "h": 1.0}, {"t": 0.0}, {"b": -1.0}, {"w": 0.0}):
        with pytest.raises(ValueError):
            hidden_barrier(**kwargs)
    hidden_barrier(t=1.7, h=1.0)   # just inside (16/9) sqrt(bh)


def test_call_and_sample_shapes():
    s = hidden_barrier()
    assert float(s(-1.0, 0.0, -1.0)) == -0.5
    with pytest.raises(ValueError):
        s(0.0, 0.0)
    axes = {"x": np.linspace(-1.5, 1.5, 7), "y": np.linspace(-0.5, 0.5, 3), "z": np.linspace(-1.5, 1.5, 5)}
    E = s.sample(axes)
    assert E.shape == (7, 3, 5) and E.dtype == np.float64 and E.flags["C_CONTIGUOUS"]
    assert E[1, 1, 1] == float(s(axes["x"][1], axes["y"][1], axes["z"][1]))
    with pytest.raises(ValueError):
        s.sample([axes["x"], axes["y"]])


def test_tilts_are_copied_and_nan_parameters_rejected():
    t = np.array([0.02, 0.04])
    s = separable_wells(2, tilts=t)
    t[:] = 1.0                                   # the caller reuses its array
    coords, energy = s.minima[0]
    assert float(s(*coords)) == pytest.approx(energy)
    for kwargs in ({"w": np.nan}, {"h": np.nan}, {"b": np.nan}, {"t": np.nan}):
        with pytest.raises(ValueError):
            hidden_barrier(**kwargs)
