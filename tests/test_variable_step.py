"""The analytic surfaces sampled on uniform, anisotropic and variable-step grids.

The flood and minima kernels never see coordinates, so their output is the
same for any step pattern; what changes is how well the grid resolves the
surface. Tolerances are therefore local to each critical point (spec 8.2).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pes_analyzer.extrema import find_minima_grid
from pes_analyzer.synthetic import hidden_barrier, muller_brown, separable_wells
from pes_analyzer.topology import compute_persistence, find_watershed_segmentation


def uniform(lo, hi, n):
    return np.linspace(lo, hi, n)


def stretched(lo, hi, n, alpha=0.6):
    """Variable step: local step ∝ 1 − α cos(2πu), finest at the ends, 4× coarser in the middle."""
    u = np.linspace(0.0, 1.0, n)
    return lo + (hi - lo) * (u - alpha / (2 * np.pi) * np.sin(2 * np.pi * u))


def bracket(ax, x):
    """Indices of the two nodes enclosing coordinate x (clipped at the ends)."""
    i = int(np.clip(np.searchsorted(ax, x) - 1, 0, len(ax) - 2))
    return i, i + 1


def corners(axes, coords):
    return list(itertools.product(*[bracket(ax, x) for ax, x in zip(axes, coords)]))


def pair_levels(n_items, edges):
    """Kruskal: edges ``(level, i, j)`` → ``{(i, j): level at which i and j first connect}``."""
    parent = list(range(n_items))
    members = {i: [i] for i in range(n_items)}
    out = {}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for level, i, j in sorted(edges):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        for a in members[ri]:
            for b in members[rj]:
                out[(min(a, b), max(a, b))] = level
        parent[rj] = ri
        members[ri] += members.pop(rj)
    return out


def _mb(scale):
    return {
        "uniform": [uniform(-1.5, 1.2, 108 * scale + 1), uniform(-0.3, 2.0, 92 * scale + 1)],
        "anisotropic": [uniform(-1.5, 1.2, 216 * scale + 1), uniform(-0.3, 2.0, 46 * scale + 1)],
        "variable": [stretched(-1.5, 1.2, 108 * scale + 1), stretched(-0.3, 2.0, 92 * scale + 1)],
    }


def _sw3(scale):
    return {
        "uniform": [uniform(-1.6, 1.6, 32 * scale + 1)] * 3,
        "anisotropic": [uniform(-1.6, 1.6, n * scale + 1) for n in (64, 32, 16)],
        "variable": [stretched(-1.6, 1.6, 32 * scale + 1)] * 3,
    }


def _hb(scale):
    return {
        "uniform": [uniform(-1.5, 1.5, 60 * scale + 1), uniform(-0.6, 0.6, 24 * scale + 1), uniform(-1.5, 1.5, 60 * scale + 1)],
        "anisotropic": [uniform(-1.5, 1.5, 120 * scale + 1), uniform(-0.6, 0.6, 6 * scale + 1), uniform(-1.5, 1.5, 30 * scale + 1)],
        "variable": [stretched(-1.5, 1.5, 60 * scale + 1), stretched(-0.6, 0.6, 24 * scale + 1), stretched(-1.5, 1.5, 60 * scale + 1)],
    }


def _sw5():
    return {
        "uniform": [uniform(-1.6, 1.6, 17)] * 5,
        "anisotropic": [uniform(-1.6, 1.6, n) for n in (33, 17, 17, 9, 9)],
        "variable": [stretched(-1.6, 1.6, 17)] * 5,
    }


# (surface, grid kind, axes). separable_wells(5) runs at one resolution only: its doubled grid has 39M cells.
GRIDS = (
    [(muller_brown(), f"{kind} x{s}", ax) for s in (1, 2) for kind, ax in _mb(s).items()]
    + [(separable_wells(3), f"{kind} x{s}", ax) for s in (1, 2) for kind, ax in _sw3(s).items()]
    + [(hidden_barrier(), f"{kind} x{s}", ax) for s in (1, 2) for kind, ax in _hb(s).items()]
    + [(separable_wells(5), kind, ax) for kind, ax in _sw5().items()]
)
IDS = [f"{surf.name}-{kind.replace(' ', '-')}" for surf, kind, _ in GRIDS]


def within_one_cell(axes, coords, index):
    """True if ``index`` lies in the cell enclosing ``coords`` or one node beyond it on every axis."""
    return all(lo - 1 <= i <= hi + 1 for (lo, hi), i in zip((bracket(ax, x) for ax, x in zip(axes, coords)), index))


def matched_basins(surf, axes, E, ws, moore_minima):
    """Basin id of the lowest corner of the cell enclosing each analytic minimum (assertion 1)."""
    basin_of = []
    for coords, energy in surf.minima:
        cs = corners(axes, coords)
        best = min(cs, key=lambda c: E[c])
        b = int(ws.labels[best])
        seed, seed_energy = ws.basins[b]
        assert energy - 1e-9 <= seed_energy <= E[best] + 1e-12, (surf.name, coords)
        assert within_one_cell(axes, coords, seed), (surf.name, coords, seed)
        assert any(within_one_cell(axes, coords, m) for m in moore_minima), (surf.name, coords)
        basin_of.append(b)
    assert len(set(basin_of)) == len(surf.minima)
    return basin_of


def local_tolerance(E, axes, coords, energy):
    return 1.5 * max(abs(E[c] - energy) for c in corners(axes, coords))


@pytest.mark.parametrize("surf, kind, axes", GRIDS, ids=IDS)
def test_minima_saddles_and_levels_on_every_grid_kind(surf, kind, axes):
    E = surf.sample(axes)
    ws = find_watershed_segmentation(E)
    moore_minima = [i for i, _ in find_minima_grid(E)]
    basin_of = matched_basins(surf, axes, E, ws, moore_minima)

    # 2. every other basin is a grid artefact: its persistence is below the largest local
    #    tolerance of any critical point; and only the matched basins survive half the
    #    smallest analytic barrier
    persistence = compute_persistence(ws.basins, ws.merges)
    tol_max = max(
        [local_tolerance(E, axes, c, e) for c, e in surf.minima]
        + [local_tolerance(E, axes, c, e) for c, e, _ in surf.saddles]
    )
    matched = set(basin_of)
    extra = [float(persistence[b]) for b in range(len(ws.basins)) if b not in matched]
    assert all(p <= tol_max for p in extra), (surf.name, kind, sorted(extra, reverse=True)[:3], tol_max)
    p_min = 0.5 * min(e - max(surf.minima[i][1], surf.minima[j][1]) for _, e, (i, j) in surf.saddles)
    assert int((persistence >= p_min).sum()) == len(surf.minima)

    # 3. minimax levels between the matched basins, from the watershed's own merge list,
    #    against Kruskal over the analytic saddles with their `joins` as edges
    want = pair_levels(len(surf.minima), [(e, i, j) for _, e, (i, j) in surf.saddles])
    saddle_at = {e: c for c, e, _ in surf.saddles}
    pos = {b: k for k, b in enumerate(basin_of)}
    parent = list(range(len(ws.basins)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    held = {b: [pos[b]] for b in basin_of}
    got = {}
    for _, level, deeper, shallower in ws.merges:
        rd, rs = find(deeper), find(shallower)
        for a in held.get(rd, []):
            for b in held.get(rs, []):
                got[(min(a, b), max(a, b))] = level
        parent[rs] = rd
        if rs in held:
            held.setdefault(rd, []).extend(held.pop(rs))
    for pair, e_star in want.items():
        tol = 1.5 * max(abs(E[c] - e_star) for c in corners(axes, saddle_at[e_star]))
        assert abs(got[pair] - e_star) <= tol, (surf.name, kind, pair, got[pair], e_star, tol)


from pes_analyzer.topology import find_steepest_descent_path


def local_max_step(axes, coords):
    """Largest step adjacent to the cell enclosing ``coords``, over all axes."""
    out = 0.0
    for ax, x in zip(axes, coords):
        i = bracket(ax, x)[0]
        out = max(out, float(np.diff(ax)[max(i - 1, 0) : i + 2].max()))
    return out


@pytest.mark.parametrize("surf, kind, axes", GRIDS, ids=IDS)
def test_steepest_descent_from_each_saddle_reaches_a_joined_minimum(surf, kind, axes):
    E = surf.sample(axes)
    for coords, _, (i, j) in surf.saddles:
        start = tuple(int(np.argmin(np.abs(ax - x))) for ax, x in zip(axes, coords))
        path, prof = find_steepest_descent_path(E, start, axes=axes)
        assert (np.diff(prof) < 0).all()
        end = np.array([ax[k] for ax, k in zip(axes, path[-1])])
        dist = min(
            np.linalg.norm(end - np.asarray(surf.minima[k][0])) / local_max_step(axes, surf.minima[k][0])
            for k in (i, j)
        )
        assert dist <= 2.0, (surf.name, kind, coords, dist)
