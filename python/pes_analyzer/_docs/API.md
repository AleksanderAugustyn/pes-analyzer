# API Reference

This is the canonical contract for the three public functions of `pes_analyzer`. Examples are runnable Python; copy/paste a block to verify behaviour.

## Conventions

- Energy grids are C-contiguous `numpy.ndarray`. Both `float64` and `float32` are accepted at the kernel boundary (extraction tries `float32` then `float64`); an `f64` array is processed as `f64`. Pass `np.ascontiguousarray(arr)` if you have a non-contiguous view.
- N-D indices are `tuple[int, ...]` in `numpy` axis order.
- `NaN` cells are treated as masked. They are impassable for saddle search and excluded from minimum search.
- Supported dimensionality: N ∈ [2, 7]. The Rust kernels enforce this at the boundary.
- Threading: the `find_minima_grid` / `find_maxima_grid` / `find_extrema_grid` scans and the flood sort run on the rayon global pool (all cores by default; set `RAYON_NUM_THREADS` to limit). Results never depend on the thread count. Run one analysis per process at a time — concurrent floods in one process each retain their grid arrays and share the pool. The three path kernels added in 0.11.0 (steepest descent, least action, minimum ascent) are sequential.

---

## `build_dense`

```python
from pes_analyzer.grid import build_dense

def build_dense(
    coords: dict[str, numpy.ndarray],
    values: numpy.ndarray,
    *,
    dtype: numpy.typing.DTypeLike | None = None,
) -> tuple[numpy.ndarray, dict[str, numpy.ndarray]]:
    ...
```

### Parameters

- **`coords`** — Ordered mapping `{axis_name: coord_per_row_1d_array}`. The insertion order of the dict determines the axis order of the output ndarray. Each coordinate array must have length `len(values)`.
- **`values`** — 1-D array of scalars (energies or any per-row value).
- **`dtype`** *(keyword-only, default `None`)* — output dtype. When `None`, the dtype of `values` is preserved if it is floating, else `np.float64`; an explicit `dtype` forces the output to that type. Pass `np.float32` to halve the memory footprint of large grids — the kernels accept the resulting `f32` grid directly.

### Returns

- **`dense`** — C-contiguous array of shape `(n_unique_axis_0, ..., n_unique_axis_{N-1})`, dtype per the `dtype` rule above (`float64` by default for `float64`/integer input). Missing cells are `np.nan`.
- **`axes`** — `{axis_name: sorted_unique_values_1d_array}`, same key order as `coords`.

### Raises

- `ValueError` if any coord array length disagrees with `len(values)` or if `coords` is empty.

### What it does

Pivots a long-form table of `(coord_0, ..., coord_{N-1}, value)` rows into the dense N-D grid that the other `pes_analyzer` functions consume. Axes are inferred from the unique values per coordinate column, sorted ascending.

### Example

```python
import numpy as np
from pes_analyzer.grid import build_dense

coords = {
    "x": np.array([0.0, 0.0, 1.0, 1.0]),
    "y": np.array([0.0, 1.0, 0.0, 1.0]),
}
values = np.array([10.0, 11.0, 20.0, 21.0])

dense, axes = build_dense(coords, values)
print(dense)
# [[10. 11.]
#  [20. 21.]]
print(axes)
# {'x': array([0., 1.]), 'y': array([0., 1.])}
```

### Notes and edge cases

- **Axis order** follows `coords` insertion order. Swap the dict to swap the axes.
- **Duplicate `(coords, ...)` rows**: last-write-wins.
- **Single-value axes are NOT squeezed.** If one axis has only one unique value, the output retains that length-1 axis. The caller is responsible for filtering active axes before calling.

---

## `minimize_grid`

```python
from pes_analyzer.grid import minimize_grid

def minimize_grid(
    energies: numpy.ndarray,
    keep: int | Sequence[int],
    *,
    threads: int | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray[intp]]:
    ...
```

### Parameters

- **`energies`** — floating array with N ≥ 2 axes, any memory layout. `NaN` cells are ignored.
- **`keep`** — one axis index or 1 to N−1 distinct axis indices. The output axes follow this order.
- **`threads`** *(keyword-only)* — workers for the slabs along the first kept axis (default `os.cpu_count()`). The result does not depend on it.

### Returns

- **`minimum`** — shape `tuple(energies.shape[k] for k in keep)`, dtype of `energies`; `NaN` where the column has no valid cell.
- **`index`** — shape `minimum.shape + (N,)`, dtype `intp`: the full N-D index of the minimising cell in the original axis order, `-1` in every entry where there is none. Ties go to the first cell in C order over the hidden axes.

### Example

```python
import numpy as np
from pes_analyzer.grid import jump_map, minimize_grid

energies = np.array([
    [[3.0, 1.0, 2.0], [0.5, 4.0, 4.0]],
    [[2.0, 2.0, 0.0], [np.nan, np.nan, np.nan]],
])                                             # keep axes 0 and 1, minimise over axis 2
minimum, index = minimize_grid(energies, keep=(0, 1))
print(minimum)
# [[1.  0.5]
#  [0.  nan]]
print(index[0, 0], index[1, 1])
# [0 0 1] [-1 -1 -1]
print(jump_map(index, keep=(0, 1)))
# [[ 1.  1.]
#  [ 1. nan]]

valid = index[..., 0] >= 0                     # mask the -1 rows: they would wrap, not fail
print(energies[tuple(np.moveaxis(index[valid], -1, 0))])
# [1.  0.5 0. ]
```

The last lines gather any grid of the same shape at the minimiser. Always mask with `index[..., 0] >= 0` first.

---

## `jump_map`

```python
from pes_analyzer.grid import jump_map

def jump_map(index: numpy.ndarray, keep: int | Sequence[int]) -> numpy.ndarray[float64]:
    ...
```

For each map point with a minimiser: the largest Chebyshev distance, over the hidden axes, between its minimiser and that of any von Neumann map neighbour with one; `0.0` if no neighbour has one; `NaN` where the point itself has none. In cells. A value of at most 1 means the map is continuous there to within one hidden step; larger values mark where the minimiser jumps between valleys that the map cannot show. `index` is the second output of `minimize_grid` with the same `keep`.

---

## Axis coordinates: `index_to_coords`, `path_length`

Every function that needs physical distances takes `axes=`: `None` (index coordinates, unit steps), a mapping `{name: 1-D array}` with N entries in axis order (what `build_dense` returns; names are ignored), or a sequence of N 1-D arrays. Each array must be finite, strictly increasing and as long as its axis; steps may differ between axes and along an axis. The distance between cells is Euclidean in the supplied coordinates; scaling axes of different units to a common one is the caller's job. Functions that do **not** take `axes` (`find_watershed_segmentation`, `find_minimax_path`, `find_iwf_grid`, the extrema functions, `minimize_grid`) give the same output for any coordinates: see `ALGORITHMS.md` § Grids with unequal steps.

```python
from pes_analyzer.grid import index_to_coords, path_length

def index_to_coords(indices: numpy.ndarray, axes) -> numpy.ndarray[float64]: ...   # (..., N) -> (..., N)
def path_length(indices: numpy.ndarray, axes=None) -> numpy.ndarray[float64]: ...  # (K, N) -> (K,) cumulative
```

`index_to_coords` raises `IndexError` for an index outside an axis, including the `-1` rows of `minimize_grid`. `path_length` starts at 0 and does not require consecutive rows to be neighbours.

```python
import numpy as np
from pes_analyzer.grid import index_to_coords, path_length

axes = {"x": np.array([0.0, 1.0, 3.0]), "y": np.array([10.0, 20.0, 40.0, 80.0])}
path = np.array([[0, 0], [1, 1], [2, 3]])
print(index_to_coords(path, axes))
# [[ 0. 10.]
#  [ 1. 20.]
#  [ 3. 80.]]
print(path_length(path, axes))
# [ 0.         10.04987562 70.0831997 ]
print(path_length(path))
# [0.         1.41421356 3.65028154]
```

---

## `find_iwf_grid`

```python
from pes_analyzer.saddle import find_iwf_grid

def find_iwf_grid(
    energies: numpy.ndarray[float64],
    start: tuple[int, ...],
    end: tuple[int, ...],
    neighborhood: str = "von_neumann",
) -> tuple[tuple[int, ...], float] | None:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float64` array of ndim N ∈ [2, 7]. `NaN` cells are treated as walls.
- **`start`** — N-tuple of `int` grid indices. Must reference a non-`NaN` cell.
- **`end`** — N-tuple of `int` grid indices. Must reference a non-`NaN` cell.
- **`neighborhood`** — `"von_neumann"` (2N axis neighbors, default) or `"moore"` (3ᴺ−1 Chebyshev r=1 neighbors). See `ALGORITHMS.md` § Neighborhood stencils.

### Returns

- `tuple[tuple[int, ...], float]` — the saddle cell `(index, energy)`.
- `None` — if `start` and `end` lie in disjoint non-`NaN` regions (no path exists).

### Raises

- `ValueError` if `energies` is not C-contiguous, if `start`/`end` have the wrong length, if either endpoint cell is `NaN`, or if `neighborhood` is not `"von_neumann"`/`"moore"`.
- `IndexError` if any `start`/`end` index is out of bounds or negative.

### What it does

Returns the saddle point between `start` and `end` using the imaginary water flow (watershed) algorithm: the lowest-energy cell on any axis-connected non-`NaN` path between them. See [`ALGORITHMS.md`](./ALGORITHMS.md#find_iwf_grid) for the underlying algorithm.

### Example

```python
import numpy as np
from pes_analyzer.saddle import find_iwf_grid

energies = np.full((3, 5), 10.0)
energies[1, 0] = 0.0   # start basin
energies[1, 4] = 0.0   # end basin
energies[1, 1:4] = [1.0, 2.0, 1.0]  # bridge with saddle at (1, 2)

print(find_iwf_grid(energies, start=(1, 0), end=(1, 4)))
# ((1, 2), 2.0)
```

### Notes and edge cases

- **`start == end`**: returns `(start, energies[start])` immediately.
- **`NaN` walls**: cells with `NaN` energy are excluded from the search. If `start` and `end` are not connected through the non-`NaN` region, the function returns `None`.
- **Neighbourhood**: axis-only (2N stencil). Diagonal moves are not allowed — see [`ALGORITHMS.md`](./ALGORITHMS.md#find_iwf_grid) for the rationale.

---

## `find_minima_grid`

```python
from pes_analyzer.extrema import find_minima_grid

def find_minima_grid(
    energies: numpy.ndarray[float64],
    *,
    neighborhood_range: int = 1,
    confirm_range: int | None = None,
) -> list[tuple[tuple[int, ...], float]]:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float64` array of ndim N ∈ [2, 7]. `NaN` cells are skipped.
- **`neighborhood_range`** *(keyword-only, default `1`)* — Chebyshev half-width `r` of the neighbor stencil. A cell is compared against every in-bounds neighbor with `max_axis |Δi| ≤ r` (the `(2r+1)ᴺ − 1` stencil). Must satisfy `1 ≤ r ≤ 5`. The default `r = 1` is the classic 3ᴺ−1 king-move stencil.
- **`confirm_range`** *(keyword-only, default `None`)* — optional second-pass check. When `None`, the function returns the direct `neighborhood_range`-stencil minima. When set to an integer `R`, the function first finds candidates at `neighborhood_range` and then re-checks each candidate against the `R`-wide stencil. Must satisfy `1 ≤ R ≤ 5` and `R ≥ neighborhood_range`. The result for `neighborhood_range=1, confirm_range=R` is identical to `neighborhood_range=R` but typically much cheaper, since most cells are culled by the `r=1` pass.

### Returns

- `list[tuple[tuple[int, ...], float]]` — every cell that qualifies as a minimum as `(index, energy)`. Sorted ascending by energy; ties broken by `f64::total_cmp` for determinism.

### Raises

- `ValueError` if `energies` is not C-contiguous, if `energies.ndim` is outside `[2, 7]`, if `neighborhood_range` is outside `[1, 5]`, if `confirm_range` is outside `[1, 5]`, or if `confirm_range < neighborhood_range`.
- `TypeError` if `neighborhood_range` or `confirm_range` is passed positionally or is not an `int`/`None`.
- `OverflowError` if `neighborhood_range` or `confirm_range` is negative.

### What it does

Returns every cell whose energy is **not strictly greater than any non-`NaN` neighbour** in the Chebyshev box of half-width `neighborhood_range`. Equivalently: no neighbour with `max_axis |Δi| ≤ neighborhood_range` has strictly lower energy. Ties are allowed; a cell with one or more equal-energy neighbours still qualifies as long as none is lower. With the default `neighborhood_range = 1` this is the full 3ᴺ−1 (king-move) stencil. See [`ALGORITHMS.md`](./ALGORITHMS.md#find_minima_grid) for the underlying algorithm.

### Example

```python
import numpy as np
from pes_analyzer.extrema import find_minima_grid

energies = np.array([
    [5.0, 5.0, 5.0],
    [5.0, 0.0, 5.0],
    [5.0, 5.0, 5.0],
])

print(find_minima_grid(energies))
# [((1, 1), 0.0)]
```

A larger `neighborhood_range` disqualifies cells beaten by farther neighbours. The example uses `NaN` walls so only the intentional dips qualify:

```python
import numpy as np
from pes_analyzer.extrema import find_minima_grid

energies = np.array([
    [np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan,    5.0,    5.0,    5.0, np.nan],
    [np.nan,    5.0,    3.0,    5.0, np.nan],
    [np.nan,    5.0,    5.0,    5.0, np.nan],
    [   1.0, np.nan, np.nan, np.nan, np.nan],
], dtype=np.float64)

print(find_minima_grid(energies, neighborhood_range=1))
# [((4, 0), 1.0), ((2, 2), 3.0)]

print(find_minima_grid(energies, neighborhood_range=2))
# [((4, 0), 1.0)]
```

At `r = 1`, `(2, 2)` is a minimum because all eight king-move neighbours are `5.0`. At `r = 2`, the 5×5 stencil around `(2, 2)` reaches `(4, 0) = 1.0` — strictly lower — and `(2, 2)` is no longer reported.

The recommended way to compute the same result more cheaply is to find candidates at `r = 1` and confirm against the wider stencil:

```python
# Same result as neighborhood_range=2, but cheaper: only candidates that
# pass the r=1 check are re-tested against the wider stencil.
print(find_minima_grid(energies, neighborhood_range=1, confirm_range=2))
# [((4, 0), 1.0)]
```

### Notes and edge cases

- **Plateaus are reported.** Every cell on a flat plateau that has no king-move neighbour with strictly lower energy qualifies. In a uniformly-flat region, every cell whose stencil contains no lower neighbour will appear in the output — including corner and edge cells of the plateau where the stencil happens to be entirely within the plateau. If you only want strictly-isolated minima, filter the output yourself.
- **Boundary cells**: the stencil is clipped at array edges. A corner cell has fewer neighbours but is tested with the same rule.
- **`NaN` neighbours** are ignored — a cell is tested only against its non-`NaN` neighbours. A cell with at least one non-`NaN` neighbour and no strictly-lower one is reported; a cell whose entire stencil is `NaN` is not.

---

## `find_maxima_grid`

```python
from pes_analyzer.extrema import find_maxima_grid

def find_maxima_grid(
    energies: numpy.ndarray[float64],
    *,
    neighborhood_range: int = 1,
    confirm_range: int | None = None,
) -> list[tuple[tuple[int, ...], float]]:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float64` array of ndim N ∈ [2, 7]. `NaN` cells are skipped.
- **`neighborhood_range`** *(keyword-only, default `1`)* — Chebyshev half-width `r` of the neighbor stencil. Must satisfy `1 ≤ r ≤ 5`.
- **`confirm_range`** *(keyword-only, default `None`)* — optional second-pass check. See [`find_minima_grid`](#find_minima_grid) for the semantics; the rule applies identically here with the comparator flipped.

### Returns

- `list[tuple[tuple[int, ...], float]]` — every cell that qualifies as a maximum as `(index, energy)`. Sorted descending by energy; ties broken by `f64::total_cmp` (reversed) for determinism.

### Raises

Same as `find_minima_grid`: `ValueError`, `TypeError`, `OverflowError` per the validation table.

### What it does

Strict dual of `find_minima_grid`: returns every cell whose energy is **not strictly less than any non-`NaN` neighbour** in the Chebyshev box of half-width `neighborhood_range`. Ties are allowed.

### Example

```python
import numpy as np
from pes_analyzer.extrema import find_maxima_grid

energies = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
])

print(find_maxima_grid(energies))
# [((1, 1), 1.0)]
```

---

## `find_extrema_grid`

```python
from pes_analyzer.extrema import find_extrema_grid

def find_extrema_grid(
    energies: numpy.ndarray[float64],
    *,
    neighborhood_range: int = 1,
    confirm_range: int | None = None,
) -> tuple[
    list[tuple[tuple[int, ...], float]],  # minima, ascending
    list[tuple[tuple[int, ...], float]],  # maxima, descending
]:
    ...
```

### Parameters

Same as `find_minima_grid` / `find_maxima_grid`. The two polarities share a single `neighborhood_range` and a single `confirm_range`.

### Returns

A 2-tuple `(minima_list, maxima_list)`. Each list has the same shape as the corresponding single-polarity function would produce. Minima are sorted ascending by energy, maxima descending.

The combined result is byte-identical to `(find_minima_grid(arr, **k), find_maxima_grid(arr, **k))`. The optimisation is purely a constant-factor saving — one stencil walk per cell in the find stage instead of two.

### Raises

Same as `find_minima_grid`.

### Example

```python
import numpy as np
from pes_analyzer.extrema import find_extrema_grid

energies = np.array([
    [1.0, 2.0, 1.0],
    [2.0, 0.0, 2.0],
    [1.0, 2.0, 1.0],
])

mins, maxs = find_extrema_grid(energies)
print(mins)  # [((1, 1), 0.0)]
print(maxs)  # plateau of 2.0s around the rim
```

---

## `find_watershed_segmentation`

```python
from pes_analyzer.topology import Watershed, find_watershed_segmentation

def find_watershed_segmentation(
    energies: numpy.ndarray[float32 | float64],
    neighborhood: str = "von_neumann",
    *,
    parents: bool = False,
) -> Watershed:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float32` or `float64` array of ndim N ∈ [2, 7]. `NaN` cells are masked.
- **`neighborhood`** — `"von_neumann"` (2N axis neighbors, default) or `"moore"` (3ᴺ−1 Chebyshev r=1 neighbors). See `ALGORITHMS.md` § Neighborhood stencils.
- **`parents`** — also record each cell's flood parent as a direction code (`Watershed.parents`, 2N bytes). Required by `find_minimax_path(..., tree=)`.

### Returns

A `Watershed` dataclass — the single owner of the grid-sized arrays:

| Field | Type | Size | Meaning |
|---|---|---|---|
| `labels` | `int32` ndarray, shape == `energies.shape`, read-only | 4N B | `-1` iff the cell is `NaN`, otherwise the basin ID the cell first joined during the flood |
| `basins` | `list[tuple[tuple[int, ...], float]]` | — | `(min_nd_index, min_energy)` per basin, ascending by energy; `basins[0]` holds the global minimum |
| `merges` | `list[tuple[tuple[int, ...], float, int, int]]` | — | `(saddle_nd_index, saddle_energy, deeper_id, shallower_id)` per union of two distinct basins, ascending by saddle energy; `basins[deeper].min_e <= basins[shallower].min_e` always holds |
| `neighborhood` | `str` | — | the stencil the flood used |
| `parents` | `uint16` ndarray or `None`, read-only | 2N B | flood-parent direction code per cell (`65535` at seeds and `NaN` cells); `None` unless `parents=True` |
| `merge_table` | `uint32` ndarray `(M, 5)`, read-only | 20M B | per merge `saddle_lin, other_lin, deeper, shallower, saddle_side` — linear cell indices of the saddle and of its already-flooded neighbour on the other component, the basin ids, and the basin the saddle cell itself belongs to |
| `dtype` | `numpy.dtype` | — | dtype of `energies` |
| `fingerprint` | `bytes` (16) | — | `energy_fingerprint(energies)`, used by `find_minimax_path(tree=)` to reject a different grid |

Plus:

- **`has_labels`** — `True` while the grid arrays are held.
- **`drop_labels()`** — sets `labels`, `parents` and `merge_table` to `None`, freeing 4N + 2N bytes for every holder at once (a `MergeTree` built on the object sees the same `None`). `basins` and `merges` survive, so tree queries keep working.

`energy_fingerprint(energies) -> bytes` is exported as well: a 16-byte blake2b over the shape, the dtype and every k-th cell with k = max(1, size // 2²⁰). It distinguishes grids, not bit-exact copies.

### Raises

- `ValueError` if `energies` is not C-contiguous, if `energies.ndim` is outside `[2, 7]`, or if `neighborhood` is not `"von_neumann"`/`"moore"`.

### What it does

Runs the imaginary-water-flow flood to completion (not just until two specified endpoints connect). Records every union that merges two previously-disconnected basins as a merge event. This is the full segmentation that `find_iwf_grid` partially computes — `find_iwf_grid` is the two-point specialization that stops at the first basin-merge between its two endpoint cells.

**Tie contract.** Cells are flooded in ascending energy; equal energies in ascending C-order linear index. Basin ids are assigned in that order, so at every merge the lower id is `deeper` (ties resolve to the lower seed index) and basin 0 is always the root of the merge tree.

### Example

```python
import numpy as np
from pes_analyzer.topology import find_watershed_segmentation

# Two corner minima joined by a ridge row whose saddle is (2, 2) = 4.
energies = np.full((5, 5), 10.0)
energies[0, 0], energies[0, 1], energies[1, 0], energies[1, 1] = 0.0, 1.0, 1.0, 2.0
energies[2, :] = [3.0, 3.0, 4.0, 3.0, 3.0]
energies[4, 4], energies[3, 4], energies[4, 3], energies[3, 3] = 0.0, 1.0, 1.0, 2.0

ws = find_watershed_segmentation(energies)
print(ws.basins)
# [((0, 0), 0.0), ((4, 4), 0.0)]       # tie: lower linear index first
print(ws.merges)
# [((2, 2), 4.0, 0, 1)]
print(ws.labels[2, 2], ws.merge_table)
# 0 [[12 13  0  1  0]]                  # saddle cell 12 adopted basin 0 first, met basin 1 via cell 13
```

### Notes

- **Memory.** Peak inside the call is 8N + 4V bytes (10N + 4V with `parents=True`; N = cells, V = non-`NaN` cells) plus a sort transient of 12V for `float32` / 20V for `float64` (the `(f64, u32)` pair is padded to 16 bytes) that is released before the flood loop starts. The returned arrays (4N labels, 2N parents) stay resident until `drop_labels()`.
- **Threading.** The sort uses the rayon pool (`RAYON_NUM_THREADS`); results do not depend on the thread count.

---

## Path kinds

| Function | The path it returns | Needs `end` | Uses step lengths | Default stencil |
|---|---|---|---|---|
| `find_minimax_path` | minimises the highest value crossed, and dips to each basin minimum on the way | yes | no | `"von_neumann"` |
| `find_steepest_descent_path` | follows the largest downward slope from `start` until no neighbour is lower | no | yes | `"moore"` |
| `find_least_action_path` | minimises ∫ cost ds | yes, cell or mask | yes | `"moore"` |
| `find_minimum_ascent_path` | minimises the total climb Σ max(ΔE, 0) | yes, cell or mask | only to break ties | `"von_neumann"` |

A kernel whose main result depends on step lengths defaults to Moore, because von Neumann measures length in the Manhattan metric; the others default to von Neumann like the flood kernels. All four return a `(K, N)` `int64` index matrix whose row 0 is `start`, plus a `(K,)` `float64` profile. The three new kernels accept `float32` or `float64` and do their arithmetic in `float64`.

---

## `find_minimax_path`

```python
from pes_analyzer.topology import find_minimax_path

def find_minimax_path(
    energies: numpy.ndarray[float32 | float64],
    start: tuple[int, ...],
    end: tuple[int, ...],
    neighborhood: str | None = None,
    *,
    tree: MergeTree | Watershed | None = None,
) -> tuple[numpy.ndarray[int64], numpy.ndarray[float64]] | None:
    ...
```

### Parameters

- **`energies`** — C-contiguous `float32` or `float64` array of ndim N ∈ [2, 7]. `NaN` cells are treated as walls.
- **`start`** — N-tuple of `int` grid indices. Must reference a non-`NaN` cell.
- **`end`** — N-tuple of `int` grid indices. Must reference a non-`NaN` cell.
- **`neighborhood`** — `"von_neumann"` (2N axis neighbors) or `"moore"` (3ᴺ−1 Chebyshev r=1 neighbors). Standalone mode defaults to `"von_neumann"`. With `tree=`, the neighbourhood is the tree's; an explicit value must match it.
- **`tree`** — a `MergeTree` or `Watershed` built with `parents=True` from the same grid. The path is then reconstructed from the recorded flood state: no re-flood and no grid-sized allocation (O(path + basins + merges) words).

### Returns

- `(path_indices, path_energies)`:
  - **`path_indices`** — `(K, N)` `int64` array. Row 0 is `start`, row K−1 is `end`. With `"von_neumann"`, consecutive rows differ by ±1 in exactly one axis; with `"moore"`, consecutive rows are at Chebyshev distance 1.
  - **`path_energies`** — `(K,)` `float64` array, the energy profile along the path.
- `None` — if `start` and `end` lie in disjoint non-`NaN` regions (no path exists).
- `start == end` returns a length-1 path.

### Raises

- `ValueError` if `energies` is not C-contiguous, if `start`/`end` have the wrong length, if either endpoint cell is `NaN`, or if `neighborhood` is not `"von_neumann"`/`"moore"`.
- `IndexError` if any `start`/`end` index is out of bounds or negative (same as `find_iwf_grid`, in both modes).
- With `tree=`, additionally `ValueError` when:
  - the tree's grid arrays were released (`"labels were dropped"`);
  - it was built without `parents=True`;
  - `neighborhood` is given and differs from `tree.neighborhood`;
  - `tree.labels.shape != energies.shape`;
  - `energies` has a different dtype or fingerprint from the tree's grid (`"different energy grid"`);
  - the tree arrays have the wrong dtype or shape (`labels` must be `int32`, `parents` `uint16` with the shape of `labels`, `merge_table` `uint32` of shape `(M, 5)`);
  - the arrays are internally inconsistent (basin ids out of range, invalid or grid-leaving direction codes, cyclic parent chains, merges that contradict earlier ones).

### What it does

Computes the *deep minimax path*: among all grid paths from `start` to `end` it minimizes the highest energy crossed (so it passes through the exact saddles `find_iwf_grid` reports), and between saddles it descends to the actual basin minimum cells. The profile's local maxima are therefore true inter-basin saddles and its local minima are true basin minima — feed `path_energies` to `analyze_path_profile` to extract them.

The minimax path is what the grid-based literature calls the minimum energy path (MEP): it crosses the same saddles at the same energies. Between saddles it follows the flood's descent chains, not the gradient, so the cells visited can differ from a steepest-descent path; `find_steepest_descent_path` gives that route.

**Standalone mode** (`tree=None`) floods with the same kernel as `find_watershed_segmentation`, recording parents and stopping as soon as `start` and `end` connect, then reconstructs the path from that partial flood; it costs 10N + 4V bytes plus the sort transient. **Tree mode** skips the flood entirely: the Kruskal forest is rebuilt from `merge_table` and the descents follow `parents`. For the same grid and neighbourhood both modes return the identical path. See `ALGORITHMS.md` (`find_minimax_path`).

### Example

```python
import numpy as np
from pes_analyzer.topology import MergeTree, analyze_path_profile, find_minimax_path, find_watershed_segmentation

energies = np.array([[0.0, 3.0, 5.0, 4.0, 1.0, 3.0, 6.0, 4.0, 2.0]])
idx, prof = find_minimax_path(energies, (0, 0), (0, 8))
print(idx[:, 1])
# [0 1 2 3 4 5 6 7 8]
print(analyze_path_profile(prof))
# PathProfile(minima=[(0, 0.0), (4, 1.0), (8, 2.0)], saddles=[(2, 5.0), (6, 6.0)])

tree = MergeTree(find_watershed_segmentation(energies, parents=True))
idx2, prof2 = find_minimax_path(energies, (0, 0), (0, 8), tree=tree)   # same path, no re-flood
assert (idx2 == idx).all()
```

### Notes and edge cases

- The path can be long: it dips to every basin minimum between barriers (a single descent chain on a 5-D map can be hundreds of steps). K is still tiny next to the grid size.
- The path is a walk, not necessarily a simple path: connecting two cells of one basin descends both to the basin minimum, which may re-walk a shared chain suffix.
- For the same `neighborhood`, `max(path_energies)` equals the `find_iwf_grid` saddle energy between the same endpoints (the minimax value is unique; with tied energies the saddle *cell* may differ).
- Tree mode validates the whole `labels` array (O(N), a fraction of a second at 10⁸ cells) before walking; the walk itself is O(K).

---

## `find_steepest_descent_path`

```python
from pes_analyzer.topology import find_steepest_descent_path

def find_steepest_descent_path(
    energies: numpy.ndarray[float32 | float64],
    start: tuple[int, ...],
    *,
    axes=None,
    neighborhood: str = "moore",
) -> tuple[numpy.ndarray[int64], numpy.ndarray[float64]]:
    ...
```

### What it does

From the current cell, the candidates are the stencil neighbours with a strictly lower, non-`NaN` energy. With none the path ends. Otherwise it moves to the candidate with the largest slope (E_current − E_neighbour) / Δs; equal slopes go to the smaller linear index. Consequences:

- Energies decrease strictly: no cell repeats, the path has at most V cells, and the function always returns (K = 1 if `start` has no lower neighbour).
- The end cell has no strictly lower stencil neighbour. With `"moore"` it is in the output of `find_minima_grid(energies)`, except for a cell whose whole stencil is `NaN`.
- A flat cell with no strictly lower neighbour ends the path even if the plateau drops further on.
- With unequal steps the choice follows the physical slope, not the index slope.
- The end cell need not be the seed of `labels[start]`: labels record the basin a cell first joined in the flood (a merge-tree segmentation), the descent follows the slope.
- Descents from a saddle are two calls, one from each of the two cells `Watershed.merge_table` records for that merge (`saddle_lin`, `other_lin`). Nothing forces the two to end in different minima.

### Raises

`ValueError` for a non-contiguous or non-float array, N outside [2, 7], a `start` of the wrong length or on a `NaN` cell, invalid `axes` or `neighborhood`; `IndexError` for `start` out of range or negative. `TypeError` for a non-integer index entry (a float is never truncated to a cell).

### Example

```python
import numpy as np
from pes_analyzer.topology import find_steepest_descent_path

energies = np.full((3, 3), 20.0)
energies[1, 1], energies[1, 2], energies[2, 1] = 10.0, 8.0, 9.0
idx, prof = find_steepest_descent_path(energies, (1, 1), neighborhood="von_neumann")
print(idx.tolist(), prof.tolist())
# [[1, 1], [1, 2]] [10.0, 8.0]

axes = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 4.0, 8.0])]      # axis 1 steps are four times longer
idx, prof = find_steepest_descent_path(energies, (1, 1), axes=axes, neighborhood="von_neumann")
print(idx.tolist(), prof.tolist())
# [[1, 1], [2, 1]] [10.0, 9.0]
```

---

## `find_least_action_path`

```python
from pes_analyzer.topology import find_least_action_path

def find_least_action_path(
    cost: numpy.ndarray[float32 | float64],
    start: tuple[int, ...],
    end: tuple[int, ...] | numpy.ndarray[bool],
    *,
    axes=None,
    neighborhood: str = "moore",
) -> tuple[numpy.ndarray[int64], numpy.ndarray[float64]] | None:
    ...
```

### What it does

Returns the stencil path from `start` to `end` that minimises the action Σ over steps of ½ (cost_a + cost_b) · Δs, the trapezoid rule for ∫ cost ds. `end` is one index or a boolean mask of target cells of the grid shape; the search stops at the first target reached (the one with the least action, then the shortest path). Among paths of equal action the shorter wins. The second output is the cumulative action at each path cell, `0` first and the total last. `None` if no target is reachable. `start` in the target set gives a one-row path.

`cost` is non-negative; `NaN` is a wall. The caller builds it from any function of position:

| `cost` | Path |
|---|---|
| `1` | shortest path around the walls |
| `sqrt(2 B (V − E0))`, clipped at 0 | WKB tunnelling path |
| `exp(V / T)` | as T → 0 its highest energy tends to the minimax level; as T → ∞ the path tends to the shortest one |
| the gradient norm ‖∇V‖, e.g. from `np.gradient` with the axis coordinates | for two fixed cells, the minimiser of the geometric Freidlin–Wentzell action of overdamped gradient dynamics with isotropic noise, ∫ ‖∇V‖ ds + (E_end − E_start) |

Two limits: the low-temperature route need not be the route of `find_minimax_path`, which dips to every basin minimum by construction (on a flat grid the least-action path is the direct one whatever T is); and with a mask as `end`, E_end differs between targets, so the cost ‖∇V‖ then minimises ∫ ‖∇V‖ ds alone.

**Grid-metric note.** The minimum is over stencil paths, so a straight line in a general direction becomes a staircase. With equal steps on all axes the Moore staircase is longer than the straight line by at most the factor √(Σⱼ (√j − √(j−1))²) over j = 1…N: 8.2% in 2-D, 12.8% in 3-D, 18.3% in 5-D, 21.8% in 7-D. For von Neumann the factor is √N: 41% in 2-D, 124% in 5-D.

**Tie-break under rounding.** Sums are `float64`. The guarantee is the exact-arithmetic optimum up to rounding of the sums; the length tie-break is exact where partial sums are exactly equal (zero-cost regions), not for totals that become equal only through rounding.

### Raises

`ValueError` for a non-contiguous or non-float array, N outside [2, 7], a `start` or tuple `end` of the wrong length or on a `NaN` cell, a mask of another shape or dtype, non-contiguous or without a `True` cell, any infinite value, any negative cost, invalid `axes` or `neighborhood`; `IndexError` for `start` or a tuple `end` out of range or negative. `TypeError` for a non-integer index entry in `start` or `end` (a float is never truncated to a cell).

### Memory

8 + 8 + 2 bytes per cell (action, length, back-pointer) for the whole grid, plus the heap; about 16 GB at 8.7×10⁸ cells, besides the input array. Sequential.

### Example

```python
import numpy as np
from pes_analyzer.topology import find_least_action_path

cost = np.array([[1.0, 1.0, 1.0], [1.0, 100.0, 1.0], [2.0, 2.0, 2.0]])
idx, action = find_least_action_path(cost, (1, 0), (1, 2), neighborhood="von_neumann")
print(idx.tolist(), action.tolist())
# [[1, 0], [0, 0], [0, 1], [0, 2], [1, 2]] [0.0, 1.0, 2.0, 3.0, 4.0]

exit_mask = np.zeros((3, 3), dtype=bool)
exit_mask[2, :] = True                          # any cell of the last row
idx, action = find_least_action_path(cost, (1, 0), exit_mask, neighborhood="von_neumann")
print(idx.tolist(), action.tolist())
# [[1, 0], [2, 0]] [0.0, 1.5]
```

---

## `find_minimum_ascent_path`

```python
from pes_analyzer.topology import find_minimum_ascent_path

def find_minimum_ascent_path(
    energies: numpy.ndarray[float32 | float64],
    start: tuple[int, ...],
    end: tuple[int, ...] | numpy.ndarray[bool],
    *,
    axes=None,
    neighborhood: str = "von_neumann",
) -> tuple[numpy.ndarray[int64], numpy.ndarray[float64]] | None:
    ...
```

### What it does

Returns the stencil path from `start` to `end` that minimises the total ascent Σ over steps of max(E_next − E_current, 0): descents are free, climbs cost what they gain. Among paths of equal ascent the shortest wins; `axes` enter only there. `end`, the mask form, the second output (cumulative climb) and `None` behave as in `find_least_action_path`. Facts:

- Along any path, ascent forward minus ascent backward equals E_end − E_start, so for two cells a and b, ascent(a → b) − ascent(b → a) = E_b − E_a.
- The total is at least max(E_end − E_start, 0) and at least the minimax level minus E_start. It is zero exactly when a path exists that never climbs.
- For fixed endpoints Σ |ΔE| = 2 · ascent + E_start − E_end, so the path also minimises the total variation of the energy.
- It is **not** the Freidlin–Wentzell action. For overdamped gradient dynamics with isotropic noise the geometric action of a path is ∫ ‖∇V‖ ds + (E_end − E_start), at least twice the total ascent with equality only along gradient lines: moving across the slope at constant energy costs action but no ascent. Twice the minimum ascent is a lower bound on that action; the action itself is minimised by `find_least_action_path` with cost ‖∇V‖.

Von Neumann is the default because the weight depends on energies only, as in the flood kernels, and a diagonal step would skip the cells between.

### Raises

As `find_least_action_path`, except that negative energies are allowed.

### Example

```python
import numpy as np
from pes_analyzer.topology import find_minimum_ascent_path

energies = np.array([[0.0, 5.0, 0.0], [0.0, 1.0, 0.0]])
idx, climb = find_minimum_ascent_path(energies, (0, 0), (0, 2))
print(idx.tolist(), climb.tolist())
# [[0, 0], [1, 0], [1, 1], [1, 2], [0, 2]] [0.0, 0.0, 1.0, 1.0, 1.0]
```

The direct route climbs 5; the detour through the second row climbs 1.

---

## `synthetic`

```python
from pes_analyzer.synthetic import AnalyticSurface, hidden_barrier, muller_brown, separable_wells
```

Analytic surfaces with exact critical points, for tests and for the data-free example. Each constructor returns an `AnalyticSurface` with `name`, `ndim`, `minima` (`(coords, energy)` ascending by energy), `saddles` (`(coords, energy, (i, j))` ascending by energy, where `i < j` are the positions in `minima` of the two minima the saddle joins along its unstable direction), `__call__(*coords)` on broadcastable arrays, and `sample(axes)` returning a dense C-contiguous `float64` grid on `axes` (mapping or sequence of 1-D arrays, as everywhere).

| Constructor | Surface | Minima | Saddles |
|---|---|---|---|
| `muller_brown()` | the 2-D Müller–Brown surface, 1979 parameters, points Newton-refined to ten decimals | 3 | 2 |
| `separable_wells(ndim, tilts=None)` | Σₐ (xₐ² − 1)² + τₐ s(xₐ), s(u) = (3u − u³)/2, default τₐ = 0.02 · 2ᵃ, each in (0, 8/3) | 2ᴺ at the corners ±1, energy Σ ±τₐ | N · 2ᴺ⁻¹, one axis at 3τₐ/8, each joining the two corners that differ in that axis |
| `hidden_barrier(b=1.0, h=5.0, t=0.5, w=1.0)` | b (x² − 1)² + w y² + h (z² − 1)² − t s(x) s(z), with 0 < t < min(8·min(b, h)/3, 16√(bh)/9) | 4 | 4 |

`hidden_barrier` is the surface whose barrier a 2-D map hides: minimised over z, the map shows a barrier of b + t (1.5 with the defaults) between the two deep minima, while every route between them has to change z and the real barrier is max(E_x, E_z) + t (5.514 with the defaults). The minimiser's z flips between −1 and +1 across x = 0, which `jump_map` shows as a line of jumps.

```python
import numpy as np
from pes_analyzer.grid import minimize_grid
from pes_analyzer.synthetic import hidden_barrier

surf = hidden_barrier()
print(surf.minima[0], surf.saddles[-1][0], round(surf.saddles[-1][1], 6), surf.saddles[-1][2])
# ((-1.0, 0.0, -1.0), -0.5) (1.0, 0.0, -0.0375) 5.014059 (1, 3)

E = surf.sample({"x": np.linspace(-1.5, 1.5, 61), "y": np.linspace(-0.5, 0.5, 21), "z": np.linspace(-1.5, 1.5, 61)})
minimum, index = minimize_grid(E, keep=(0, 1))
print(round(float(minimum[30, 10] - minimum[10, 10]), 6))     # the map's barrier between the deep minima
# 1.5
```

---

## Topology helpers

The pure-Python helpers in `pes_analyzer.topology` analyse the merge tree produced by `find_watershed_segmentation` and the profile produced by `find_minimax_path`.

### `analyze_path_profile(path_energies, min_persistence=0.0) -> PathProfile`

Extracts the alternating local minima and maxima (saddles) of a 1-D energy profile, then repeatedly cancels the adjacent minimum/saddle pair with the smallest energy gap until every surviving pair clears `min_persistence`. The profile's global minimum is never cancelled. Path endpoints participate like any extremum (a rising start / falling end counts as a minimum); plateau extrema report the last plateau cell. Raises `ValueError` on empty, non-1-D, or NaN-containing input.

`PathProfile` is a frozen dataclass with `minima: list[tuple[int, float]]` and `saddles: list[tuple[int, float]]` — `(path_index, energy)` pairs in path order. Map a `path_index` back to grid coordinates via row `k` of the `path_indices` array from `find_minimax_path`.

### `compute_persistence(basins, merges) -> numpy.ndarray[float64]`

Per-basin topological persistence. The deepest basin (`basins[0]`) has persistence `+inf`. Every other basin has persistence `saddle_energy − basin_minimum_energy`, computed from the merge event in which the basin was absorbed.

### `prune_merge_tree(basins, merges, threshold) -> (list[int], list[merge])`

Drops basins whose persistence is strictly less than `threshold`. Returns `(surviving_basin_ids, kept_merges)`. `kept_merges` is the subset of input merges whose `shallower` basin survives; the `deeper` basin always survives too (proof: if a kept merge has `shallower.persistence >= threshold` and `deeper.min_e <= shallower.min_e`, then `deeper.persistence >= shallower.persistence >= threshold`).

### `MergeTree(ws)`

Traversable rooted tree over the watershed basins, built from the `Watershed`
returned by `find_watershed_segmentation`. One node per basin, rooted at the
deepest basin (`basins[0]`, id 0). The tree is **physics-free**: it knows
nothing about ground states, fission, or any domain convention — it exposes
neutral traversal, membership, and geometry primitives that a consumer composes
with its own predicates.

The tree never copies the grid arrays: `labels`, `parents` and `merge_table`
are read-through properties of the owning `Watershed`, so `drop_labels()` on
either object releases them for both.

Public attributes and properties:

- **`ws`** — the owning `Watershed`.
- **`labels`**, **`parents`**, **`merge_table`** — the `Watershed` arrays (`None` after `drop_labels()`).
- **`neighborhood`**, **`dtype`**, **`fingerprint`** — forwarded from the `Watershed`.
- **`has_labels`** — `False` after `drop_labels()`.
- **`nodes`** — `dict[int, BasinNode]` keyed by basin ID.
- **`root`** — `int | None`; basin id 0 (the global minimum) when any basin exists, else `None`.

Methods:

| Method | Returns |
|---|---|
| `node(bid)` | the `BasinNode` for basin `bid` |
| `persistence(bid)` | topological persistence of `bid` (same value as `compute_persistence`) |
| `neighbors(bid)` | children + parent basin IDs |
| `path(a, b)` | inclusive tree path from `a` to `b` (through their lowest common ancestor) |
| `bfs(start, *, advance=None)` | iterator of `(basin_id, depth)`; `advance(from_bid, to_bid) -> bool` gates edge traversal — return `False` to skip that edge and the subtree beyond it |
| `drop_labels()` | release `labels`, `parents`, `merge_table` (4N + 2N bytes) for every holder of the `Watershed`; tree queries keep working, the four membership queries below then raise `RuntimeError("MergeTree labels were dropped ...")` |
| `basin_of_point(index)` | basin ID at grid cell `index` (`-1` for a `NaN` cell) |
| `basins_containing(points)` | `dict[basin_id, list[index]]` grouping the points by basin (`NaN` cells skipped) |
| `basin_mask(bid)` | boolean grid, `True` on the cells of basin `bid` |
| `touches_edge(bid, axis, side='max')` | `True` iff any cell of `bid` lies on the `axis` boundary; `side` is `'min'`, `'max'`, or `'both'`. Only the boundary face is compared — no full-grid temporary |

### `BasinNode`

Dataclass for a single node in a `MergeTree`. Fields:

| Field | Type | Meaning |
|---|---|---|
| `basin_id` | `int` | basin ID (the `nodes` key) |
| `minimum_index` | `tuple[int, ...]` | N-D index of the basin minimum |
| `minimum_energy` | `float` | energy at the minimum |
| `parent` | `int \| None` | parent basin ID (`None` for the root) |
| `saddle_to_parent` | `tuple[tuple[int, ...], float] \| None` | `(saddle_index, saddle_energy)` of the merge into the parent (`None` for the root) |
| `persistence` | `float` | topological persistence (`+inf` for the root) |
| `children` | `list[int]` | child basin IDs |

### Example

```python
import numpy as np
from pes_analyzer.topology import find_watershed_segmentation, MergeTree

# Three basins along a row: minima at columns 0, 4, 8.
energies = np.array([[0.0, 3.0, 8.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]])
ws = find_watershed_segmentation(energies)

tree = MergeTree(ws)
print(tree.root)                          # 0 (deepest basin)
print(tree.node(0).children)              # [1]
print(tree.node(2).saddle_to_parent)      # ((0, 6), 5.0)
print(tree.persistence(1))                # 7.0
print(tree.path(1, 2))                    # [1, 2]
print(tree.basin_of_point((0, 4)))        # 1
tree.drop_labels()                         # frees ws.labels for both holders
print(tree.path(1, 2))                    # [1, 2]  — tree queries still work
```

The deleted `identify_critical_points` helper (a domain-specific labeller for
ground state / secondary minimum / saddles / fission exit) is no longer part of
the library. That physics now lives in the consumer: a downstream MapMaker-style
script composes these `MergeTree` primitives with its own predicates (e.g. a
ground-state elongation constraint) to label critical points. See `USAGE.md` for
the end-to-end pipeline these primitives plug into.

---

## `plot`

`import pes_analyzer.plot` is explicit; `import pes_analyzer` does not load matplotlib. Every helper takes `ax=None` (then `matplotlib.pyplot.gca()`), draws, and returns the `Axes`. The library never creates files, sets a backend, or calls `show`; figures, titles and `savefig` are yours. Energy axes and colour bars are labelled `label`, default `"energy"`.

Map orientation: `values[i, j]` sits at `x = axes[0][i]`, `y = axes[1][j]`. Axis 0 is horizontal, the order of `keep` in `minimize_grid`, so `minimum, index = minimize_grid(E, keep=(c, a4))` plots with c horizontal. Transpose a grid stored image-wise (`E[iy, ix]`).

Where `axes` is a mapping, its keys label the plot axes and name the coordinate columns of `tables`; the kernels keep ignoring the names.

### `plot_map(values, axes=None, *, ax=None, cmap=None, levels=None, contours=None, mask=None, colorbar=True, label="energy", **mesh_kw) -> Axes`

- `values`: 2-D, at least two cells per side. `NaN` cells, and the masked cells of a masked array, are blank.
- `axes`: two 1-D arrays or a two-entry mapping (keys become the axis labels); `None` gives index coordinates labelled `index 0`, `index 1`.
- The fill is `pcolormesh` on **cell edges**: midpoints between neighbouring coordinates, half a step beyond the ends. Unequal steps show as unequal cells; no value is interpolated. The mesh is `rasterized=True`, so a PDF stays small while axes and text remain vector. `mesh_kw` reaches `pcolormesh` (`vmin`, `vmax`, `norm`, `alpha`, …).
- `levels`: a strictly increasing sequence of at least two boundaries → discrete bands (`BoundaryNorm(levels, cmap.N, extend="both")`); values outside the range take the colour map's under and over colours. For 1-unit bands with white above 20: `levels=np.arange(-10, 21)` and a colour map with `set_over("white")`. `levels` together with `norm` is a `ValueError`.
- `contours`: a sequence of levels → black labelled contour lines through the cell **centres**. Contour lines interpolate linearly between centres; the fill does not.
- `mask`: boolean array of the same shape; `True` cells are hatched over the fill. This is the jump-map overlay (`mask=jumps >= 5`) and serves any cell mask. The jump map itself is `plot_map(jumps, axes, label="jump (cells)")`.
- `colorbar=True` attaches a colour bar labelled `label`.

### `plot_path(indices, axes=None, *, keep=(0, 1), ax=None, **line_kw) -> Axes`

Projects a `(K, N)` path onto the two axes `keep` (the map's `keep`) with `index_to_coords`, or with the indices themselves when `axes` is `None`, and draws a line with a marker at every cell. `line_kw` overrides the style, so an `(M, N)` array of minima is drawn as points with `linestyle="none", marker="o"`. A float array, or a `keep` that is not two distinct axes in `[0, N)`, is a `ValueError`; non-integer `keep` entries are a `TypeError`, never truncated.

### `plot_profile(values, *, indices=None, axes=None, profile=None, ax=None, label="energy", **line_kw) -> Axes`

Draws a `(K,)` quantity along a path: the energy profile that `find_minimax_path` and `find_steepest_descent_path` return, or the cumulative action or climb that `find_least_action_path` and `find_minimum_ascent_path` return (then pass `label="action"` or `"climb"`). x is `path_length(indices, axes)` when `indices` is given (labelled `length`, or `length (cells)` without `axes`), else the step number. `profile`, a `PathProfile` of the same values, marks its minima with circles and its saddles with squares. Several paths: call again on the same `ax`.

### `merge_tree_layout(tree, *, min_persistence=0.0, max_basins=2000) -> MergeTreeLayout`

Pure Python; `tree` is a `MergeTree` or a `Watershed`. Selects the basins to draw: survivors of `min_persistence`, then the `max_basins` most persistent (ties to the lower id; `None` disables the cap), closed under parent. Every root (a basin that never merged, infinite persistence) is always drawn, so a grid cut by `NaN` walls gives a forest with one block per root. Placement is crossing-free: each subtree owns a contiguous block of integer slots; a basin's children, sorted by saddle energy, take sides alternately right, left, right, … so the lowest saddle sits nearest; roots go left to right by id. See `ALGORITHMS.md`.

Fields of the frozen dataclass:

| Field | Content |
|---|---|
| `basin_ids` | drawn basin ids, ascending |
| `x` | basin id → slot in `[0, n)` |
| `branches` | basin id → `(x, e_min, e_top)`; `e_top` is the basin's own saddle energy, `top` for a root |
| `connectors` | `(x_child, x_parent, e_saddle, child_id)`, ascending saddle energy |
| `top` | the highest of the connector energies and the drawn minima, plus 5% of the span down to the lowest drawn minimum; plus 1 when the span is zero |

```python
import numpy as np
from pes_analyzer.plot import merge_tree_layout
from pes_analyzer.topology import find_watershed_segmentation

energies = np.array([[0.0, 3.0, 8.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]])
layout = merge_tree_layout(find_watershed_segmentation(energies))
print(layout.x)             # {0: 0, 1: 1, 2: 2}
print(layout.connectors)    # [(2, 1, 5.0, 2), (1, 0, 8.0, 1)]
print(layout.branches)      # {0: (0, 0.0, 8.4), 1: (1, 1.0, 8.0), 2: (2, 2.0, 5.0)}
```

### `plot_merge_tree(tree, *, ax=None, min_persistence=0.0, max_basins=2000, labels=None, saddle_labels=None, label="energy", color="0.3") -> Axes`

Draws the layout's branches and connectors as one `LineCollection`, x ticks hidden. `labels` maps a basin id to text placed at its minimum (with a circle); `saddle_labels` maps a basin id to text at the midpoint of the connector where that basin merges (with a square). Plain `annotate`, no de-overlap: the pruned tree is sparse, and anything denser is your figure. An id that is not drawn, or a root in `saddle_labels`, is a `ValueError` naming the id, so a pruned basin is never silently dropped. For your own annotations call `merge_tree_layout` with the same arguments.

```python
from matplotlib.figure import Figure
from pes_analyzer.grid import jump_map, minimize_grid
from pes_analyzer.plot import plot_map, plot_merge_tree, plot_path
from pes_analyzer.synthetic import hidden_barrier
from pes_analyzer.topology import MergeTree, find_watershed_segmentation

surf = hidden_barrier()
axes = {"x": np.linspace(-1.5, 1.5, 61), "y": np.linspace(-0.5, 0.5, 21), "z": np.linspace(-1.5, 1.5, 61)}
energies = surf.sample(axes)
minimum, index = minimize_grid(energies, keep=(0, 1))
jumps = jump_map(index, keep=(0, 1))

fig = Figure(figsize=(10, 4))
ax_map, ax_tree = fig.subplots(1, 2)
plot_map(minimum, [axes["x"], axes["y"]], ax=ax_map, levels=np.arange(-1.0, 7.0), mask=jumps >= 5)
tree = MergeTree(find_watershed_segmentation(energies))
plot_merge_tree(tree, ax=ax_tree, min_persistence=0.1, labels={0: "A"})
fig.savefig("hidden_barrier.pdf")
```

## `tables`

A table is a plain `dict[str, numpy.ndarray]`: columns of one length in insertion order, so `pandas.DataFrame(table)` works if you want it. Columns hold integers, booleans and floats only. Index columns are `index_0 … index_{N−1}` (`int64`); coordinate columns appear only when `axes` is given and are named by the mapping keys or `x0 … x{N−1}` for a sequence (`float64`, from `index_to_coords`); values are `float64` (`float32` input is cast exactly). A coordinate column whose name coincides with another column of the table (an axis called `energy`, `step`, `index_0`, or `energy` under the `min_` prefix) is a `ValueError`.

### `minima_table(points, axes=None, *, ndim=None) -> table`

Any list of `(index, energy)` pairs (`find_minima_grid`, `find_maxima_grid`, a `Watershed`'s `basins`) in input order: `index_*`, coordinates, `energy`. N comes from the first point, else from `axes`, else from `ndim`; an empty list with neither is a `ValueError`.

### `basins_table(tree, axes=None, *, min_persistence=0.0) -> table`

The merge tree as a table, one row per selected basin, ascending id: `basin`, `min_index_*`, `min_<name>`, `min_energy`, `parent`, `saddle_index_*`, `saddle_<name>`, `saddle_energy`, `persistence`. The parent of a selected basin is always selected. Every root has `parent` −1, saddle indices −1, saddle coordinates and `saddle_energy` `NaN`, `persistence` `inf`.

### `path_table(indices, energies, axes=None) -> table`

`step`, `index_*`, coordinates, `length` (cumulative; in cells when `axes` is `None`), `energy`. `energies` are the energies **at the path cells**: for `find_least_action_path` and `find_minimum_ascent_path`, whose second output is a cumulative weight, gather them with `E[tuple(idx.T)]` and add the weight as one more column (`table["action"] = action`) before writing.

### `write_csv(path, table)` and `read_csv(path) -> table`

Comma separated, header row, `\n` endings, UTF-8, no index column. Integers and booleans as integers, floats as Python `repr` (shortest round trip; `nan`, `inf`, `-inf` spelled so), so the file is exact and deterministic. Any other dtype is a `ValueError`. `read_csv` is the inverse for these files only: a column is `int64` when every entry parses as an integer, else `float64`; a header-only file comes back with zero-length `float64` columns, the one case where a dtype changes.

```python
import numpy as np
from pes_analyzer.tables import minima_table, path_table, write_csv, read_csv

axes = {"x": np.array([0.0, 1.0, 3.0]), "y": np.array([10.0, 20.0, 40.0, 80.0])}
table = minima_table([((0, 1), 2.5), ((2, 3), -1.0)], axes)
print(list(table))          # ['index_0', 'index_1', 'x', 'y', 'energy']
write_csv("minima.csv", table)
# index_0,index_1,x,y,energy
# 0,1,0.0,20.0,2.5
# 2,3,3.0,80.0,-1.0
back = read_csv("minima.csv")
print(back["index_0"].dtype, back["x"])     # int64 [0. 3.]
path = path_table(np.array([[0, 0], [1, 1], [2, 3]]), [1.0, 2.0, 0.5], axes)
print(list(path))           # ['step', 'index_0', 'index_1', 'x', 'y', 'length', 'energy']
```

```python
from pes_analyzer.tables import basins_table
from pes_analyzer.topology import MergeTree, find_watershed_segmentation

energies = np.array([[0.0, 3.0, 8.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]])
tree = MergeTree(find_watershed_segmentation(energies))
table = basins_table(tree)
print(table["basin"], table["parent"])              # [0 1 2] [-1  0  1]
print(table["saddle_index_1"], table["saddle_energy"])   # [-1  2  6] [nan  8.  5.]
print(table["persistence"])                         # [inf  7.  3.]
print(basins_table(tree, min_persistence=4.0)["basin"])  # [0 1]
```

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ValueError: energies must be C-contiguous` | passed a slice or transposed array | `np.ascontiguousarray(arr)` before calling |
| `ValueError: ndim must be in [2, 7]` | wrong array shape | reshape or filter inactive axes |
| `IndexError: index ... out of bounds for axis ...` / `negative index ... is not allowed` | `start`/`end` outside the grid | check the index values (a wrong tuple *length* is a `ValueError`) |
| `ValueError: energy at \`start\` is NaN` | endpoint cell is masked | pick an endpoint inside the non-`NaN` region |
| `ValueError: tree was built without parents=True` | `find_minimax_path(tree=)` on a watershed without flood parents | rebuild with `find_watershed_segmentation(energies, parents=True)` |
| `ValueError: tree was built from a different energy grid` | `tree=` with an array of different dtype/values | pass the exact array the watershed was built from |
| `RuntimeError: MergeTree labels were dropped` | membership query after `drop_labels()` | query before dropping, or rebuild the watershed |
| `ValueError: neighborhood_range must be in [1, 5]` | passed `0` or `> 5` | choose `neighborhood_range ∈ {1, 2, 3, 4, 5}` |
| `ValueError: confirm_range must be in [1, 5]` | passed `0` or `> 5` | choose `confirm_range ∈ {1, 2, 3, 4, 5}` or `None` |
| `ValueError: confirm_range (c) must be >= neighborhood_range (n)` | `confirm_range < neighborhood_range` | raise `confirm_range` or lower `neighborhood_range` (most callers want `neighborhood_range=1, confirm_range=R`) |
| `ValueError: end mask has no True cell` | empty target set | set at least one `True` cell, or pass an index tuple |
| `ValueError: cost array contains a negative value` | `find_least_action_path` with a negative cost | clip the cost at 0 (`np.clip(c, 0, None)`) |
| `ValueError: ... contains an infinite value` | `±inf` in the input of a search kernel | replace with `NaN` (a wall) or a finite value |
| `ValueError: axis 2 must be strictly increasing` | descending or repeated axis coordinates | sort the axis, or pass the `axes` dict `build_dense` returned |
| `IndexError: index out of bounds for axis 0 with size 3` from `index_to_coords` | a `-1` row of `minimize_grid` | mask with `index[..., 0] >= 0` first |
| `ValueError: column 'energy' is defined twice` | an axis name collides with a table column | rename the axis in the `axes` mapping |
| `ValueError: basin 93940 is not drawn (pruned or capped)` | `labels=` names a basin outside the `plot_merge_tree` selection | lower `min_persistence` or raise `max_basins` |
| `ValueError: pass either levels= or norm=, not both` | both discrete bands and a norm given to `plot_map` | drop one |
| `ValueError: tables hold integer, boolean and floating columns only` | a string or object column in `write_csv` | keep text out of the table; the format is numeric |
