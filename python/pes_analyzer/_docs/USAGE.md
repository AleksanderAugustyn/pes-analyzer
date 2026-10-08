# Usage: the pes_analyzer pipeline end-to-end

This walks the typical workflow from a raw long-form table to labelled basins and
a minimax path. Every function here has a full contract in `API.md`; this
page shows how they compose. For algorithm details and the neighborhood-stencil
rationale, see `ALGORITHMS.md`.

## The pipeline

```
build_dense            sparse (coords, value) rows  ->  dense N-D grid
find_minima_grid       dense grid                   ->  local minima
find_watershed_segmentation  dense grid             ->  basin labels + merge tree
MergeTree              (labels, basins, merges)     ->  traversable basin tree
find_minimax_path + analyze_path_profile     ->  barrier profile between two cells
plot.plot_map / plot_path / plot_profile / plot_merge_tree   ->  figures on your Axes
tables.*_table + write_csv                                  ->  CSV reference outputs
```

## 1. Build the dense grid

`build_dense` pivots a long-form table into the dense `float64` array every other
function consumes. Dict insertion order fixes the axis order; missing cells become
`NaN`.

```python
import numpy as np
from pes_analyzer.grid import build_dense

coords = {
    "x": np.array([0.0, 0.0, 1.0, 1.0]),
    "y": np.array([0.0, 1.0, 0.0, 1.0]),
}
values = np.array([10.0, 11.0, 20.0, 21.0])

energies, axes = build_dense(coords, values)
# energies.shape == (2, 2); axes == {"x": [0., 1.], "y": [0., 1.]}
```

Map an N-D grid index back to physical coordinates with `axes`: cell `(i, j)` sits
at `(axes["x"][i], axes["y"][j])`.

## 2. Find the minima

```python
from pes_analyzer.extrema import find_minima_grid

minima = find_minima_grid(energies)            # [((i, ...), energy), ...] ascending
```

Extrema use the **Chebyshev king-move stencil** (`3**N - 1` neighbours by default).
Widen with `neighborhood_range=R`, or use the cheaper two-pass idiom
`find_minima_grid(energies, neighborhood_range=1, confirm_range=R)` for an
identical result. `find_maxima_grid` and `find_extrema_grid` are the dual and the
combined single-sweep variants.

## 3. Segment the whole surface

`find_watershed_segmentation` floods the entire grid and records every basin merge
as a saddle event — the full generalization of the two-point `find_iwf_grid`. Ask
for `parents=True` when you will also want minimax paths (step 5).

```python
from pes_analyzer.topology import find_watershed_segmentation

ws = find_watershed_segmentation(energies, parents=True)
ws.labels       # int32 array (shape == energies.shape); -1 marks NaN cells; read-only
ws.basins       # [((min_index, ...), min_energy), ...] ascending; basins[0] is the global min
ws.merges       # [((saddle_index, ...), saddle_energy, deeper_id, shallower_id), ...] ascending
ws.parents      # uint16 flood-parent codes (None without parents=True)
ws.merge_table  # uint32 (M, 5) per-merge cell/basin ids used by the path reconstruction
```

## 4. Traverse the merge tree

`MergeTree` turns the `Watershed` into a rooted, traversable tree. It is
**physics-free**: it gives you neutral traversal, membership, and geometry
primitives, and you compose them with your own predicates to label ground states,
fission exits, or whatever your domain needs. It never copies the grid arrays.

```python
from pes_analyzer.topology import MergeTree

tree = MergeTree(ws)
tree.root                       # basin id 0 (global minimum), or None if no basins
node = tree.node(0)             # BasinNode: minimum_index, minimum_energy, parent,
                                #            children, saddle_to_parent, persistence
tree.persistence(1)             # saddle_energy - basin_min_energy (root is +inf)
tree.path(1, 2)                 # tree path through the lowest common ancestor
tree.basin_of_point((0, 1))     # basin id at a grid cell (-1 for NaN)
tree.basin_mask(1)              # boolean grid of basin 1
tree.touches_edge(1, axis=0)    # does basin 1 reach an axis-0 boundary? (face slice only)
```

Filter noise with persistence: `compute_persistence(ws.basins, ws.merges)` gives the
per-basin value, and `prune_merge_tree(ws.basins, ws.merges, threshold)` drops
basins below a persistence floor.

## 5. Profile the barrier between two cells

`find_minimax_path` returns the deep minimax path: it minimizes the highest
energy crossed (passing through exactly the watershed saddles) and descends to true
basin minima between barriers. With `tree=` it reuses the watershed's flood state
instead of flooding again, and the neighbourhood is the tree's. Feed the energy
profile to `analyze_path_profile` to extract the alternating minima and saddles.

```python
from pes_analyzer.topology import analyze_path_profile, find_minimax_path

result = find_minimax_path(energies, start=(0, 0), end=(1, 1), tree=tree)
if result is not None:
    path_indices, path_energies = result        # (K, N) int64, (K,) float64
    profile = analyze_path_profile(path_energies, min_persistence=0.0)
    # profile.minima, profile.saddles: lists of (path_index, energy)
    # recover grid coords of path_index k via path_indices[k]
```
`find_minimax_path` returns `None` when `start` and `end` lie in disjoint
non-`NaN` regions. Without `tree=` it floods the grid itself (early-stopped at the
endpoints; `neighborhood` defaults to `"von_neumann"`).

## 5b. Minimise onto two axes, and see what the map hides

```python
from pes_analyzer.grid import jump_map, minimize_grid

minimum, index = minimize_grid(energies, keep=(0, 1))   # map over axes 0 and 1, minimised over the rest
jumps = jump_map(index, keep=(0, 1))                      # cells the minimiser moves between map neighbours
```

`minimum` is the map; `index` gathers any other grid at the minimiser (mask with `index[..., 0] >= 0`); `jumps` marks where the minimiser switches valley. Compare the map's barrier with the merge tree's saddle level between the same two basins: on the `synthetic.hidden_barrier` surface they differ by a factor of three.

## 5c. Other path kinds

`find_steepest_descent_path(energies, start, axes=axes)` follows the slope from a cell; `find_least_action_path(cost, start, end, axes=axes)` minimises ∫ cost ds for a cost you build; `find_minimum_ascent_path(energies, start, end)` minimises the total climb. `end` may be a boolean mask of target cells. See the "Path kinds" table in `API.md`.

## 6. Plot

`pes_analyzer.plot` is imported explicitly and draws on an `Axes` you own; nothing is interpolated or smoothed, and saving is yours. A self-contained run on the `synthetic.hidden_barrier` surface (the walkthrough's toy grid above is 2-D and too small to minimise):

```python
import matplotlib.pyplot as plt
import numpy as np
from pes_analyzer.extrema import find_minima_grid
from pes_analyzer.grid import jump_map, minimize_grid
from pes_analyzer.plot import plot_map, plot_merge_tree, plot_path, plot_profile
from pes_analyzer.synthetic import hidden_barrier
from pes_analyzer.topology import MergeTree, analyze_path_profile, find_minimax_path, find_watershed_segmentation

axes = {"x": np.linspace(-1.5, 1.5, 61), "y": np.linspace(-0.5, 0.5, 21), "z": np.linspace(-1.5, 1.5, 61)}
energies = hidden_barrier().sample(axes)
minima = find_minima_grid(energies)
tree = MergeTree(find_watershed_segmentation(energies, parents=True))
minimum, index = minimize_grid(energies, keep=(0, 1))
jumps = jump_map(index, keep=(0, 1))
path_indices, path_energies = find_minimax_path(energies, tree.ws.basins[0][0], tree.ws.basins[1][0], tree=tree)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
plot_map(minimum, [axes["x"], axes["y"]], ax=axs[0, 0], mask=jumps >= 5)   # map; hatched where the minimiser jumps
plot_path(path_indices, axes, keep=(0, 1), ax=axs[0, 0], color="red")     # the minimax path projected onto it
plot_map(jumps, [axes["x"], axes["y"]], ax=axs[0, 1], label="jump (cells)")
plot_merge_tree(tree, ax=axs[1, 0], min_persistence=0.1, labels={0: "A", 1: "B"})
plot_profile(path_energies, indices=path_indices, axes=axes, profile=analyze_path_profile(path_energies), ax=axs[1, 1])
fig.savefig("hidden_barrier.pdf")
```

On the map the barrier between the two deep basins rises 1.5 above them, while the merge tree's saddle between them rises 5.51 (at 5.0125 on this grid, the minima at −0.5): the valley switch the hatched cells mark is where the map hides it.

## 7. Save

One table per object, one CSV writer, continuing the run above. The same files are the reference outputs of the shipped examples.

```python
from pes_analyzer.tables import basins_table, minima_table, path_table, write_csv

write_csv("minima.csv", minima_table(minima, axes))
write_csv("basins.csv", basins_table(tree, axes, min_persistence=0.1))
write_csv("path.csv", path_table(path_indices, path_energies, axes))
```

`read_csv` reads them back as the same column dicts.

## 8. Release the grid arrays

Once membership queries and paths are done, drop the grid-sized arrays — they are
the bulk of the resident memory (4N + 2N bytes):

```python
tree.drop_labels()              # frees ws.labels / ws.parents / ws.merge_table for every holder
tree.path(1, 2)                 # tree queries keep working
tree.basin_of_point((0, 1))     # -> RuntimeError: labels were dropped
```

## Contracts you must get right

- **Neighborhood asymmetry.** Flood kernels (`find_iwf_grid`,
  `find_watershed_segmentation`, `find_minimax_path`) default to axis-only
  von Neumann neighbours (`2N`), with opt-in `neighborhood="moore"` (`3**N - 1`).
  Extrema use the Chebyshev king-move stencil with `neighborhood_range`. This
  asymmetry is intentional — see `ALGORITHMS.md`.
- **Match `neighborhood` across the tree and the path.** Pass the `MergeTree`
  (or `Watershed`) to `find_minimax_path(tree=...)` so the path reuses the
  flood state and its neighbourhood; a bare `neighborhood` on the path must match
  the tree's or their saddles disagree. For one `neighborhood`, `max(path_energies)`
  equals the `find_iwf_grid` saddle energy between the same endpoints.
- **Threading.** Extremum scans and the flood sort run on the rayon pool (all
  cores; `RAYON_NUM_THREADS` limits it). Results never depend on the thread count.
- **`NaN` cells are impassable walls.** They are excluded from sorting and
  neighbour scans. A `NaN` at a required `start`/`end` is a usage error and raises
  `ValueError`.
- **Arrays must be C-contiguous `float32` or `float64`** of ndim `N ∈ [2, 7]`. Pass a slice or
  transpose through `np.ascontiguousarray(arr)` first.
- **Maps are axis-0-horizontal.** `plot_map(values, axes)` puts `values[i, j]` at `x = axes[0][i]`; pass `values.T` for an image-wise grid.

## Locating these docs at runtime

```python
import pes_analyzer
pes_analyzer.docs_path()        # -> Path to the bundled _docs/ directory
```
