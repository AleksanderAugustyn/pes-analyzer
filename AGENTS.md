# AGENTS.md

Guidance for coding agents (Claude Code, Codex) working in this repository.

## What this is

`pes_analyzer` is a Python package whose compute kernels are written in Rust (PyO3 + `numpy`/`ndarray`), built with [maturin](https://www.maturin.rs/). It finds minima, maxima, saddle points, and watershed topology on dense N-D potential energy surface grids (N ∈ [2, 7]). Python is the API surface; Rust is the inner loops.

## Build & test

```bash
maturin develop --release   # rebuild python/pes_analyzer/_native.abi3.so and install into active venv
pytest tests/               # Python integration tests — run against the compiled .so
cargo test                  # Rust unit tests for kernels/helpers — no Python rebuild needed
pytest tests/test_iwf_nd.py::test_name   # single test
```

- **Always build with `--release`.** Debug kernels are dramatically slower; the test suite goes from seconds to minutes.
- **`pytest` tests the compiled extension, not the source.** After any `src/` change you must re-run `maturin develop --release` or the tests still exercise the old `.so`. Python-only edits (e.g. `python/pes_analyzer/grid.py`, `topology/_tree.py`) need no rebuild.
- The release profile uses `lto = "fat"` + `codegen-units = 1`; expect tens of seconds of link time.

## Architecture essentials

Detailed tour in `ARCHITECTURE.md`; consumer API + algorithm internals are bundled in `python/pes_analyzer/_docs/` (`API.md`, `ALGORITHMS.md`, `USAGE.md`) and ship in the wheel. Key points:

- **Layering:** pure-Rust kernels in `src/<area>/<algo>.rs` (unit-testable with `ndarray` views) ← thin PyO3 wrappers in `src/<area>/mod.rs` ← Python re-exports in `python/pes_analyzer/`. Shared internals (N-D indexing, union-find, input validation) live in `src/common/` and contain no PyO3.
- **All compute runs inside `py.allow_threads(|| ...)`**, operating on `PyReadonlyArrayDyn` views. Keep new kernels GIL-free the same way.
- **Two submodule registration patterns** — match the existing one when adding a submodule:
  - *Rust-only* (`saddle`, `extrema`): registered by patching `sys.modules` in `<area>/mod.rs::register`, with a hand-written `python/pes_analyzer/<area>/__init__.pyi` stub so type checkers can see the dynamically-registered module. **Both halves are required.**
  - *Rust + Python helpers* (`topology`): a real on-disk package with `__init__.py` that imports from `pes_analyzer._native.<area>` and re-exports. No `sys.modules` patch in that case.
- PyO3 wrappers must validate inputs through `common::validate::*`, reject non-contiguous arrays, and use `PyTuple::new_bound` for `tuple[int, ...]` returns (the default `Vec<usize>` conversion yields a `list`).

## Conventions

- **Rust edition 2024** (needs Rust ≥ 1.85). PyO3 0.22 with `abi3-py310`.
- **`NaN` cells are impassable walls** in the kernels (excluded from sorting/neighbour scans). `NaN` at a required `start`/`end` is a usage error → `ValueError` raised in the wrapper before the kernel runs.
- **Flood kernels (`find_iwf_grid`, `find_watershed_segmentation`, `find_minimax_path`) default to axis-only (2N) von Neumann neighbours**, with opt-in `neighborhood="moore"` (3ᴺ−1, range 1). **Extrema use the king-move Chebyshev stencil** with `neighborhood_range`. The asymmetry is physically motivated (see `ALGORITHMS.md`) — don't "unify" them, and pass the `MergeTree` (or `Watershed`) to `find_minimax_path(tree=...)` so the minimax path reuses the flood state and its neighbourhood — a bare `neighborhood` on the minimax path must match the tree's. **Length-dependent kernels** (`find_steepest_descent_path`, `find_least_action_path`) take `axes=` and default to Moore; `find_minimum_ascent_path` defaults to von Neumann. `synthetic` surfaces carry exact critical points with incidence; tests on uniform, anisotropic and variable-step grids live in `tests/test_variable_step.py`.
- **rayon:** extremum scans and the flood sort are parallel; results never depend on thread count; `RAYON_NUM_THREADS` limits the pool.
- **Version is duplicated** in `Cargo.toml` and `pyproject.toml`; `Cargo.toml` is the source of truth. Update both by hand when bumping.