"""Integration tests for pes_analyzer.grid.build_dense."""

import numpy as np
import pytest

from pes_analyzer.grid import build_dense


def test_2d_round_trip_minimal():
    coords = {
        "c":  np.array([1.0, 1.0, 2.0, 2.0]),
        "a4": np.array([0.1, 0.2, 0.1, 0.2]),
    }
    values = np.array([10.0, 20.0, 30.0, 40.0])
    dense, axes = build_dense(coords, values)

    assert dense.shape == (2, 2)
    assert dense.dtype == np.float64
    assert dense[0, 0] == 10.0
    assert dense[0, 1] == 20.0
    assert dense[1, 0] == 30.0
    assert dense[1, 1] == 40.0
    np.testing.assert_array_equal(axes["c"], [1.0, 2.0])
    np.testing.assert_array_equal(axes["a4"], [0.1, 0.2])


def test_missing_cell_is_nan():
    coords = {
        "c":  np.array([1.0, 1.0, 2.0]),
        "a4": np.array([0.1, 0.2, 0.1]),
    }
    values = np.array([10.0, 20.0, 30.0])
    dense, _ = build_dense(coords, values)
    assert dense.shape == (2, 2)
    # (c=2.0, a4=0.2) is the missing cell.
    assert np.isnan(dense[1, 1])


def test_duplicate_rows_last_write_wins():
    coords = {
        "c":  np.array([1.0, 1.0]),
        "a4": np.array([0.5, 0.5]),
    }
    values = np.array([10.0, 99.0])
    dense, _ = build_dense(coords, values)
    assert dense.shape == (1, 1)
    assert dense[0, 0] == 99.0


def test_single_axis_value_is_not_squeezed():
    coords = {
        "c":  np.array([1.0, 1.0]),
        "a4": np.array([0.1, 0.2]),
    }
    values = np.array([10.0, 20.0])
    dense, axes = build_dense(coords, values)
    assert dense.shape == (1, 2)
    assert len(axes["c"]) == 1


def test_insertion_order_preserved():
    coords = {
        "z": np.array([1.0, 2.0, 1.0, 2.0]),
        "a": np.array([0.0, 0.0, 1.0, 1.0]),
    }
    values = np.array([10.0, 20.0, 30.0, 40.0])
    dense, axes = build_dense(coords, values)
    assert dense.shape == (2, 2)  # axis 0 is 'z', axis 1 is 'a'
    # dense[z_idx=1, a_idx=0] should correspond to row (z=2.0, a=0.0) → 20.0
    assert dense[1, 0] == 20.0
    assert list(axes.keys()) == ["z", "a"]


def test_5d_construction():
    coords = {
        "c":  np.array([1.0, 1.0, 1.0]),
        "a3": np.array([0.0, 0.0, 0.0]),
        "a4": np.array([0.0, 0.1, 0.0]),
        "a5": np.array([0.0, 0.0, 0.0]),
        "a6": np.array([0.0, 0.0, 0.1]),
    }
    values = np.array([5.0, 6.0, 7.0])
    dense, axes = build_dense(coords, values)
    assert dense.shape == (1, 1, 2, 1, 2)
    assert dense[0, 0, 0, 0, 0] == 5.0
    assert dense[0, 0, 1, 0, 0] == 6.0
    assert dense[0, 0, 0, 0, 1] == 7.0
    # The (a4=0.1, a6=0.1) cell is missing.
    assert np.isnan(dense[0, 0, 1, 0, 1])


def test_rejects_empty_coords():
    with pytest.raises(ValueError, match="at least one axis"):
        build_dense({}, np.array([1.0]))


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length 3.*length 2"):
        build_dense(
            {"c": np.array([1.0, 1.0, 2.0])},
            np.array([10.0, 20.0]),
        )


def test_values_dtype_coerced_to_float64():
    coords = {"c": np.array([1.0, 2.0]), "a4": np.array([0.1, 0.1])}
    values = np.array([10, 20], dtype=np.int32)
    dense, _ = build_dense(coords, values)
    assert dense.dtype == np.float64


def test_output_is_c_contiguous():
    coords = {
        "c":  np.array([1.0, 1.0, 2.0, 2.0]),
        "a4": np.array([0.1, 0.2, 0.1, 0.2]),
    }
    values = np.array([10.0, 20.0, 30.0, 40.0])
    dense, _ = build_dense(coords, values)
    assert dense.flags["C_CONTIGUOUS"]


def test_build_dense_float32_preserved():
    coords = {"x": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
              "y": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)}
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    dense, axes = build_dense(coords, vals)
    assert dense.dtype == np.float32


def test_build_dense_dtype_override():
    coords = {"x": np.array([0.0, 1.0], dtype=np.float64)}
    vals = np.array([1.0, 2.0], dtype=np.float64)
    dense, _ = build_dense(coords, vals, dtype=np.float32)
    assert dense.dtype == np.float32


def test_build_dense_float64_back_compat():
    coords = {"x": np.array([0.0, 1.0], dtype=np.float64)}
    vals = np.array([1.0, 2.0], dtype=np.float64)
    dense, _ = build_dense(coords, vals)
    assert dense.dtype == np.float64


from pes_analyzer.grid import _normalize_axes, index_to_coords, path_length


def test_normalize_axes_accepts_mapping_and_sequence():
    ax = _normalize_axes({"x": [0, 1, 3], "y": np.array([0.5, 1.5])}, (3, 2))
    assert [a.tolist() for a in ax] == [[0.0, 1.0, 3.0], [0.5, 1.5]]
    assert all(a.dtype == np.float64 and a.flags["C_CONTIGUOUS"] for a in ax)
    assert _normalize_axes([[0.0, 1.0, 3.0], [0.5, 1.5]], (3, 2))[0].tolist() == [0.0, 1.0, 3.0]
    assert _normalize_axes(None, (3, 2)) is None


def test_normalize_axes_allows_a_length_one_axis():
    ax = _normalize_axes([[2.5], [0.0, 1.0]], (1, 2))
    assert ax[0].tolist() == [2.5]


@pytest.mark.parametrize(
    "axes, shape",
    [
        ([[0.0, 1.0]], (2, 2)),                      # wrong number of axes
        ([[0.0, 1.0, 2.0], [0.0, 1.0]], (2, 2)),     # wrong length
        ([[0.0, np.nan], [0.0, 1.0]], (2, 2)),       # non-finite
        ([[1.0, 0.0], [0.0, 1.0]], (2, 2)),          # decreasing
        ([[0.0, 0.0], [0.0, 1.0]], (2, 2)),          # not strictly increasing
        ([[[0.0, 1.0]], [0.0, 1.0]], (2, 2)),        # not 1-D
    ],
)
def test_normalize_axes_rejects_bad_input(axes, shape):
    with pytest.raises(ValueError):
        _normalize_axes(axes, shape)


def test_index_to_coords_and_path_length():
    axes = {"x": np.array([0.0, 1.0, 3.0]), "y": np.array([10.0, 20.0, 40.0, 80.0])}
    path = np.array([[0, 0], [1, 1], [2, 3]])
    np.testing.assert_array_equal(index_to_coords(path, axes), [[0.0, 10.0], [1.0, 20.0], [3.0, 80.0]])
    np.testing.assert_allclose(path_length(path, axes), [0.0, 10.04987562112089, 70.08319970033543])
    np.testing.assert_allclose(path_length(path), [0.0, 1.4142135623730951, 3.6502815398728847])
    assert path_length(path[:1]).tolist() == [0.0]


def test_index_to_coords_rejects_negative_and_out_of_range():
    axes = [np.array([0.0, 1.0, 3.0]), np.array([10.0, 20.0])]
    with pytest.raises(IndexError):
        index_to_coords(np.array([[-1, 0]]), axes)       # the -1 rows of minimize_grid
    with pytest.raises(IndexError):
        index_to_coords(np.array([[0, 2]]), axes)
    with pytest.raises(ValueError):
        index_to_coords(np.array([[0.0, 1.0]]), axes)    # not integers
    with pytest.raises(ValueError):
        index_to_coords(np.array([[0, 1, 1]]), axes)     # wrong N


import itertools
import warnings

from pes_analyzer.grid import jump_map, minimize_grid
from pes_analyzer.synthetic import hidden_barrier


def _noisy_grid():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(5, 6, 7, 4))
    e[rng.random(e.shape) < 0.2] = np.nan
    e[2, 3] = np.nan                       # a whole column without a valid cell for keep=(0, 1)
    return e


def _reference_min(e, keep):
    hidden = tuple(a for a in range(e.ndim) if a not in keep)
    moved = np.transpose(e, keep + hidden).reshape(tuple(e.shape[k] for k in keep) + (-1,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns
        return np.nanmin(moved, axis=-1)


@pytest.mark.parametrize("keep", [k for r in (1, 2, 3) for k in itertools.permutations(range(4), r)])
def test_minimize_grid_matches_nanmin_for_every_axis_choice(keep):
    e = _noisy_grid()
    minimum, index = minimize_grid(e, keep)
    np.testing.assert_array_equal(minimum, _reference_min(e, keep))
    assert minimum.dtype == e.dtype and index.dtype == np.intp
    assert index.shape == minimum.shape + (4,)
    valid = index[..., 0] >= 0
    np.testing.assert_array_equal(valid, ~np.isnan(minimum))
    assert (index[~valid] == -1).all()
    gathered = e[tuple(np.moveaxis(index[valid], -1, 0))]        # index round-trip
    np.testing.assert_array_equal(gathered, minimum[valid])
    grid_pos = np.indices(minimum.shape)
    for j, k in enumerate(keep):                                  # kept components = map position
        np.testing.assert_array_equal(index[..., k][valid], grid_pos[j][valid])


def test_minimize_grid_non_ascending_keep_and_non_contiguous_input():
    e = np.asfortranarray(_noisy_grid())
    assert not e.flags["C_CONTIGUOUS"]
    minimum, index = minimize_grid(e, (2, 0))
    assert minimum.shape == (7, 5)
    np.testing.assert_array_equal(minimum, _reference_min(np.ascontiguousarray(e), (2, 0)))
    valid = index[..., 0] >= 0
    np.testing.assert_array_equal(index[..., 2][valid], np.indices((7, 5))[0][valid])


def test_minimize_grid_single_axis_float32_and_threads():
    e = _noisy_grid()
    minimum, index = minimize_grid(e, 1)
    assert minimum.shape == (6,) and index.shape == (6, 4)
    m32, _ = minimize_grid(e.astype(np.float32), (3, 0))
    assert m32.dtype == np.float32 and m32.shape == (4, 5)
    m1, i1 = minimize_grid(e, (0, 2), threads=1)
    m8, i8 = minimize_grid(e, (0, 2), threads=8)
    np.testing.assert_array_equal(m1, m8)
    np.testing.assert_array_equal(i1, i8)


def test_minimize_grid_ties_go_to_the_first_hidden_cell_in_c_order():
    t = np.zeros((2, 3, 4))
    t[0, 1, 2] = -1.0
    t[0, 1, 3] = -1.0
    _, index = minimize_grid(t, 0)
    assert tuple(index[0]) == (0, 1, 2)


def test_minimize_grid_keeps_a_plus_infinity_minimiser_like_nanmin():
    e = np.array([[np.nan, np.inf, np.nan], [np.nan, np.nan, np.nan], [1.0, np.inf, np.nan]])
    minimum, index = minimize_grid(e, 0)
    np.testing.assert_array_equal(minimum, [np.inf, np.nan, 1.0])
    assert index.tolist() == [[0, 1], [-1, -1], [2, 0]]


@pytest.mark.parametrize("keep", [(), (0, 1, 2, 3), (0, 0), (4,), (-1,), (0.5,)])
def test_minimize_grid_rejects_bad_keep(keep):
    with pytest.raises(ValueError):
        minimize_grid(_noisy_grid(), keep)


def test_minimize_grid_rejects_bad_input():
    with pytest.raises(ValueError):
        minimize_grid(np.zeros((3, 3), dtype=np.int64), 0)
    with pytest.raises(ValueError):
        minimize_grid(np.zeros(5), 0)
    with pytest.raises(ValueError):
        minimize_grid(np.zeros((3, 3)), 0, threads=0)


def test_jump_map_on_the_hidden_barrier():
    surf = hidden_barrier()
    x = np.linspace(-1.5, 1.5, 61)
    y = np.linspace(-0.5, 0.5, 21)
    E = surf.sample([x, y, x])
    minimum, index = minimize_grid(E, (0, 1))
    jm = jump_map(index, (0, 1))
    assert jm.shape == (61, 21) and jm.dtype == np.float64
    # the minimiser's z flips from -1 (index 10) to +1 (index 50) between x = 0 and x = 0.05
    assert index[29, 10, 2] == 10 and index[30, 10, 2] == 10 and index[31, 10, 2] == 50
    np.testing.assert_array_equal(jm[30], 40.0)
    np.testing.assert_array_equal(jm[31], 40.0)
    assert (np.delete(jm, [30, 31], axis=0) == 0.0).all()
    assert minimum[30, 10] - minimum[10, 10] == pytest.approx(1.5)        # the apparent barrier b + t


def test_jump_map_handles_empty_columns():
    energies = np.array([[[3.0, 1.0, 2.0], [0.5, 4.0, 4.0]], [[2.0, 2.0, 0.0], [np.nan, np.nan, np.nan]]])
    minimum, index = minimize_grid(energies, keep=(0, 1))
    np.testing.assert_array_equal(minimum, [[1.0, 0.5], [0.0, np.nan]])
    assert index[0, 0].tolist() == [0, 0, 1] and index[1, 1].tolist() == [-1, -1, -1]
    np.testing.assert_array_equal(jump_map(index, keep=(0, 1)), [[1.0, 1.0], [1.0, np.nan]])
    with pytest.raises(ValueError):
        jump_map(index, keep=(0,))          # index has the wrong rank for one kept axis
