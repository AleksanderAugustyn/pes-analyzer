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
