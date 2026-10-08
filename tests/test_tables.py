"""Tables of minima, basins and paths, and the CSV writer and reader (pure Python; no rebuild)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pes_analyzer.grid import path_length
from pes_analyzer.tables import basins_table, minima_table, path_table
from pes_analyzer.topology import MergeTree, Watershed, find_watershed_segmentation

AXES = {"x": np.array([0.0, 1.0, 3.0]), "y": np.array([10.0, 20.0, 40.0, 80.0])}


def _row_grid():
    """Three basins along one row: minima at columns 0, 4, 8 (the API.md toy)."""
    return np.array([[0.0, 3.0, 8.0, 4.0, 1.0, 3.0, 5.0, 4.0, 2.0]])


def _forest():
    basins = [((0, 0), 0.0), ((0, 3), 1.0), ((1, 0), 0.5), ((1, 3), 2.0)]
    merges = [((0, 1), 3.0, 0, 1), ((1, 1), 4.0, 2, 3)]
    return Watershed(labels=None, basins=basins, merges=merges, neighborhood="von_neumann",
                     parents=None, merge_table=None, dtype=np.dtype("float64"), fingerprint=b"\x00" * 16)


def test_minima_table_columns_and_order():
    t = minima_table([((0, 1), 2.5), ((2, 3), -1.0)], AXES)
    assert list(t) == ["index_0", "index_1", "x", "y", "energy"]
    assert t["index_0"].dtype == np.int64 and t["x"].dtype == np.float64
    assert t["index_1"].tolist() == [1, 3] and t["x"].tolist() == [0.0, 3.0] and t["y"].tolist() == [20.0, 80.0]
    assert t["energy"].tolist() == [2.5, -1.0]
    seq = minima_table([((0, 1), 2.5)], [AXES["x"], AXES["y"]])
    assert list(seq) == ["index_0", "index_1", "x0", "x1", "energy"]
    bare = minima_table([((0, 1), np.float32(2.5))])
    assert list(bare) == ["index_0", "index_1", "energy"] and bare["energy"].dtype == np.float64


def test_minima_table_empty_needs_a_dimension():
    with pytest.raises(ValueError):
        minima_table([])
    t = minima_table([], AXES)
    assert list(t) == ["index_0", "index_1", "x", "y", "energy"] and all(c.shape == (0,) for c in t.values())
    t = minima_table([], ndim=3)
    assert list(t) == ["index_0", "index_1", "index_2", "energy"]
    with pytest.raises(ValueError):
        minima_table([((0, 1), 2.5)], ndim=3)


def test_path_table_matches_path_length():
    idx = np.array([[0, 0], [1, 1], [2, 3]])
    t = path_table(idx, [1.0, 2.0, 0.5], AXES)
    assert list(t) == ["step", "index_0", "index_1", "x", "y", "length", "energy"]
    assert t["step"].tolist() == [0, 1, 2] and t["step"].dtype == np.int64
    np.testing.assert_allclose(t["length"], path_length(idx, AXES))
    np.testing.assert_allclose(path_table(idx, [1.0, 2.0, 0.5])["length"], path_length(idx))
    with pytest.raises(ValueError):
        path_table(idx, [1.0, 2.0])
    with pytest.raises(ValueError):
        path_table(idx.astype(float), [1.0, 2.0, 0.5])


def test_basins_table_matches_the_tree():
    E = _row_grid()
    tree = MergeTree(find_watershed_segmentation(E))
    axes = {"r": np.array([0.0]), "c": np.arange(9, dtype=float) * 0.5}
    t = basins_table(tree, axes)
    assert list(t) == ["basin", "min_index_0", "min_index_1", "min_r", "min_c", "min_energy", "parent",
                       "saddle_index_0", "saddle_index_1", "saddle_r", "saddle_c", "saddle_energy", "persistence"]
    assert t["basin"].tolist() == [0, 1, 2]
    assert t["min_index_1"].tolist() == [0, 4, 8] and t["min_c"].tolist() == [0.0, 2.0, 4.0]
    assert t["min_energy"].tolist() == [0.0, 1.0, 2.0]
    assert t["parent"].tolist() == [-1, 0, 1]
    assert t["saddle_index_1"].tolist() == [-1, 2, 6] and t["saddle_index_0"].tolist() == [-1, 0, 0]
    assert math.isnan(t["saddle_c"][0]) and t["saddle_c"][1:].tolist() == [1.0, 3.0]
    assert math.isnan(t["saddle_energy"][0]) and t["saddle_energy"][1:].tolist() == [8.0, 5.0]
    assert math.isinf(t["persistence"][0]) and t["persistence"][1:].tolist() == [7.0, 3.0]
    pruned = basins_table(tree, min_persistence=4.0)
    assert pruned["basin"].tolist() == [0, 1] and list(pruned)[1] == "min_index_0"
    for node_id in (0, 1, 2):
        node = tree.node(node_id)
        assert t["min_energy"][node_id] == node.minimum_energy and t["persistence"][node_id] == node.persistence


def test_basins_table_accepts_watershed_and_gives_every_root_sentinels():
    t = basins_table(_forest())
    assert t["parent"].tolist() == [-1, 0, -1, 2]
    assert t["saddle_index_0"].tolist() == [-1, 0, -1, 1]
    assert [math.isnan(v) for v in t["saddle_energy"]] == [True, False, True, False]
    assert [math.isinf(v) for v in t["persistence"]] == [True, False, True, False]


def test_tables_reject_colliding_axis_names():
    bad = {"energy": np.array([0.0, 1.0, 3.0]), "y": np.array([10.0, 20.0, 40.0, 80.0])}
    with pytest.raises(ValueError, match="energy"):
        minima_table([((0, 1), 2.5)], bad)
    with pytest.raises(ValueError, match="step"):
        path_table(np.array([[0, 0]]), [1.0], {"step": bad["energy"], "y": bad["y"]})
    tree = MergeTree(find_watershed_segmentation(_row_grid()))
    with pytest.raises(ValueError, match="min_energy"):
        basins_table(tree, {"r": np.array([0.0]), "energy": np.arange(9, dtype=float)})


from pes_analyzer.tables import read_csv, write_csv


def _roundtrip(tmp_path, table):
    write_csv(tmp_path / "t.csv", table)
    return read_csv(tmp_path / "t.csv")


def test_csv_round_trip_values_and_dtypes(tmp_path):
    table = {
        "i": np.array([3, -1, 0], dtype=np.int64),
        "b": np.array([True, False, True]),
        "f": np.array([0.1, np.nan, np.inf]),
        "g": np.array([-np.inf, 1e-300, 123456789.123456789]),
        "h": np.array([1.0, 2.0, 3.0]),
    }
    back = _roundtrip(tmp_path, table)
    assert list(back) == list(table)
    assert back["i"].dtype == np.int64 and back["i"].tolist() == [3, -1, 0]
    assert back["b"].dtype == np.int64 and back["b"].tolist() == [1, 0, 1]
    for name in ("f", "g", "h"):
        assert back[name].dtype == np.float64
        np.testing.assert_array_equal(back[name], table[name])
    text = (tmp_path / "t.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "i,b,f,g,h"
    assert text.splitlines()[1] == "3,1,0.1,-inf,1.0"
    assert "\r" not in text


def test_csv_float32_is_exact_and_empty_tables_come_back_float64(tmp_path):
    e32 = np.array([0.1, 2.5], dtype=np.float32)
    back = _roundtrip(tmp_path, {"energy": e32})
    np.testing.assert_array_equal(back["energy"], e32.astype(np.float64))
    empty = minima_table([], ndim=2)
    back = _roundtrip(tmp_path, empty)
    assert list(back) == ["index_0", "index_1", "energy"]
    assert all(c.shape == (0,) and c.dtype == np.float64 for c in back.values())


def test_write_csv_rejects_non_numeric_and_ragged_columns(tmp_path):
    with pytest.raises(ValueError, match="dtype"):
        write_csv(tmp_path / "t.csv", {"name": np.array(["a", "b"])})
    with pytest.raises(ValueError, match="rows"):
        write_csv(tmp_path / "t.csv", {"a": np.array([1, 2]), "b": np.array([1.0])})
    with pytest.raises(ValueError, match="1-D"):
        write_csv(tmp_path / "t.csv", {"a": np.zeros((2, 2))})
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_csv(tmp_path / "empty.csv")


def test_csv_round_trip_of_the_three_tables(tmp_path):
    tree = MergeTree(find_watershed_segmentation(_row_grid()))
    axes = {"r": np.array([0.0]), "c": np.arange(9, dtype=float) * 0.5}
    idx = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]])
    for table in (minima_table(tree.ws.basins, axes), basins_table(tree, axes), path_table(idx, _row_grid()[0, :5], axes)):
        back = _roundtrip(tmp_path, table)
        assert list(back) == list(table)
        for name in table:
            assert back[name].dtype == table[name].dtype
            np.testing.assert_array_equal(back[name], table[name])
