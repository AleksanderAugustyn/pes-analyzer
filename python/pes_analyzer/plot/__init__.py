"""pes_analyzer.plot: matplotlib helpers for maps, paths, profiles and merge trees.

Imported explicitly (``import pes_analyzer.plot``); ``import pes_analyzer``
stays matplotlib-free. Every helper draws on a caller-supplied ``ax``
(``matplotlib.pyplot.gca()`` when ``None``) and returns it; figures, titles,
saving and backends are the caller's. See ``_docs/API.md`` section ``plot``.
"""
from __future__ import annotations

from ._map import plot_map, plot_path
from ._tree import MergeTreeLayout, merge_tree_layout

__all__ = ["MergeTreeLayout", "merge_tree_layout", "plot_map", "plot_path"]
