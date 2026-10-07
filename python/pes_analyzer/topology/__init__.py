"""pes_analyzer.topology: watershed segmentation and merge-tree analysis.

The Rust-backed kernels are wrapped in ``_flood.py`` (``Watershed``,
``find_watershed_segmentation``, ``find_minimax_path``); the
pure-Python analysis layer provides
``compute_persistence`` / ``prune_merge_tree`` (in ``_tree.py``) and the
traversable ``MergeTree`` (in ``merge_tree.py``).
"""

from __future__ import annotations

from ._flood import (
    Watershed,
    energy_fingerprint,
    find_minimax_path,
    find_watershed_segmentation,
)
from ._metric_paths import (
    find_least_action_path,
    find_minimum_ascent_path,
    find_steepest_descent_path,
)
from ._path import PathProfile, analyze_path_profile
from ._tree import (
    compute_persistence,
    prune_merge_tree,
)
from .merge_tree import BasinNode, MergeTree

__all__ = [
    "Watershed",
    "energy_fingerprint",
    "find_watershed_segmentation",
    "find_minimax_path",
    "find_steepest_descent_path",
    "find_least_action_path",
    "find_minimum_ascent_path",
    "PathProfile",
    "analyze_path_profile",
    "compute_persistence",
    "prune_merge_tree",
    "MergeTree",
    "BasinNode",
]
