"""Shared return type for the original count-only initialization routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InitialFactors:
    """Full-gene initial factors, before any Poisson refinement.

    ``proportions`` has shape (observations, components), and ``profiles``
    has shape (components, all input genes).  Both are row-normalized.  The
    optional feature indices describe the subset used to construct the
    initializer, not the gene axis of the returned full-gene profiles.
    """

    proportions: np.ndarray
    profiles: np.ndarray
    diagnostics: dict[str, object]
    selected_feature_indices: np.ndarray | None = None
    feature_scores: np.ndarray | None = None
    cluster_labels: np.ndarray | None = None
