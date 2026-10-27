"""spDecon's fixed-K expression clustering using Scanpy and igraph."""
from __future__ import annotations

import igraph as ig
import numpy as np
from anndata import AnnData
import scanpy as sc


def louvain_labels(
    representation: np.ndarray,
    n_components: int,
    *,
    initial_neighbors: int = 20,
) -> tuple[np.ndarray, int]:
    """Return K cluster labels and the neighbor count used.

    Search resolutions in [0.5, 2.0], selecting the highest weighted
    modularity among ten Louvain starts (seeds 0 through 9) at each step.
    Expand the neighborhood if this search cannot recover K groups.
    Raise RuntimeError if the bounded search fails; never silently change K.
    Trivial one-group and one-spot-per-group partitions use no graph.
    """
    values = np.asarray(representation, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("representation must be a finite, nonempty spot-by-feature matrix")
    if n_components != int(n_components) or not 1 <= n_components <= len(values):
        raise ValueError("n_components must be an integer between 1 and the number of spots")
    if initial_neighbors != int(initial_neighbors) or initial_neighbors < 2:
        raise ValueError("initial_neighbors must be an integer of at least 2")
    k = int(n_components)
    if k == 1:
        return np.zeros(len(values), dtype=int), 0
    if k == len(values):
        return np.arange(len(values)), 0

    data = AnnData(values)
    neighbors = min(int(initial_neighbors), len(values) - 1)
    while True:
        sc.pp.neighbors(data, n_neighbors=neighbors, use_rep="X", random_state=0)
        graph = ig.Graph.Weighted_Adjacency(
            data.obsp["connectivities"], mode="undirected", attr="weight"
        )
        if graph.ecount() == 0:
            raise ValueError("The expression-neighbor graph has no edges")
        lower, upper = 0.5, 2.0
        while upper - lower > 0.01:
            resolution = float(np.sqrt(lower * upper))
            best_score, best_labels = -np.inf, None
            for seed in range(10):
                sc.tl.louvain(
                    data, resolution=resolution, random_state=seed,
                    directed=True, use_weights=False, flavor="vtraag",
                    key_added="spdecon_louvain",
                )
                candidate = np.asarray(data.obs["spdecon_louvain"], dtype=int)
                score = graph.modularity(candidate, weights="weight", resolution=resolution)
                if score > best_score:
                    best_score, best_labels = score, candidate.copy()
            if best_labels is None:
                raise RuntimeError("Louvain produced no finite modularity score")
            groups, labels = np.unique(best_labels, return_inverse=True)
            if len(groups) == k:
                return labels, neighbors
            if len(groups) < k:
                lower = resolution
            else:
                upper = resolution
        if neighbors == len(values) - 1:
            break
        neighbors = min(len(values) - 1, int(np.ceil(1.5 * neighbors)))
    raise RuntimeError(
        f"Louvain initialization could not find exactly {k} groups; "
        "check K and the expression structure of the input"
    )
