"""spDecon expression-neighborhood initialization and raw-count KL refinement.

Prevalence filtering, log-CPM, PCA, fixed-K Louvain clustering, and ten NMF
rounds provide the initial factors. All features are restored before the
raw-count refinement. This branch uses no coordinates or annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from sklearn.decomposition import NMF, PCA

from poisson_hmrf import PoissonHMRFConfig, fit_poisson_hmrf
from poisson_hmrf.independent import _row_normalize, v2_exposure_balanced_input
from rare_init_poisson.model import _full_profiles
from raw_count_poisson import FROZEN_CONFIG as RAW_FROZEN_CONFIG

from .clustering import louvain_labels


EPS = 1e-10


@dataclass
class LouvainInitPoissonFit:
    proportions: np.ndarray
    profiles: np.ndarray
    initial_proportions: np.ndarray
    initial_profiles: np.ndarray
    cluster_labels: np.ndarray
    selected_feature_indices: np.ndarray
    history: list[dict[str, object]]
    diagnostics: dict[str, object]
    config: dict[str, object]


def _raw_config(seed: int) -> PoissonHMRFConfig:
    prefix = "refinement_"
    values = {
        key.removeprefix(prefix): value
        for key, value in RAW_FROZEN_CONFIG.items()
        if key.startswith(prefix)
    }
    values["random_state"] = int(seed)
    return PoissonHMRFConfig(**values)


def _initial_factors_from_labels(
    balanced: np.ndarray,
    log_selected: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    *,
    random_state: int,
    nmf_iterations: int,
    profile_update_iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (balanced.shape[0],):
        raise ValueError("cluster labels have an incompatible shape")
    if set(labels.tolist()) != set(range(n_components)):
        raise ValueError("cluster labels must contain every component")

    hard_weights = np.full((len(labels), n_components), EPS)
    hard_weights[np.arange(len(labels)), labels] = 1.0
    cluster_profiles = np.stack(
        [log_selected[labels == k].mean(axis=0) for k in range(n_components)]
    )
    cluster_profiles = np.maximum(cluster_profiles, EPS)
    smoother = NMF(
        n_components=n_components,
        init="custom",
        solver="mu",
        beta_loss="frobenius",
        max_iter=int(nmf_iterations),
        tol=0.0,
        random_state=int(random_state),
    )
    weights = smoother.fit_transform(
        log_selected,
        W=hard_weights.copy(),
        H=cluster_profiles.copy(),
    )
    # Translate the log-domain cluster memberships into full-feature raw-rate
    # profiles.  The final optimizer never factorizes log-normalized values.
    full_profiles = _full_profiles(
        balanced,
        weights,
        iterations=int(profile_update_iterations),
    )
    component_scale = np.maximum(full_profiles.sum(axis=1), EPS)
    initial_p = _row_normalize(weights * component_scale[None, :])
    initial_b = _row_normalize(full_profiles)
    return initial_p, initial_b, {
        "cluster_sizes": np.bincount(labels, minlength=n_components).tolist(),
        "log_nmf_iterations": int(smoother.n_iter_),
        "log_nmf_reconstruction_error": float(smoother.reconstruction_err_),
    }


def fit_louvain_init_poisson(
    counts: np.ndarray,
    n_components: int,
    *,
    random_state: int = 0,
    detection_fraction: float = 0.05,
    log_scale: float = 1e4,
    n_pcs: int = 50,
    n_neighbors: int = 20,
    log_nmf_iterations: int = 10,
    profile_update_iterations: int = 75,
) -> LouvainInitPoissonFit:
    """Fit literal raw-count Poisson from a count-only Louvain basin."""

    started = time.perf_counter()
    counts = np.asarray(counts, dtype=float)
    n_components = int(n_components)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite nonnegative matrix")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("counts must not contain empty spots")
    if not 1 <= n_components <= min(counts.shape):
        raise ValueError("invalid n_components")
    if not 0 <= detection_fraction < 1 or log_scale <= 0:
        raise ValueError("invalid count preprocessing settings")

    selected = np.flatnonzero((counts > 0).mean(axis=0) > detection_fraction)
    if len(selected) < n_components:
        raise ValueError("prevalence filter retained fewer than K features")
    retained = counts[:, selected]
    log_selected = np.log1p(
        log_scale
        * retained
        / np.maximum(retained.sum(axis=1, keepdims=True), EPS)
    )
    actual_pcs = min(int(n_pcs), len(counts) - 1, len(selected))
    if actual_pcs < 1:
        raise ValueError("not enough observations/features for PCA")
    representation = PCA(
        n_components=actual_pcs,
        svd_solver="randomized",
        random_state=int(random_state),
    ).fit_transform(log_selected)
    labels, used_neighbors = louvain_labels(
        representation,
        n_components,
        initial_neighbors=n_neighbors,
    )

    initialization_started = time.perf_counter()
    balanced, _ = v2_exposure_balanced_input(counts)
    initial_p, initial_b, init_details = _initial_factors_from_labels(
        balanced,
        log_selected,
        labels,
        n_components,
        random_state=int(random_state),
        nmf_iterations=int(log_nmf_iterations),
        profile_update_iterations=int(profile_update_iterations),
    )
    initialization_runtime = time.perf_counter() - initialization_started

    refinement_started = time.perf_counter()
    refined = fit_poisson_hmrf(
        counts,
        np.zeros((counts.shape[0], 2), dtype=float),
        n_components,
        initial_p,
        initial_b,
        _raw_config(int(random_state)),
    )
    refinement_runtime = time.perf_counter() - refinement_started
    diagnostics = dict(refined.diagnostics)
    diagnostics.update(
        {
            "method": "Louvain-initialized raw-count Poisson",
            "initialization": (
                "count prevalence filter + log-CPM + PCA + spDecon "
                "Louvain K-class search + ten nonspatial NMF rounds"
            ),
            "selected_feature_count": int(len(selected)),
            "selected_feature_fraction": float(len(selected) / counts.shape[1]),
            "pca_components": int(actual_pcs),
            "louvain_neighbors": int(used_neighbors),
            **init_details,
            "initialization_runtime_sec": float(initialization_runtime),
            "refinement_runtime_sec": float(refinement_runtime),
            "standalone_runtime_sec": float(time.perf_counter() - started),
            "coordinates_used": False,
            "graph_used_for_final_model": False,
            "spatial_graph_used": False,
            "external_warm_start_used": False,
            "truth_used": False,
            "annotation_used": False,
            "other_modality_used": False,
            "initial_component_mass": initial_p.mean(axis=0).tolist(),
            "final_component_mass": refined.proportions.mean(axis=0).tolist(),
        }
    )
    config = {
        "input": "literal raw counts",
        "initialization_preprocessing": {
            "detection_fraction_strictly_above": float(detection_fraction),
            "log_transform": f"log1p({log_scale:g} * count / retained_library)",
            "pca_components": int(n_pcs),
            "initial_louvain_neighbors": int(n_neighbors),
            "louvain_random_states": 10,
            "log_domain_nonspatial_nmf_iterations": int(log_nmf_iterations),
            "full_profile_update_iterations": int(profile_update_iterations),
        },
        "final_objective": RAW_FROZEN_CONFIG["model"],
        "final_refinement": {
            key: value
            for key, value in RAW_FROZEN_CONFIG.items()
            if key.startswith("refinement_")
        },
    }
    return LouvainInitPoissonFit(
        proportions=refined.proportions,
        profiles=refined.profiles,
        initial_proportions=initial_p,
        initial_profiles=initial_b,
        cluster_labels=labels,
        selected_feature_indices=selected,
        history=refined.history,
        diagnostics=diagnostics,
        config=config,
    )
