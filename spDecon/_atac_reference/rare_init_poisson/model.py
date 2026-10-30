"""Poisson-residual feature initialization followed by raw Poisson refinement."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from sklearn.decomposition import NMF

from spDecon._atac_reference.poisson_hmrf import PoissonHMRFConfig, fit_poisson_hmrf
from spDecon._atac_reference.poisson_hmrf.independent import _row_normalize, v2_exposure_balanced_input
from spDecon._atac_reference.poisson_hmrf.initialization import InitialFactors
from spDecon._atac_reference.raw_count_poisson import FROZEN_CONFIG as RAW_FROZEN_CONFIG
from spDecon._atac_reference.v2.model import exposure_null_residuals


EPS = 1e-10


@dataclass
class RareInitPoissonFit:
    proportions: np.ndarray
    profiles: np.ndarray
    initial_proportions: np.ndarray
    initial_profiles: np.ndarray
    selected_feature_indices: np.ndarray
    feature_scores: np.ndarray
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


def _full_profiles(
    balanced: np.ndarray,
    weights: np.ndarray,
    *,
    iterations: int,
) -> np.ndarray:
    gram = weights.T @ weights
    ridge = 1e-8 * max(float(np.trace(gram)) / max(len(gram), 1), 1.0)
    profiles = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]), weights.T @ balanced
    )
    profiles = np.maximum(profiles, EPS)
    denominator = np.maximum(weights.sum(axis=0)[:, None], EPS)
    for _ in range(int(iterations)):
        fitted = np.maximum(weights @ profiles, EPS)
        profiles *= (weights.T @ (balanced / fitted)) / denominator
        profiles = np.maximum(profiles, EPS)
    return profiles


def initialize_rare_poisson(
    counts: np.ndarray,
    n_components: int,
    *,
    n_initialization_features: int,
    random_state: int = 0,
    profile_update_iterations: int = 75,
) -> InitialFactors:
    """Construct the original residual-selected initial P and full-gene B.

    Feature ranking and initialization use only the current count matrix.
    Selected features choose the initial spot factors; all features are restored
    before returning.  This routine does not run Poisson refinement.
    """

    counts = np.asarray(counts, dtype=float)
    n_components = int(n_components)
    n_initialization_features = int(n_initialization_features)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite nonnegative matrix")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("counts must not contain empty spots")
    if not 1 <= n_components <= min(counts.shape):
        raise ValueError("invalid n_components")
    if not n_components <= n_initialization_features <= counts.shape[1]:
        raise ValueError("invalid n_initialization_features")

    balanced, exposure = v2_exposure_balanced_input(counts)
    residuals = exposure_null_residuals(counts, exposure)
    feature_scores = residuals.var(axis=0)
    selected = np.argsort(-feature_scores, kind="mergesort")[
        :n_initialization_features
    ]

    initialization_started = time.perf_counter()
    initializer = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=200,
        tol=1e-4,
        random_state=int(random_state),
    )
    weights = initializer.fit_transform(balanced[:, selected])
    full_profiles = _full_profiles(
        balanced,
        weights,
        iterations=int(profile_update_iterations),
    )
    component_scale = np.maximum(full_profiles.sum(axis=1), EPS)
    initial_p = _row_normalize(weights * component_scale[None, :])
    initial_b = _row_normalize(full_profiles)
    initialization_runtime = time.perf_counter() - initialization_started
    return InitialFactors(
        proportions=initial_p,
        profiles=initial_b,
        selected_feature_indices=selected,
        feature_scores=feature_scores,
        diagnostics={
            "n_initialization_features": int(n_initialization_features),
            "feature_fraction": float(n_initialization_features / counts.shape[1]),
            "initializer_nmf_iterations": int(initializer.n_iter_),
            "initializer_reconstruction_error": float(
                initializer.reconstruction_err_
            ),
            "full_profile_update_iterations": int(profile_update_iterations),
            "initialization_runtime_sec": float(initialization_runtime),
        },
    )


def fit_rare_init_poisson(
    counts: np.ndarray,
    n_components: int,
    *,
    n_initialization_features: int,
    random_state: int = 0,
    profile_update_iterations: int = 75,
) -> RareInitPoissonFit:
    """Fit raw Poisson from a high-Poisson-residual-variance feature basin.

    Feature ranking and initialization use only the current count matrix.
    Selected features choose the initial spot factors; all features are restored
    before the final literal raw-count Poisson refinement.
    """

    started = time.perf_counter()
    counts = np.asarray(counts, dtype=float)
    initial = initialize_rare_poisson(
        counts,
        n_components,
        n_initialization_features=n_initialization_features,
        random_state=random_state,
        profile_update_iterations=profile_update_iterations,
    )
    n_components = int(n_components)
    n_initialization_features = int(n_initialization_features)
    initial_p, initial_b = initial.proportions, initial.profiles

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
            "method": "rare-aware initialized raw-count Poisson",
            "initialization": "top exposure-null Pearson-residual-variance features",
            **initial.diagnostics,
            "refinement_runtime_sec": float(refinement_runtime),
            "standalone_runtime_sec": float(time.perf_counter() - started),
            "coordinates_used": False,
            "graph_used": False,
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
        "feature_score": "variance of clipped exposure-null Pearson residuals",
        "n_initialization_features": int(n_initialization_features),
        "initializer": "NNDSVDa multiplicative generalized-KL NMF",
        "full_profile_update_iterations": int(profile_update_iterations),
        "final_objective": RAW_FROZEN_CONFIG["model"],
        "final_refinement": {
            key: value
            for key, value in RAW_FROZEN_CONFIG.items()
            if key.startswith("refinement_")
        },
    }
    return RareInitPoissonFit(
        proportions=refined.proportions,
        profiles=refined.profiles,
        initial_proportions=initial_p,
        initial_profiles=initial_b,
        selected_feature_indices=initial.selected_feature_indices,
        feature_scores=initial.feature_scores,
        history=refined.history,
        diagnostics=diagnostics,
        config=config,
    )
