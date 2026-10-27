"""Independent raw-count Poisson refinement with a data-only v2 initializer."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from sklearn.decomposition import NMF

from .config import PoissonHMRFConfig
from .independent import _row_normalize, v2_exposure_balanced_input
from .model import EPS, PoissonHMRFFit, fit_poisson_hmrf, raw_poisson_kl_per_count


@dataclass
class IndependentRawPoissonFit:
    """Factors and audit trail for a standalone raw-count Poisson pipeline."""

    proportions: np.ndarray
    profiles: np.ndarray
    initial_proportions: np.ndarray
    initial_profiles: np.ndarray
    history: list[dict[str, object]]
    diagnostics: dict[str, object]
    config: dict[str, object]


def fit_independent_raw_poisson(
    counts: np.ndarray,
    n_components: int,
    random_state: int = 0,
    refinement_config: PoissonHMRFConfig | dict[str, Any] | None = None,
    initialization_max_iter: int = 200,
    initialization_tolerance: float = 1e-4,
) -> IndependentRawPoissonFit:
    """Fit raw-count Poisson/KL using only counts and K.

    The internal initializer is the standalone v2 NNDSVDa generalized-KL NMF
    on exposure-balanced counts.  The final objective switches to raw counts,
    with raw library size as exposure.  No spatial result is accepted or used.
    """

    started = time.perf_counter()
    counts = np.asarray(counts, dtype=float)
    balanced, relative_exposure = v2_exposure_balanced_input(counts)
    n_components = int(n_components)
    if not 1 <= n_components <= min(counts.shape):
        raise ValueError("n_components must be between 1 and min(counts.shape)")
    if initialization_max_iter < 1 or initialization_tolerance <= 0:
        raise ValueError("initialization convergence controls are invalid")

    initialization_started = time.perf_counter()
    initializer = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=int(initialization_max_iter),
        tol=float(initialization_tolerance),
        random_state=int(random_state),
    )
    weights = initializer.fit_transform(balanced)
    components = np.maximum(initializer.components_, 0.0)
    component_scale = np.maximum(components.sum(axis=1), EPS)
    initial_p = _row_normalize(weights * component_scale[None, :])
    initial_b = _row_normalize(components)
    initialization_runtime = time.perf_counter() - initialization_started

    if refinement_config is None:
        config = PoissonHMRFConfig(
            method="rna_only_poisson", random_state=int(random_state)
        )
    elif isinstance(refinement_config, dict):
        config = PoissonHMRFConfig(**refinement_config)
    else:
        config = refinement_config
    if config.method != "rna_only_poisson" or config.coordinate_permutation:
        raise ValueError("independent raw Poisson refinement must be nonspatial")

    placeholder_coordinates = np.zeros((counts.shape[0], 2), dtype=float)
    refined: PoissonHMRFFit = fit_poisson_hmrf(
        counts,
        placeholder_coordinates,
        n_components,
        initial_p,
        initial_b,
        config,
    )
    total_runtime = time.perf_counter() - started
    diagnostics = dict(refined.diagnostics)
    diagnostics.update(
        {
            "method": "independent_raw_count_poisson",
            "model": "Y_ig ~ Poisson(library_i * (P B)_ig); no spatial term",
            "input_representation": "raw counts",
            "literal_raw_count_poisson": True,
            "spatial_input_used": False,
            "coordinates_used": False,
            "graph_used": False,
            "external_warm_start_used": False,
            "initialization": (
                "internal counts-and-K-only v2 exposure-balanced NNDSVDa "
                "multiplicative generalized-KL NMF"
            ),
            "initialization_nmf_iterations": int(initializer.n_iter_),
            "initialization_nmf_reconstruction_error": float(
                initializer.reconstruction_err_
            ),
            "initialization_runtime_sec": float(initialization_runtime),
            "refinement_runtime_sec": float(refined.diagnostics["runtime_sec"]),
            "standalone_runtime_sec": float(total_runtime),
            "raw_relative_exposure_min": float(relative_exposure.min()),
            "raw_relative_exposure_max": float(relative_exposure.max()),
            "initial_raw_poisson_kl_per_count": float(
                raw_poisson_kl_per_count(counts, initial_p, initial_b)
            ),
            "final_raw_poisson_kl_per_count": float(
                raw_poisson_kl_per_count(
                    counts, refined.proportions, refined.profiles
                )
            ),
            "proportion_refinement_rms_change": float(
                np.sqrt(np.mean(np.square(refined.proportions - initial_p)))
            ),
            "profile_refinement_rms_change": float(
                np.sqrt(np.mean(np.square(refined.profiles - initial_b)))
            ),
        }
    )
    exported_config = {
        "input_mode": "raw_counts",
        "initialization": {
            "input_mode": "v2_exposure_balanced",
            "init": "nndsvda",
            "solver": "mu",
            "beta_loss": "kullback-leibler",
            "max_iter": int(initialization_max_iter),
            "tolerance": float(initialization_tolerance),
            "random_state": int(random_state),
        },
        "refinement": refined.config,
    }
    return IndependentRawPoissonFit(
        proportions=refined.proportions,
        profiles=refined.profiles,
        initial_proportions=initial_p,
        initial_profiles=initial_b,
        history=refined.history,
        diagnostics=diagnostics,
        config=exported_config,
    )
