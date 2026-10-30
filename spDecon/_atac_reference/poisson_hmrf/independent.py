"""Independent equal-spot Poisson/KL refinement.

This module deliberately does not accept a spatial solution as an
initialization.  It reconstructs the standalone v2 all-gene KL initializer
from counts and K, then refines the same v2 exposure-balanced matrix under
explicit simplex constraints and the convergence-audited count-only solver.

Because exposure balancing generally creates noninteger values, the final
objective is most precisely described as generalized KL (or Poisson
quasi-likelihood), rather than a literal raw-count Poisson likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from sklearn.decomposition import NMF

from spDecon._atac_reference.v2 import compute_exposure

from .config import PoissonHMRFConfig
from .model import EPS, PoissonHMRFFit, fit_poisson_hmrf, raw_poisson_kl_per_count


@dataclass
class IndependentBalancedPoissonKLFit:
    """Factors and audit trail for the standalone nonspatial pipeline."""

    proportions: np.ndarray
    profiles: np.ndarray
    initial_proportions: np.ndarray
    initial_profiles: np.ndarray
    history: list[dict[str, object]]
    diagnostics: dict[str, object]
    config: dict[str, object]


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return values / np.maximum(values.sum(axis=1, keepdims=True), EPS)


def v2_exposure_balanced_input(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact matrix representation factorized by standalone v2."""

    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite nonnegative matrix")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("counts contain an empty spot")
    exposure = compute_exposure(counts)
    return counts / exposure[:, None], exposure


def fit_independent_balanced_poisson_kl(
    counts: np.ndarray,
    n_components: int,
    random_state: int = 0,
    refinement_config: PoissonHMRFConfig | dict[str, Any] | None = None,
    initialization_max_iter: int = 200,
    initialization_tolerance: float = 1e-4,
) -> IndependentBalancedPoissonKLFit:
    """Fit a fully nonspatial, count-and-K-only balanced KL model.

    Stage 1 repeats the standalone v2 NNDSVDa/multiplicative-KL factorization.
    Stage 2 optimizes P and B on the *same* exposure-balanced matrix with row
    simplex constraints and projected/KKT convergence checks.  No coordinates,
    graph, Diffuse factors, labels, or truth enter either stage.
    """

    started = time.perf_counter()
    balanced, exposure = v2_exposure_balanced_input(counts)
    n_components = int(n_components)
    if not 1 <= n_components <= min(balanced.shape):
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
    if config.method != "rna_only_poisson":
        raise ValueError("independent balanced refinement must be nonspatial")
    if config.coordinate_permutation:
        raise ValueError("coordinate_permutation is invalid for a nonspatial model")

    # Coordinates are intentionally absent from the public interface.  The
    # count-only core validates a placeholder array but never constructs or
    # evaluates a graph for rna_only_poisson.
    placeholder_coordinates = np.zeros((balanced.shape[0], 2), dtype=float)
    refined: PoissonHMRFFit = fit_poisson_hmrf(
        balanced,
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
            "method": "independent_v2_balanced_poisson_kl",
            "model": (
                "generalized-KL refinement of v2 exposure-balanced values; "
                "equal-spot Poisson quasi-likelihood"
            ),
            "input_representation": "counts / (library / median_positive_library)",
            "literal_raw_count_poisson": False,
            "spatial_input_used": False,
            "coordinates_used": False,
            "graph_used": False,
            "external_warm_start_used": False,
            "initialization": (
                "internal counts-and-K-only NNDSVDa multiplicative generalized-KL NMF"
            ),
            "initialization_nmf_iterations": int(initializer.n_iter_),
            "initialization_nmf_reconstruction_error": float(
                initializer.reconstruction_err_
            ),
            "initialization_runtime_sec": float(initialization_runtime),
            "refinement_runtime_sec": float(refined.diagnostics["runtime_sec"]),
            "standalone_runtime_sec": float(total_runtime),
            "balanced_row_sum_mean": float(balanced.sum(axis=1).mean()),
            "balanced_row_sum_max_relative_deviation": float(
                np.max(
                    np.abs(balanced.sum(axis=1) - balanced.sum(axis=1).mean())
                    / max(float(balanced.sum(axis=1).mean()), EPS)
                )
            ),
            "raw_relative_exposure_min": float(exposure.min()),
            "raw_relative_exposure_max": float(exposure.max()),
            "initial_balanced_kl_per_mass": float(
                raw_poisson_kl_per_count(balanced, initial_p, initial_b)
            ),
            "final_balanced_kl_per_mass": float(
                raw_poisson_kl_per_count(
                    balanced, refined.proportions, refined.profiles
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
        "input_mode": "v2_exposure_balanced",
        "initialization": {
            "init": "nndsvda",
            "solver": "mu",
            "beta_loss": "kullback-leibler",
            "max_iter": int(initialization_max_iter),
            "tolerance": float(initialization_tolerance),
            "random_state": int(random_state),
        },
        "refinement": refined.config,
    }
    return IndependentBalancedPoissonKLFit(
        proportions=refined.proportions,
        profiles=refined.profiles,
        initial_proportions=initial_p,
        initial_profiles=initial_b,
        history=refined.history,
        diagnostics=diagnostics,
        config=exported_config,
    )
