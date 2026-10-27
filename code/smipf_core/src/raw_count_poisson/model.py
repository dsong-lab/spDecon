"""Frozen public entry point for raw-count Poisson v1.0.0.

The research implementation remains in :mod:`poisson_hmrf`.  This module is
the stable method contract: callers can supply only counts, K, and a numerical
seed.  Every initialization and refinement hyperparameter is pinned here.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from poisson_hmrf import (
    IndependentRawPoissonFit,
    PoissonHMRFConfig,
    fit_independent_raw_poisson,
)


METHOD_NAME = "raw-count Poisson"
METHOD_VERSION = "1.0.0"

_INITIALIZATION = {
    "input": "counts/(library_size/median_positive_library_size)",
    "init": "nndsvda",
    "solver": "mu",
    "beta_loss": "kullback-leibler",
    "max_iter": 200,
    "tolerance": 1e-4,
}

_REFINEMENT = {
    "method": "rna_only_poisson",
    "max_outer_iterations": 500,
    "max_p_steps": 100,
    "max_b_steps": 100,
    "kkt_check_interval": 25,
    "minimum_outer_iterations": 3,
    "convergence_patience": 5,
    "relative_tolerance": 1e-6,
    "p_kkt_tolerance": 1e-4,
    "b_kkt_tolerance": 1e-4,
    "initial_projected_change": 0.05,
    "armijo": 1e-4,
    "max_backtracks": 40,
    "minimum_step": 1e-20,
    "maximum_step": 1e8,
    "lower_bound": 1e-8,
    "device": "cuda",
    "dtype": "float32",
    "verbose": False,
}

FROZEN_CONFIG = MappingProxyType(
    {
        "model": "Y_ig ~ Poisson(library_i * (P B)_ig)",
        "input": "literal_raw_counts",
        "exposure": "observed_raw_library_size",
        "P_constraint": "row_simplex_with_1e-8_floor",
        "B_constraint": "row_simplex_with_1e-8_floor",
        "spatial_term": "none",
        "gene_regularization": "none",
        "external_warm_start": "forbidden",
        **{f"initialization_{key}": value for key, value in _INITIALIZATION.items()},
        **{f"refinement_{key}": value for key, value in _REFINEMENT.items()},
    }
)


def fit_raw_count_poisson(
    counts: np.ndarray,
    n_components: int,
    *,
    random_state: int = 0,
) -> IndependentRawPoissonFit:
    """Fit frozen raw-count Poisson v1.0.0.

    No optimization, normalization, spatial, or initialization switch is
    exposed.  CUDA is requested for the frozen float32 refinement and the
    shared numerical core falls back to CPU when CUDA is unavailable.
    """

    config = PoissonHMRFConfig(
        **{**_REFINEMENT, "random_state": int(random_state)}
    )
    fit = fit_independent_raw_poisson(
        counts,
        int(n_components),
        random_state=int(random_state),
        refinement_config=config,
        initialization_max_iter=int(_INITIALIZATION["max_iter"]),
        initialization_tolerance=float(_INITIALIZATION["tolerance"]),
    )
    fit.diagnostics.update(
        {
            "frozen_method_name": METHOD_NAME,
            "frozen_method_version": METHOD_VERSION,
            "frozen_config": dict(FROZEN_CONFIG),
        }
    )
    fit.config = {
        **fit.config,
        "frozen_method_name": METHOD_NAME,
        "frozen_method_version": METHOD_VERSION,
        "frozen_config": dict(FROZEN_CONFIG),
    }
    return fit
