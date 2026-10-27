"""Configuration for the optional spGSNMF-v3 dual-geometry experiment.

The defaults deliberately leave the frozen v2 implementation untouched.  A
caller must select one of the four named experimental methods explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


METHODS = (
    "baseline_v2",
    "rna_refined_control",
    "laplacian_v3",
    "huber_tv_v3",
    "affinity_v3",
)
NULL_MODELS = (
    "none",
    "coordinate_permutation",
    "degree_preserving",
    "edge_label_permutation",
)
PROFILE_SOLVERS = ("diagonal_newton", "projected_gradient", "lbfgs_logits")
GRADIENT_RATIOS = (0.01, 0.05, 0.10, 0.25)

# This is a frozen *RNA* benchmark roster.  It intentionally excludes the
# later ATAC development datasets that coexist in the official benchmark
# configuration.
FROZEN_BENCHMARK_DATASETS = (
    "MPOA",
    "Kidney",
    "Mammary",
    "MOB",
    "Cerebellum",
    "RETROFIT_simulation",
    "Fetal_intestine",
)


@dataclass(frozen=True)
class DualGeometryConfig:
    """Serializable controls shared by M0--M3 and the spatial nulls."""

    # Safe opt-in contract: constructing the experimental estimator without
    # an explicit method reproduces frozen v2 rather than enabling a spatial
    # objective implicitly.
    method: str = "baseline_v2"
    gradient_ratio: float = 0.05
    n_neighbors: int = 6
    local_radius_multiplier: float = 3.0
    local_scale_neighbor_rank: int = 7
    negative_ratio: float = 1.0
    huber_delta: float = 0.02
    affinity_ridge: float = 0.1
    affinity_prefit_max_iter: int = 100
    affinity_block_max_iter: int = 25
    affinity_update_interval: int = 5
    affinity_optimizer_gradient_tolerance: float = 1e-7
    affinity_optimizer_function_tolerance: float = 1e-12
    max_iter: int = 75
    minimum_iterations: int = 10
    tolerance: float = 1e-6
    patience: int = 5
    stationarity_fraction: float = 0.05
    p_kkt_absolute_tolerance: float = 1e-6
    b_kkt_absolute_tolerance: float = 1e-8
    affinity_gradient_tolerance: float = 1e-5
    mirror_initial_log_step: float = 0.05
    projected_gradient_initial_change: float = 0.05
    projected_gradient_step_growth: float = 1.5
    projected_gradient_min_step: float = 1e-10
    projected_gradient_max_step: float = 1e10
    projected_gradient_armijo: float = 1e-4
    projected_gradient_max_backtracks: int = 35
    proportion_pg_inner_steps: int = 3
    profile_pg_inner_steps: int = 3
    profile_solver: str = "projected_gradient"
    profile_pg_post_inner_steps: int = 20
    profile_diagonal_newton_max_iter: int = 20
    profile_curvature_floor: float = 1e-8
    profile_block_stationarity_fraction: float = 0.05
    profile_lbfgs_max_iter: int = 25
    profile_lbfgs_gradient_tolerance: float = 1e-9
    profile_lbfgs_function_tolerance: float = 1e-12
    profile_logit_seed_mass: float = 1e-8
    inner_stationarity_fraction: float = 0.5
    p_kkt_max_absolute_tolerance: float = 1e-5
    b_kkt_max_absolute_tolerance: float = 1e-5
    # Retained only so old experimental config files remain parseable.  The
    # v3 optimizer no longer uses multiplicative profile updates.
    profile_mu_inner_steps: int = 3
    random_state: int = 0
    v2_max_iter: int = 200
    v2_tolerance: float = 1e-4
    rna_sacrifice_cap: float = 0.02
    null_model: str = "none"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "DualGeometryConfig":
        """Raise a clear error for settings that violate the experiment."""

        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.null_model not in NULL_MODELS:
            raise ValueError(f"null_model must be one of {NULL_MODELS}")
        if self.profile_solver not in PROFILE_SOLVERS:
            raise ValueError(f"profile_solver must be one of {PROFILE_SOLVERS}")
        if not math.isfinite(float(self.gradient_ratio)) or not (
            0.0 <= float(self.gradient_ratio) <= 1.0
        ):
            raise ValueError("gradient_ratio must lie in [0, 1]")
        if int(self.n_neighbors) != self.n_neighbors or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if (
            int(self.local_scale_neighbor_rank) != self.local_scale_neighbor_rank
            or self.local_scale_neighbor_rank < 1
        ):
            raise ValueError("local_scale_neighbor_rank must be a positive integer")
        if not math.isfinite(float(self.local_radius_multiplier)) or not (
            1.0 < float(self.local_radius_multiplier) <= 10.0
        ):
            raise ValueError("local_radius_multiplier must lie in (1, 10]")
        if not math.isfinite(float(self.negative_ratio)) or not (
            0.0 < float(self.negative_ratio) <= 10.0
        ):
            raise ValueError("negative_ratio must lie in (0, 10]")
        if not math.isfinite(float(self.huber_delta)) or float(self.huber_delta) <= 0.0:
            raise ValueError("huber_delta must be finite and positive")
        if not math.isfinite(float(self.affinity_ridge)) or float(self.affinity_ridge) <= 0.0:
            raise ValueError("affinity_ridge must be finite and positive")
        for name in (
            "affinity_prefit_max_iter",
            "affinity_block_max_iter",
            "affinity_update_interval",
            "max_iter",
            "minimum_iterations",
            "patience",
            "projected_gradient_max_backtracks",
            "proportion_pg_inner_steps",
            "profile_pg_inner_steps",
            "profile_pg_post_inner_steps",
            "profile_lbfgs_max_iter",
            "profile_diagonal_newton_max_iter",
            "profile_mu_inner_steps",
            "v2_max_iter",
        ):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "tolerance",
            "v2_tolerance",
            "p_kkt_absolute_tolerance",
            "b_kkt_absolute_tolerance",
            "p_kkt_max_absolute_tolerance",
            "b_kkt_max_absolute_tolerance",
            "affinity_gradient_tolerance",
            "affinity_optimizer_gradient_tolerance",
            "affinity_optimizer_function_tolerance",
            "projected_gradient_initial_change",
            "projected_gradient_step_growth",
            "projected_gradient_min_step",
            "projected_gradient_max_step",
            "projected_gradient_armijo",
            "profile_lbfgs_gradient_tolerance",
            "profile_lbfgs_function_tolerance",
            "profile_logit_seed_mass",
            "profile_curvature_floor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_iterations > self.max_iter:
            raise ValueError("minimum_iterations must not exceed max_iter")
        if not math.isfinite(float(self.stationarity_fraction)) or not (
            0.0 < float(self.stationarity_fraction) < 1.0
        ):
            raise ValueError("stationarity_fraction must lie in (0, 1)")
        if not math.isfinite(float(self.inner_stationarity_fraction)) or not (
            0.0 < float(self.inner_stationarity_fraction) < 1.0
        ):
            raise ValueError("inner_stationarity_fraction must lie in (0, 1)")
        if not math.isfinite(float(self.mirror_initial_log_step)) or not (
            0.0 < float(self.mirror_initial_log_step) <= 1.0
        ):
            raise ValueError("mirror_initial_log_step must lie in (0, 1]")
        if not 0.0 < float(self.projected_gradient_initial_change) <= 1.0:
            raise ValueError("projected_gradient_initial_change must lie in (0, 1]")
        if float(self.projected_gradient_step_growth) < 1.0:
            raise ValueError("projected_gradient_step_growth must be at least 1")
        if float(self.projected_gradient_min_step) >= float(
            self.projected_gradient_max_step
        ):
            raise ValueError(
                "projected_gradient_min_step must be smaller than "
                "projected_gradient_max_step"
            )
        if float(self.projected_gradient_armijo) >= 1.0:
            raise ValueError("projected_gradient_armijo must be smaller than 1")
        if float(self.profile_logit_seed_mass) >= 1.0:
            raise ValueError("profile_logit_seed_mass must be smaller than 1")
        if not math.isfinite(float(self.profile_block_stationarity_fraction)) or not (
            0.0 < float(self.profile_block_stationarity_fraction) < 1.0
        ):
            raise ValueError("profile_block_stationarity_fraction must lie in (0, 1)")
        if not math.isfinite(float(self.rna_sacrifice_cap)) or not (
            0.0 <= float(self.rna_sacrifice_cap) <= 1.0
        ):
            raise ValueError("rna_sacrifice_cap must lie in [0, 1]")
        if int(self.random_state) != self.random_state or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        if self.method != "affinity_v3" and self.null_model == "edge_label_permutation":
            raise ValueError("edge_label_permutation is defined only for affinity_v3")
        if self.method == "rna_refined_control" and self.null_model != "none":
            raise ValueError("rna_refined_control does not use a spatial null")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly hyperparameters for result provenance."""

        return asdict(self)


# Friendly aliases make the config convenient for both package users and the
# benchmark runner without creating multiple, drifting schemas.
ModelConfig = DualGeometryConfig
SpGSNMFV3Config = DualGeometryConfig


def method_configs(**overrides: Any) -> dict[str, DualGeometryConfig]:
    """Construct the matched M0/M0R/M1--M3 configurations."""

    unknown = set(overrides).difference(DualGeometryConfig.__dataclass_fields__)
    if unknown:
        raise TypeError(f"unknown configuration fields: {sorted(unknown)}")
    return {
        method: DualGeometryConfig(method=method, **overrides)
        for method in METHODS
    }


__all__ = [
    "DualGeometryConfig",
    "FROZEN_BENCHMARK_DATASETS",
    "GRADIENT_RATIOS",
    "METHODS",
    "ModelConfig",
    "NULL_MODELS",
    "PROFILE_SOLVERS",
    "SpGSNMFV3Config",
    "method_configs",
]
