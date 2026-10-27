"""Configuration for the opt-in Poisson-HMRF spDecon experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PoissonHMRFConfig:
    """Frozen controls; importing this module never changes a default model."""

    method: str = "poisson_hmrf"
    gradient_ratio: float = 0.05
    affinity_ridge: float = 1e-6
    n_neighbors: int = 6
    local_scale_neighbor_rank: int = 7
    coordinate_permutation: bool = False
    quadrature_power: int = 11
    max_outer_iterations: int = 500
    max_p_steps: int = 100
    max_b_steps: int = 100
    kkt_check_interval: int = 25
    minimum_outer_iterations: int = 3
    convergence_patience: int = 5
    relative_tolerance: float = 1e-6
    p_kkt_tolerance: float = 1e-4
    b_kkt_tolerance: float = 1e-4
    affinity_gradient_tolerance: float = 1e-5
    affinity_max_iterations: int = 500
    initial_projected_change: float = 0.05
    armijo: float = 1e-4
    max_backtracks: int = 40
    minimum_step: float = 1e-20
    maximum_step: float = 1e8
    lower_bound: float = 1e-8
    random_state: int = 0
    device: str = "cuda"
    dtype: str = "float32"
    verbose: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "PoissonHMRFConfig":
        if self.method not in ("rna_only_poisson", "poisson_hmrf"):
            raise ValueError("method must be rna_only_poisson or poisson_hmrf")
        if not math.isfinite(self.gradient_ratio) or not 0 <= self.gradient_ratio <= 1:
            raise ValueError("gradient_ratio must lie in [0, 1]")
        if not math.isfinite(self.affinity_ridge) or self.affinity_ridge <= 0:
            raise ValueError("affinity_ridge must be positive")
        for name in (
            "n_neighbors",
            "local_scale_neighbor_rank",
            "quadrature_power",
            "max_outer_iterations",
            "max_p_steps",
            "max_b_steps",
            "kkt_check_interval",
            "minimum_outer_iterations",
            "convergence_patience",
            "affinity_max_iterations",
            "max_backtracks",
        ):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "relative_tolerance",
            "p_kkt_tolerance",
            "b_kkt_tolerance",
            "affinity_gradient_tolerance",
            "initial_projected_change",
            "armijo",
            "minimum_step",
            "maximum_step",
            "lower_bound",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_step >= self.maximum_step:
            raise ValueError("minimum_step must be smaller than maximum_step")
        if self.armijo >= 1 or self.initial_projected_change > 1:
            raise ValueError("Armijo and initial change controls are invalid")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.random_state < 0:
            raise ValueError("random_state must be nonnegative")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
