"""Optional spGSNMF-v3 dual-geometry estimator.

This module is deliberately separate from :mod:`v2`.  M0 delegates to the
frozen v2 implementation exactly, while M1--M3 start from that same fit and
add one spatial mechanism at a time:

``laplacian_v3``
    Quadratic graph smoothness.
``huber_tv_v3``
    Component-wise pseudo-Huber total variation, which preserves sharp
    boundaries better than a quadratic penalty.
``affinity_v3``
    A learned, symmetric, signed component-affinity matrix trained with real
    local edges and an equal number of local non-edges.

The RNA term is the equal-spot conditional KL used by v2.  Keeping that term
fixed is essential: changing to raw-count Poisson KL would confound the
spatial ablation with a change in library-size weighting.  Raw-count Poisson
deviance is nevertheless reported as a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

try:  # ``PYTHONPATH=/path/to/spGSNMF/code``
    from v2.model import V2Fit, fit_spgsnmf_v2, row_normalize
except ModuleNotFoundError:  # package import from a parent module
    from ..v2.model import V2Fit, fit_spgsnmf_v2, row_normalize  # type: ignore

from .config import DualGeometryConfig
from .graph import (
    SpatialGraph,
    build_spatial_graph,
    degree_preserving_randomized_graph,
    permute_edge_labels,
    permuted_coordinate_graph,
    sample_local_negative_edges,
)


EPS = 1e-12
SIMPLEX_LOWER_BOUND = 1e-10


@dataclass
class DualGeometryFit:
    """Complete fit plus the artifacts needed for an auditable benchmark."""

    proportions: np.ndarray
    profiles: np.ndarray
    affinity: np.ndarray | None
    bias: float | None
    positive_edges: np.ndarray
    negative_edges: np.ndarray
    history: list[dict[str, object]]
    diagnostics: dict[str, object]
    config: dict[str, object]
    graph: SpatialGraph
    base_fit: V2Fit


def _validate_inputs(
    counts: np.ndarray,
    coordinates: np.ndarray,
    n_components: int,
    section_labels: Sequence[object] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None]:
    counts = np.asarray(counts, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    n_components = int(n_components)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite nonnegative spot-by-gene matrix")
    if counts.shape[0] < 2 or counts.shape[1] < 2:
        raise ValueError("counts must contain at least two spots and two genes")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("counts must not contain empty spots")
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != counts.shape[0]
        or coordinates.shape[1] < 2
        or np.any(~np.isfinite(coordinates[:, :2]))
    ):
        raise ValueError("coordinates must contain two finite columns per spot")
    if not 1 <= n_components <= min(counts.shape):
        raise ValueError("n_components must lie between 1 and min(counts.shape)")
    labels: np.ndarray | None = None
    if section_labels is not None:
        labels = np.asarray(section_labels, dtype=object)
        if labels.ndim != 1 or labels.shape[0] != counts.shape[0]:
            raise ValueError("section_labels must contain one value per spot")
    return counts, coordinates[:, :2], n_components, labels


def project_rows_to_lower_simplex(
    matrix: np.ndarray, lower_bound: float = 1e-10
) -> np.ndarray:
    """Project rows onto ``x >= lower_bound, sum(x) = 1``.

    The transform to a standard simplex is exact and avoids the subtle row
    sum drift produced by clipping after an ordinary simplex projection.
    """

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise ValueError("matrix must be finite and two-dimensional")
    lower_bound = float(lower_bound)
    width = matrix.shape[1]
    if lower_bound < 0 or lower_bound * width >= 1:
        raise ValueError("lower_bound must satisfy 0 <= K * lower_bound < 1")
    available = 1.0 - width * lower_bound
    shifted = (matrix - lower_bound) / available
    ordered = np.sort(shifted, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, width + 1, dtype=float)[None, :]
    support = ordered - cumulative / indices > 0.0
    support_size = np.maximum(support.sum(axis=1), 1)
    theta = cumulative[np.arange(matrix.shape[0]), support_size - 1] / support_size
    projected = np.maximum(shifted - theta[:, None], 0.0)
    result = lower_bound + available * projected
    # Correct only floating-point residue; this does not change the projection.
    result /= result.sum(axis=1, keepdims=True)
    return result


def project_rows_to_weighted_lower_simplex(
    matrix: np.ndarray,
    weights: np.ndarray,
    lower_bound: float = SIMPLEX_LOWER_BOUND,
) -> np.ndarray:
    """Project rows in a positive diagonal metric onto the lower simplex.

    This solves ``min_x .5 * sum_j w_j (x_j-v_j)^2`` independently per row.
    A monotone scalar bisection finds the equality multiplier and is robust to
    the many active bounds seen in sparse metagene profiles.
    """

    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape != weights.shape:
        raise ValueError("matrix and weights must be two-dimensional and aligned")
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(weights)):
        raise ValueError("matrix and weights must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("weights must be strictly positive")
    width = matrix.shape[1]
    if lower_bound < 0.0 or width * lower_bound >= 1.0:
        raise ValueError("lower_bound must satisfy 0 <= G * lower_bound < 1")
    result = np.empty_like(matrix)
    for row in range(matrix.shape[0]):
        values = matrix[row]
        metric = weights[row]
        breakpoints = metric * (values - lower_bound)
        span = max(
            1.0,
            float(np.max(np.abs(breakpoints), initial=0.0)),
        )
        low = float(np.min(breakpoints, initial=0.0) - span)
        high = float(np.max(breakpoints, initial=0.0) + span)

        def total(multiplier: float) -> float:
            return float(
                np.maximum(lower_bound, values - multiplier / metric).sum()
            )

        while total(low) < 1.0:
            span *= 2.0
            low -= span
        while total(high) > 1.0:
            span *= 2.0
            high += span
        for _ in range(70):
            middle = 0.5 * (low + high)
            if total(middle) > 1.0:
                low = middle
            else:
                high = middle
        projected = np.maximum(lower_bound, values - high / metric)
        # Remove bisection residue by an ordinary exact projection.  At this
        # scale it does not change the active set or weighted optimum.
        result[row] = project_rows_to_lower_simplex(
            projected[None, :], lower_bound=lower_bound
        )[0]
    return result


def entropic_mirror_candidate(
    matrix: np.ndarray,
    gradient: np.ndarray,
    step_size: float,
    lower_bound: float = 1e-10,
) -> np.ndarray:
    """Exponentiated-gradient step on a product of row simplexes.

    A Euclidean update is poorly scaled for this model: P gradients are means
    over spots, while B contains many entries near ``1 / n_genes``.  The
    entropic mirror map is invariant to row-constant gradients, preserves
    nonnegativity, and measures changes relative to the current simplex mass.
    """

    matrix = np.asarray(matrix, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    step_size = float(step_size)
    if matrix.shape != gradient.shape:
        raise ValueError("matrix and gradient must have identical shapes")
    if step_size <= 0 or not np.isfinite(step_size):
        raise ValueError("step_size must be finite and positive")
    width = matrix.shape[1]
    available = 1.0 - width * lower_bound
    if available <= 0.0:
        raise ValueError("lower_bound must satisfy K * lower_bound < 1")
    interior = np.maximum((matrix - lower_bound) / available, EPS)
    interior /= interior.sum(axis=1, keepdims=True)
    logits = np.log(interior) - step_size * available * gradient
    logits -= np.max(logits, axis=1, keepdims=True)
    # The subtraction above prevents overflow.  Clipping the negative tail is
    # only a finite-precision guard and has no effect in the line-search range.
    result = np.exp(np.maximum(logits, -745.0))
    result /= result.sum(axis=1, keepdims=True)
    return lower_bound + available * result


def _mirror_stationarity(
    matrix: np.ndarray, gradient: np.ndarray, lower_bound: float = 1e-10
) -> float:
    """RMS natural-gradient residual for a row-simplex block."""

    available = 1.0 - matrix.shape[1] * lower_bound
    interior = np.maximum((matrix - lower_bound) / max(available, EPS), 0.0)
    interior /= np.maximum(interior.sum(axis=1, keepdims=True), EPS)
    row_mean = np.sum(interior * gradient, axis=1, keepdims=True)
    return _rms(interior * (gradient - row_mean))


def _initial_mirror_step(
    gradient: np.ndarray, target_log_change: float
) -> float:
    tangent = _tangent(np.asarray(gradient, dtype=float))
    scale = float(np.max(np.abs(tangent), initial=0.0))
    return float(target_log_change / max(scale, EPS))


def projected_gradient_mapping(
    matrix: np.ndarray,
    gradient: np.ndarray,
    step_size: float = 1.0,
    lower_bound: float = SIMPLEX_LOWER_BOUND,
) -> np.ndarray:
    """Return the exact lower-simplex projected-gradient mapping.

    Unlike a multiplicative or entropic residual, this mapping detects a
    descent direction that enters a coordinate currently on the lower bound.
    It is zero if and only if the row-simplex KKT conditions hold.  Dividing
    by ``step_size`` keeps the local residual on the gradient scale.
    """

    matrix = np.asarray(matrix, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    step_size = float(step_size)
    if matrix.shape != gradient.shape:
        raise ValueError("matrix and gradient must have identical shapes")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    projected = project_rows_to_lower_simplex(
        matrix - step_size * gradient, lower_bound=lower_bound
    )
    return (matrix - projected) / step_size


def _initial_projected_step(
    gradient: np.ndarray,
    target_change: float,
    minimum: float,
    maximum: float,
) -> float:
    """Scale the first Euclidean proposal to a modest tangent displacement."""

    tangent = _tangent(np.asarray(gradient, dtype=float))
    scale = float(np.max(np.abs(tangent), initial=0.0))
    if scale <= EPS:
        return float(min(max(1.0, minimum), maximum))
    return float(min(max(float(target_change) / scale, minimum), maximum))


def _safe_bb_step(
    previous_matrix: np.ndarray,
    matrix: np.ndarray,
    previous_gradient: np.ndarray,
    gradient: np.ndarray,
    fallback: float,
    minimum: float,
    maximum: float,
) -> float:
    """Return a bounded BB1 proposal, falling back on a carried step.

    The proposal never bypasses Armijo verification; it only reduces the
    number of backtracks when local curvature is informative.
    """

    displacement = np.asarray(matrix, dtype=float) - np.asarray(
        previous_matrix, dtype=float
    )
    gradient_change = np.asarray(gradient, dtype=float) - np.asarray(
        previous_gradient, dtype=float
    )
    numerator = float(np.sum(displacement * displacement))
    denominator = float(np.sum(displacement * gradient_change))
    scale = math.sqrt(
        max(numerator, 0.0)
        * max(float(np.sum(gradient_change * gradient_change)), 0.0)
    )
    if (
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and numerator > 0.0
        and denominator > EPS * max(scale, 1.0)
    ):
        proposal = numerator / denominator
    else:
        proposal = float(fallback)
    return float(min(max(proposal, minimum), maximum))


def conditional_kl_and_gradients(
    composition: np.ndarray,
    proportions: np.ndarray,
    profiles: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return mean equal-spot conditional KL and exact P/B gradients."""

    composition = np.asarray(composition, dtype=float)
    proportions = np.asarray(proportions, dtype=float)
    profiles = np.asarray(profiles, dtype=float)
    fitted = np.maximum(proportions @ profiles, EPS)
    if composition.shape != fitted.shape:
        raise ValueError("composition and P @ B must have the same shape")
    term = fitted - composition
    positive = composition > 0
    term = term.copy()
    term[positive] += composition[positive] * np.log(
        composition[positive] / fitted[positive]
    )
    residual = 1.0 - composition / fitted
    scale = float(composition.shape[0])
    gradient_p = (residual @ profiles.T) / scale
    gradient_b = (proportions.T @ residual) / scale
    return float(term.sum() / scale), gradient_p, gradient_b


def conditional_kl(
    composition: np.ndarray, proportions: np.ndarray, profiles: np.ndarray
) -> float:
    return conditional_kl_and_gradients(composition, proportions, profiles)[0]


def profile_logits_objective_gradient(
    composition: np.ndarray,
    proportions: np.ndarray,
    logits: np.ndarray,
    lower_bound: float = SIMPLEX_LOWER_BOUND,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Conditional KL, exact logit gradient, and lower-simplex profiles.

    Each profile row is parameterized as ``lower + available * softmax(z)``.
    Row-centering fixes the additive softmax gauge numerically.  The full
    logits are retained (rather than dropping an arbitrary reference gene),
    so no particular low-expression gene controls the conditioning.
    """

    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or np.any(~np.isfinite(logits)):
        raise ValueError("logits must be a finite two-dimensional matrix")
    available = 1.0 - logits.shape[1] * float(lower_bound)
    if available <= 0.0:
        raise ValueError("lower_bound must satisfy G * lower_bound < 1")
    centered = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(np.maximum(centered, -745.0))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    profiles = float(lower_bound) + available * probabilities
    value, _, gradient_profiles = conditional_kl_and_gradients(
        composition, proportions, profiles
    )
    row_mean = np.sum(probabilities * gradient_profiles, axis=1, keepdims=True)
    gradient_logits = available * probabilities * (gradient_profiles - row_mean)
    # Remove floating residue in the null gauge direction.
    gradient_logits -= gradient_logits.mean(axis=1, keepdims=True)
    return value, gradient_logits, profiles


def fit_profile_block_logits(
    composition: np.ndarray,
    proportions: np.ndarray,
    initial_profiles: np.ndarray,
    *,
    max_iter: int = 25,
    gradient_tolerance: float = 1e-9,
    function_tolerance: float = 1e-12,
    seed_mass: float = 1e-8,
    lower_bound: float = SIMPLEX_LOWER_BOUND,
) -> tuple[np.ndarray, dict[str, object]]:
    """Approximately solve the convex B block with gauge-fixed L-BFGS.

    A small probability seed is assigned to exact-floor coordinates before
    optimization.  This is essential: their logit derivative would otherwise
    be exactly zero and could not discover a KKT-violating entry direction.
    The returned candidate is accepted only when it decreases the KL from the
    exact input profile matrix; callers still apply the full-objective guard.
    """

    composition = np.asarray(composition, dtype=float)
    proportions = np.asarray(proportions, dtype=float)
    initial_profiles = np.asarray(initial_profiles, dtype=float)
    max_iter = int(max_iter)
    gradient_tolerance = float(gradient_tolerance)
    function_tolerance = float(function_tolerance)
    seed_mass = float(seed_mass)
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    for name, value in (
        ("gradient_tolerance", gradient_tolerance),
        ("function_tolerance", function_tolerance),
        ("seed_mass", seed_mass),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if initial_profiles.ndim != 2 or np.any(~np.isfinite(initial_profiles)):
        raise ValueError("initial_profiles must be finite and two-dimensional")
    available = 1.0 - initial_profiles.shape[1] * float(lower_bound)
    if available <= 0.0:
        raise ValueError("lower_bound must satisfy G * lower_bound < 1")
    probabilities = np.maximum(
        (initial_profiles - float(lower_bound)) / available, 0.0
    )
    probabilities = np.maximum(probabilities, seed_mass)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    initial_logits = np.log(probabilities)
    initial_logits -= initial_logits.mean(axis=1, keepdims=True)
    shape = initial_logits.shape
    exact_initial_value = conditional_kl(
        composition, proportions, initial_profiles
    )

    def objective(flat_logits: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _ = profile_logits_objective_gradient(
            composition,
            proportions,
            np.asarray(flat_logits, dtype=float).reshape(shape),
            lower_bound=lower_bound,
        )
        return value, gradient.ravel()

    result = minimize(
        objective,
        initial_logits.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iter,
            "ftol": function_tolerance,
            "gtol": gradient_tolerance,
            "maxls": 30,
            "maxcor": 10,
        },
    )
    final_logits = np.asarray(result.x, dtype=float).reshape(shape)
    final_logits -= final_logits.mean(axis=1, keepdims=True)
    final_value, final_logit_gradient, candidate = profile_logits_objective_gradient(
        composition,
        proportions,
        final_logits,
        lower_bound=lower_bound,
    )
    _, _, final_profile_gradient = conditional_kl_and_gradients(
        composition, proportions, candidate
    )
    mapping = projected_gradient_mapping(
        candidate, final_profile_gradient, lower_bound=lower_bound
    )
    accepted = bool(
        np.isfinite(final_value)
        and final_value <= exact_initial_value + 1e-12 * max(abs(exact_initial_value), 1.0)
    )
    if not accepted:
        candidate = initial_profiles.copy()
        final_value = exact_initial_value
        _, _, final_profile_gradient = conditional_kl_and_gradients(
            composition, proportions, candidate
        )
        mapping = projected_gradient_mapping(
            candidate, final_profile_gradient, lower_bound=lower_bound
        )
    return candidate, {
        "accepted": accepted,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "initial_unit_loss": float(exact_initial_value),
        "final_unit_loss": float(final_value),
        "logit_gradient_norm": float(np.linalg.norm(final_logit_gradient)),
        "projected_mapping_rms": _rms(mapping),
        "projected_mapping_max": float(np.max(np.abs(mapping), initial=0.0)),
    }


def fit_profile_block_diagonal_newton(
    composition: np.ndarray,
    proportions: np.ndarray,
    initial_profiles: np.ndarray,
    *,
    max_iter: int = 20,
    curvature_floor: float = 1e-8,
    initial_change: float = 0.05,
    armijo: float = 1e-4,
    max_backtracks: int = 30,
    stationarity_fraction: float = 0.05,
    lower_bound: float = SIMPLEX_LOWER_BOUND,
) -> tuple[np.ndarray, dict[str, object]]:
    """Monotone active-bound diagonal-Newton solve of the convex B block.

    The exact KL Hessian diagonal rescales genes whose curvature differs by
    orders of magnitude.  Each quadratic model is minimized by a weighted
    lower-simplex projection, so coordinates can enter or leave the bound in
    one step.  Armijo on the real KL, not the quadratic model, guarantees a
    monotone accepted block.
    """

    composition = np.asarray(composition, dtype=float)
    proportions = np.asarray(proportions, dtype=float)
    profiles = np.asarray(initial_profiles, dtype=float).copy()
    max_iter = int(max_iter)
    max_backtracks = int(max_backtracks)
    curvature_floor = float(curvature_floor)
    initial_change = float(initial_change)
    if max_iter < 1 or max_backtracks < 1:
        raise ValueError("iteration and backtrack limits must be positive")
    if not np.isfinite(curvature_floor) or curvature_floor <= 0.0:
        raise ValueError("curvature_floor must be finite and positive")
    if not np.isfinite(initial_change) or not 0.0 < initial_change <= 1.0:
        raise ValueError("initial_change must lie in (0, 1]")
    if not 0.0 < float(armijo) < 1.0:
        raise ValueError("armijo must lie in (0, 1)")
    if not 0.0 < float(stationarity_fraction) < 1.0:
        raise ValueError("stationarity_fraction must lie in (0, 1)")
    initial_value, _, initial_gradient = conditional_kl_and_gradients(
        composition, proportions, profiles
    )
    initial_mapping = projected_gradient_mapping(
        profiles, initial_gradient, lower_bound=lower_bound
    )
    initial_mapping_rms = _rms(initial_mapping)
    value = float(initial_value)
    accepted_steps = 0
    function_evaluations = 1
    backtracks = 0
    for iteration in range(max_iter):
        value, _, gradient = conditional_kl_and_gradients(
            composition, proportions, profiles
        )
        mapping = projected_gradient_mapping(
            profiles, gradient, lower_bound=lower_bound
        )
        if (
            iteration > 0
            and _rms(mapping)
            <= stationarity_fraction * max(initial_mapping_rms, EPS)
        ):
            break
        fitted = np.maximum(proportions @ profiles, EPS)
        curvature = (
            np.square(proportions).T @ (composition / np.square(fitted))
        ) / float(composition.shape[0])
        curvature = np.maximum(curvature, curvature_floor)
        # The exact Hessian diverges like 1/B^2 at a near-zero fitted gene,
        # making pure Newton merely double a floor coordinate per iteration.
        # Cap the local metric to propose at most a modest coordinate-scale
        # change; Armijo on the exact KL safely shortens it when necessary.
        curvature_cap = np.maximum(
            np.abs(gradient) / initial_change, curvature_floor
        )
        curvature = np.minimum(curvature, curvature_cap)
        quadratic_center = profiles - gradient / curvature
        quadratic_candidate = project_rows_to_weighted_lower_simplex(
            quadratic_center, curvature, lower_bound=lower_bound
        )
        direction = quadratic_candidate - profiles
        directional = float(np.sum(gradient * direction))
        if not np.isfinite(directional) or directional >= -EPS:
            break
        step = 1.0
        accepted = False
        for _ in range(max_backtracks):
            candidate = profiles + step * direction
            candidate_value = conditional_kl(composition, proportions, candidate)
            function_evaluations += 1
            if candidate_value <= value + float(armijo) * step * directional:
                profiles = candidate
                value = float(candidate_value)
                accepted_steps += 1
                accepted = True
                break
            step *= 0.5
            backtracks += 1
        if not accepted:
            break
    final_value, _, final_gradient = conditional_kl_and_gradients(
        composition, proportions, profiles
    )
    final_mapping = projected_gradient_mapping(
        profiles, final_gradient, lower_bound=lower_bound
    )
    accepted = bool(
        np.isfinite(final_value)
        and final_value <= initial_value + 1e-12 * max(abs(initial_value), 1.0)
    )
    if not accepted:
        profiles = np.asarray(initial_profiles, dtype=float).copy()
        final_value = initial_value
        final_mapping = initial_mapping
    return profiles, {
        "accepted": accepted,
        "success": bool(accepted_steps > 0 or initial_mapping_rms <= EPS),
        "status": 0 if accepted else 1,
        "message": "monotone_diagonal_newton",
        "iterations": int(accepted_steps),
        "function_evaluations": int(function_evaluations),
        "backtracks": int(backtracks),
        "initial_unit_loss": float(initial_value),
        "final_unit_loss": float(final_value),
        "projected_mapping_rms": _rms(final_mapping),
        "projected_mapping_max": float(
            np.max(np.abs(final_mapping), initial=0.0)
        ),
    }


def fit_profile_block_projected_gradient(
    composition: np.ndarray,
    proportions: np.ndarray,
    initial_profiles: np.ndarray,
    *,
    max_iter: int = 20,
    initial_change: float = 0.05,
    step_growth: float = 1.5,
    minimum_step: float = 1e-10,
    maximum_step: float = 1e10,
    armijo: float = 1e-4,
    max_backtracks: int = 35,
    lower_bound: float = SIMPLEX_LOWER_BOUND,
) -> tuple[np.ndarray, dict[str, object]]:
    """Monotone true-projection B polish used after the current P update."""

    profiles = np.asarray(initial_profiles, dtype=float).copy()
    initial_value = conditional_kl(composition, proportions, profiles)
    value = float(initial_value)
    step = math.nan
    accepted_steps = 0
    evaluations = 1
    backtracks = 0
    previous_profiles: np.ndarray | None = None
    previous_gradient: np.ndarray | None = None
    for _ in range(int(max_iter)):
        _, _, gradient = conditional_kl_and_gradients(
            composition, proportions, profiles
        )
        if previous_profiles is not None and previous_gradient is not None:
            proposal = _safe_bb_step(
                previous_profiles,
                profiles,
                previous_gradient,
                gradient,
                fallback=(step * step_growth if np.isfinite(step) else 1.0),
                minimum=minimum_step,
                maximum=maximum_step,
            )
        elif np.isfinite(step):
            proposal = min(step * step_growth, maximum_step)
        else:
            proposal = _initial_projected_step(
                gradient, initial_change, minimum_step, maximum_step
            )
        local_step = float(proposal)
        old_profiles = profiles.copy()
        old_gradient = gradient.copy()
        accepted = False
        for _ in range(int(max_backtracks)):
            candidate = project_rows_to_lower_simplex(
                profiles - local_step * gradient, lower_bound=lower_bound
            )
            displacement = candidate - profiles
            directional = float(np.sum(gradient * displacement))
            candidate_value = conditional_kl(
                composition, proportions, candidate
            )
            evaluations += 1
            if (
                directional <= 0.0
                and candidate_value
                <= value + float(armijo) * directional
                + 1e-12 * max(abs(value), 1.0)
            ):
                profiles = candidate
                value = float(candidate_value)
                step = local_step
                accepted_steps += 1
                previous_profiles = old_profiles
                previous_gradient = old_gradient
                accepted = True
                break
            local_step *= 0.5
            backtracks += 1
        if not accepted:
            break
    _, _, final_gradient = conditional_kl_and_gradients(
        composition, proportions, profiles
    )
    mapping = projected_gradient_mapping(
        profiles, final_gradient, lower_bound=lower_bound
    )
    accepted = bool(
        value <= initial_value + 1e-12 * max(abs(initial_value), 1.0)
    )
    return profiles, {
        "accepted": accepted,
        "success": bool(accepted_steps > 0),
        "status": 0 if accepted else 1,
        "message": "post_P_projected_gradient_polish",
        "iterations": int(accepted_steps),
        "function_evaluations": int(evaluations),
        "backtracks": int(backtracks),
        "initial_unit_loss": float(initial_value),
        "final_unit_loss": float(value),
        "final_step": float(step),
        "projected_mapping_rms": _rms(mapping),
        "projected_mapping_max": float(np.max(np.abs(mapping), initial=0.0)),
    }


def raw_count_poisson_deviance(
    counts: np.ndarray, proportions: np.ndarray, profiles: np.ndarray
) -> float:
    """Poisson deviance per observed molecule, for diagnosis only."""

    counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
    libraries = counts.sum(axis=1, keepdims=True)
    expected = np.maximum(libraries * (proportions @ profiles), EPS)
    term = expected - counts
    positive = counts > 0
    term[positive] += counts[positive] * np.log(counts[positive] / expected[positive])
    return float(2.0 * term.sum() / max(float(counts.sum()), 1.0))


def laplacian_loss_gradient(
    proportions: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Weighted mean-edge quadratic loss and gradient."""

    proportions = np.asarray(proportions, dtype=float)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if weights is None:
        weights = np.ones(edges.shape[0], dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (edges.shape[0],):
        raise ValueError("weights must contain one value per edge")
    gradient = np.zeros_like(proportions)
    if not edges.size:
        return 0.0, gradient
    denominator = max(float(weights.sum()), EPS)
    left, right = edges[:, 0], edges[:, 1]
    difference = proportions[left] - proportions[right]
    contribution = 2.0 * weights[:, None] * difference / denominator
    np.add.at(gradient, left, contribution)
    np.add.at(gradient, right, -contribution)
    loss = np.sum(weights[:, None] * np.square(difference)) / denominator
    return float(loss), gradient


def huber_tv_loss_gradient(
    proportions: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray | None = None,
    delta: float = 0.02,
) -> tuple[float, np.ndarray]:
    """Component-wise pseudo-Huber/TV loss and exact gradient."""

    proportions = np.asarray(proportions, dtype=float)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if weights is None:
        weights = np.ones(edges.shape[0], dtype=float)
    weights = np.asarray(weights, dtype=float)
    delta = float(delta)
    if delta <= 0 or not np.isfinite(delta):
        raise ValueError("delta must be finite and positive")
    if weights.shape != (edges.shape[0],):
        raise ValueError("weights must contain one value per edge")
    gradient = np.zeros_like(proportions)
    if not edges.size:
        return 0.0, gradient
    denominator = max(float(weights.sum()), EPS)
    left, right = edges[:, 0], edges[:, 1]
    difference = proportions[left] - proportions[right]
    radius = np.sqrt(np.square(difference) + delta * delta)
    contribution = weights[:, None] * difference / radius / denominator
    np.add.at(gradient, left, contribution)
    np.add.at(gradient, right, -contribution)
    loss = np.sum(weights[:, None] * (radius - delta)) / denominator
    return float(loss), gradient


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def affinity_scores(
    proportions: np.ndarray, affinity: np.ndarray, bias: float, edges: np.ndarray
) -> np.ndarray:
    """Symmetric bilinear edge scores ``P_i.T @ A @ P_j + b``."""

    proportions = np.asarray(proportions, dtype=float)
    affinity = np.asarray(affinity, dtype=float)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if affinity.shape != (proportions.shape[1], proportions.shape[1]):
        raise ValueError("affinity must be K by K")
    if edges.size == 0:
        return np.empty(0, dtype=float)
    return np.einsum(
        "ei,ij,ej->e",
        proportions[edges[:, 0]],
        affinity,
        proportions[edges[:, 1]],
        optimize=True,
    ) + float(bias)


def affinity_loss_gradients(
    proportions: np.ndarray,
    affinity: np.ndarray,
    bias: float,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    ridge: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, float, dict[str, float]]:
    """Balanced logistic affinity loss and P/A/b gradients.

    Positive and negative classes are averaged separately, as prescribed in
    the experiment.  ``logaddexp`` and a branch-stable sigmoid keep the result
    finite even for logits with magnitude in the thousands.
    """

    proportions = np.asarray(proportions, dtype=float)
    affinity = np.asarray(affinity, dtype=float)
    positive_edges = np.asarray(positive_edges, dtype=int).reshape(-1, 2)
    negative_edges = np.asarray(negative_edges, dtype=int).reshape(-1, 2)
    ridge = float(ridge)
    if positive_edges.shape[0] == 0 or negative_edges.shape[0] == 0:
        raise ValueError("affinity loss requires positive and negative edges")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    affinity = 0.5 * (affinity + affinity.T)
    gradient_p = np.zeros_like(proportions)
    gradient_a_raw = np.zeros_like(affinity)
    gradient_b = 0.0

    positive_scores = affinity_scores(
        proportions, affinity, bias, positive_edges
    )
    negative_scores = affinity_scores(
        proportions, affinity, bias, negative_edges
    )
    positive_loss = float(np.mean(np.logaddexp(0.0, -positive_scores)))
    negative_loss = float(np.mean(np.logaddexp(0.0, negative_scores)))

    for edges, scores, labels in (
        (positive_edges, positive_scores, 1.0),
        (negative_edges, negative_scores, 0.0),
    ):
        q = (_sigmoid(scores) - labels) / float(edges.shape[0])
        left, right = edges[:, 0], edges[:, 1]
        left_values = proportions[left]
        right_values = proportions[right]
        left_contribution = q[:, None] * (right_values @ affinity.T)
        right_contribution = q[:, None] * (left_values @ affinity)
        np.add.at(gradient_p, left, left_contribution)
        np.add.at(gradient_p, right, right_contribution)
        gradient_a_raw += np.einsum(
            "e,ei,ej->ij", q, left_values, right_values, optimize=True
        )
        gradient_b += float(q.sum())

    gradient_a = 0.5 * (gradient_a_raw + gradient_a_raw.T)
    ridge_value = ridge * float(np.square(affinity).sum())
    gradient_a += 2.0 * ridge * affinity
    total = positive_loss + negative_loss + ridge_value
    pieces = {
        "positive_bce": positive_loss,
        "negative_bce": negative_loss,
        "balanced_bce": positive_loss + negative_loss,
        "ridge": ridge_value,
    }
    return float(total), gradient_p, gradient_a, float(gradient_b), pieces


def center_affinity(
    affinity: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    """Remove the A/b constant-direction ambiguity without changing logits."""

    affinity = np.asarray(affinity, dtype=float)
    affinity = 0.5 * (affinity + affinity.T)
    shift = float(affinity.mean())
    return affinity - shift, float(bias) + shift


def _upper_indices(k: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(k)


def _pack_affinity(affinity: np.ndarray, bias: float) -> np.ndarray:
    indices = _upper_indices(affinity.shape[0])
    return np.concatenate([affinity[indices], np.asarray([bias], dtype=float)])


def _unpack_affinity(parameters: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    indices = _upper_indices(k)
    values = np.asarray(parameters[:-1], dtype=float)
    affinity = np.zeros((k, k), dtype=float)
    affinity[indices] = values
    affinity[(indices[1], indices[0])] = values
    return affinity, float(parameters[-1])


def _pack_symmetric_gradient(gradient: np.ndarray, gradient_b: float) -> np.ndarray:
    indices = _upper_indices(gradient.shape[0])
    values = gradient[indices].copy()
    values[indices[0] != indices[1]] *= 2.0
    return np.concatenate([values, np.asarray([gradient_b], dtype=float)])


def fit_affinity_block(
    proportions: np.ndarray,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    ridge: float,
    *,
    initial_affinity: np.ndarray | None = None,
    initial_bias: float = 0.0,
    max_iter: int = 100,
    tolerance: float = 1e-8,
    function_tolerance: float = 1e-12,
) -> tuple[np.ndarray, float, dict[str, object]]:
    """Solve the convex symmetric-A block with L-BFGS."""

    proportions = np.asarray(proportions, dtype=float)
    k = proportions.shape[1]
    if initial_affinity is None:
        initial_affinity = np.zeros((k, k), dtype=float)
    initial_affinity, initial_bias = center_affinity(
        np.asarray(initial_affinity, dtype=float), float(initial_bias)
    )
    tolerance = float(tolerance)
    function_tolerance = float(function_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(function_tolerance) or function_tolerance <= 0.0:
        raise ValueError("function_tolerance must be finite and positive")
    initial = _pack_affinity(initial_affinity, initial_bias)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        affinity, bias = _unpack_affinity(parameters, k)
        value, _, gradient_a, gradient_b, _ = affinity_loss_gradients(
            proportions,
            affinity,
            bias,
            positive_edges,
            negative_edges,
            ridge=ridge,
        )
        return value, _pack_symmetric_gradient(gradient_a, gradient_b)

    initial_value = float(objective(initial)[0])
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iter),
            "ftol": function_tolerance,
            "gtol": tolerance,
            "maxls": 30,
        },
    )
    affinity, bias = _unpack_affinity(result.x, k)
    affinity, bias = center_affinity(affinity, bias)
    final_value, _, final_gradient_a, final_gradient_b, _ = affinity_loss_gradients(
        proportions,
        affinity,
        bias,
        positive_edges,
        negative_edges,
        ridge=ridge,
    )
    # Centering preserves logits and weakly reduces the ridge.  In the rare
    # event an optimizer returns a worse numerical point, retain the start.
    if not np.isfinite(final_value) or final_value > initial_value + 1e-10:
        affinity, bias = initial_affinity.copy(), float(initial_bias)
        final_value, _, final_gradient_a, final_gradient_b, _ = affinity_loss_gradients(
            proportions,
            affinity,
            bias,
            positive_edges,
            negative_edges,
            ridge=ridge,
        )
        accepted = False
    else:
        accepted = True
    return affinity, bias, {
        "accepted": accepted,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "packed_gradient_norm": float(
            np.linalg.norm(
                _pack_symmetric_gradient(final_gradient_a, final_gradient_b)
            )
        ),
        "initial_unit_loss": initial_value,
        "final_unit_loss": float(final_value),
    }


def _tangent(gradient: np.ndarray) -> np.ndarray:
    return gradient - gradient.mean(axis=1, keepdims=True)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))


def _rna_reference(
    composition: np.ndarray, proportions: np.ndarray, profiles: np.ndarray
) -> dict[str, float]:
    fitted = np.maximum(proportions @ profiles, EPS)
    n = float(composition.shape[0])
    positive = np.ones_like(composition) @ profiles.T / n
    negative = (composition / fitted) @ profiles.T / n
    _, raw_gradient, _ = conditional_kl_and_gradients(
        composition, proportions, profiles
    )
    reference = proportions * 0.5 * (np.abs(positive) + np.abs(negative))
    return {
        "reference_rms": _rms(reference),
        "raw_rms": _rms(raw_gradient),
        "tangent_rms": _rms(_tangent(raw_gradient)),
    }


def _spatial_reference(
    method: str,
    proportions: np.ndarray,
    graph: SpatialGraph,
    *,
    delta: float,
    affinity: np.ndarray | None,
    bias: float | None,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    ridge: float,
) -> dict[str, float]:
    absolute = np.zeros_like(proportions)
    if method == "rna_refined_control":
        gradient = np.zeros_like(proportions)
    elif method == "laplacian_v3":
        _, gradient = laplacian_loss_gradient(
            proportions, graph.edges, graph.weights
        )
        if graph.edges.size:
            denominator = max(float(graph.weights.sum()), EPS)
            difference = proportions[graph.edges[:, 0]] - proportions[graph.edges[:, 1]]
            contribution = (
                2.0 * graph.weights[:, None] * np.abs(difference) / denominator
            )
            np.add.at(absolute, graph.edges[:, 0], contribution)
            np.add.at(absolute, graph.edges[:, 1], contribution)
    elif method == "huber_tv_v3":
        _, gradient = huber_tv_loss_gradient(
            proportions, graph.edges, graph.weights, delta
        )
        if graph.edges.size:
            denominator = max(float(graph.weights.sum()), EPS)
            difference = proportions[graph.edges[:, 0]] - proportions[graph.edges[:, 1]]
            contribution = graph.weights[:, None] * np.abs(difference) / np.sqrt(
                np.square(difference) + delta * delta
            ) / denominator
            np.add.at(absolute, graph.edges[:, 0], contribution)
            np.add.at(absolute, graph.edges[:, 1], contribution)
    elif method == "affinity_v3":
        if affinity is None or bias is None:
            raise ValueError("affinity_v3 requires an initialized A and bias")
        _, gradient, _, _, _ = affinity_loss_gradients(
            proportions,
            affinity,
            bias,
            positive_edges,
            negative_edges,
            ridge=ridge,
        )
        for edges, labels in ((positive_edges, 1.0), (negative_edges, 0.0)):
            scores = affinity_scores(proportions, affinity, bias, edges)
            q = np.abs(_sigmoid(scores) - labels) / float(edges.shape[0])
            left, right = edges[:, 0], edges[:, 1]
            np.add.at(
                absolute,
                left,
                q[:, None] * np.abs(proportions[right] @ affinity.T),
            )
            np.add.at(
                absolute,
                right,
                q[:, None] * np.abs(proportions[left] @ affinity),
            )
    else:
        raise ValueError(f"unknown spatial method: {method}")
    reference = 0.5 * proportions * absolute
    return {
        "reference_rms": _rms(reference),
        "raw_rms": _rms(gradient),
        "tangent_rms": _rms(_tangent(gradient)),
    }


def calibrate_spatial_lambda(
    target_ratio: float,
    rna_reference: Mapping[str, float],
    spatial_reference: Mapping[str, float],
) -> tuple[float, dict[str, float | str]]:
    """Choose lambda from simplex-tangent RNA/spatial gradients.

    The prompt defines strength through ``||grad_P L||``.  Because P is
    row-simplex constrained, row-constant normal components cannot move P and
    are removed before taking the norm.  Non-cancelled contribution norms are
    retained in the audit fields, but they are not substituted for the stated
    gradient-ratio contract.
    """

    target_ratio = float(target_ratio)
    numerator = float(rna_reference["tangent_rms"])
    denominator = float(spatial_reference["tangent_rms"])
    if target_ratio <= 0:
        return 0.0, {"status": "ratio_zero", "lambda": 0.0}
    if not np.isfinite(denominator) or denominator <= EPS:
        return 0.0, {
            "status": "spatial_reference_zero",
            "lambda": 0.0,
            "target_ratio": target_ratio,
        }
    value = target_ratio * numerator / denominator
    value = float(min(max(value, 0.0), 1e8))
    raw_denominator = max(float(rna_reference["tangent_rms"]), EPS)
    return value, {
        "status": "calibrated_simplex_tangent_gradient",
        "lambda": value,
        "target_ratio": target_ratio,
        "rna_reference_rms": float(rna_reference["reference_rms"]),
        "spatial_reference_rms": float(spatial_reference["reference_rms"]),
        "rna_calibration_tangent_rms": numerator,
        "spatial_calibration_tangent_rms": denominator,
        "achieved_reference_ratio": (
            value
            * float(spatial_reference["reference_rms"])
            / max(float(rna_reference["reference_rms"]), EPS)
        ),
        "rna_raw_rms": float(rna_reference["raw_rms"]),
        "rna_tangent_rms": float(rna_reference["tangent_rms"]),
        "spatial_raw_rms": float(spatial_reference["raw_rms"]),
        "spatial_tangent_rms": float(spatial_reference["tangent_rms"]),
        "achieved_tangent_ratio": value * denominator / raw_denominator,
    }


def _unit_spatial_objective(
    method: str,
    proportions: np.ndarray,
    graph: SpatialGraph,
    *,
    delta: float,
    affinity: np.ndarray | None,
    bias: float | None,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray, dict[str, float]]:
    if method == "laplacian_v3":
        value, gradient = laplacian_loss_gradient(
            proportions, graph.edges, graph.weights
        )
        return value, gradient, {"laplacian": value}
    if method == "rna_refined_control":
        return 0.0, np.zeros_like(proportions), {"spatial_control": 0.0}
    if method == "huber_tv_v3":
        value, gradient = huber_tv_loss_gradient(
            proportions, graph.edges, graph.weights, delta
        )
        return value, gradient, {"huber_tv": value}
    if method == "affinity_v3":
        if affinity is None or bias is None:
            raise ValueError("affinity_v3 requires A and bias")
        value, gradient, _, _, pieces = affinity_loss_gradients(
            proportions,
            affinity,
            bias,
            positive_edges,
            negative_edges,
            ridge=ridge,
        )
        return value, gradient, pieces
    raise ValueError(f"unknown method: {method}")


def _objective(
    composition: np.ndarray,
    proportions: np.ndarray,
    profiles: np.ndarray,
    method: str,
    graph: SpatialGraph,
    spatial_lambda: float,
    *,
    delta: float,
    affinity: np.ndarray | None,
    bias: float | None,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    ridge: float,
) -> tuple[float, float, float, dict[str, float]]:
    data = conditional_kl(composition, proportions, profiles)
    spatial, _, pieces = _unit_spatial_objective(
        method,
        proportions,
        graph,
        delta=delta,
        affinity=affinity,
        bias=bias,
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        ridge=ridge,
    )
    return float(data + spatial_lambda * spatial), data, spatial, pieces


def _mean_edge_distance(proportions: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if edges.size == 0:
        return math.nan, math.nan
    difference = proportions[edges[:, 0]] - proportions[edges[:, 1]]
    return float(np.mean(np.abs(difference).sum(axis=1)) / 2.0), float(
        np.mean(np.sqrt(np.square(difference).sum(axis=1)))
    )


def _affinity_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """AUC from pairwise comparisons; ties receive half credit."""

    if not positive_scores.size or not negative_scores.size:
        return math.nan
    # Chunking prevents a large temporary matrix on unusually dense graphs.
    wins = 0.0
    for start in range(0, positive_scores.size, 512):
        comparison = positive_scores[start : start + 512, None] - negative_scores[None, :]
        wins += float(np.sum(comparison > 0) + 0.5 * np.sum(comparison == 0))
    return wins / float(positive_scores.size * negative_scores.size)


def _collapse_diagnostics(
    proportions: np.ndarray,
    profiles: np.ndarray,
    base_proportions: np.ndarray,
    initial_rna: float,
    final_rna: float,
    rna_cap: float,
    affinity: np.ndarray | None,
    bias: float | None,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
) -> dict[str, object]:
    component_mass = proportions.mean(axis=0)
    spot_entropy = -np.sum(proportions * np.log(np.maximum(proportions, EPS)), axis=1)
    normalized_entropy = spot_entropy / max(math.log(proportions.shape[1]), EPS)
    variance = float(np.mean(np.square(proportions - proportions.mean(axis=0))))
    base_variance = float(
        np.mean(np.square(base_proportions - base_proportions.mean(axis=0)))
    )
    normalized_mass = component_mass / max(float(component_mass.sum()), EPS)
    effective_components = float(
        np.exp(-np.sum(normalized_mass * np.log(np.maximum(normalized_mass, EPS))))
    )
    profile_norm = np.sqrt(np.maximum(np.square(profiles).sum(axis=1), EPS))
    cosine = profiles @ profiles.T / (profile_norm[:, None] * profile_norm[None, :])
    if profiles.shape[0] > 1:
        profile_max_cosine = float(np.max(cosine[~np.eye(profiles.shape[0], dtype=bool)]))
    else:
        profile_max_cosine = 0.0
    flags = {
        "spatial_variance_collapse": bool(
            base_variance > EPS and variance / base_variance < 0.05
        ),
        "dominant_component": bool(float(component_mass.max()) > 0.95),
        "vanished_component": bool(float(component_mass.min()) < 1e-4),
        "duplicated_profiles": bool(profile_max_cosine > 0.9995),
        "rna_sacrifice": bool(
            final_rna > initial_rna * (1.0 + float(rna_cap)) + 1e-12
        ),
    }
    affinity_values: dict[str, object] = {}
    if affinity is not None and bias is not None:
        positive_scores = affinity_scores(
            proportions, affinity, bias, positive_edges
        )
        negative_scores = affinity_scores(
            proportions, affinity, bias, negative_edges
        )
        scores = np.concatenate([positive_scores, negative_scores])
        saturated = float(np.mean(np.abs(scores) > 10.0)) if scores.size else 0.0
        flags["affinity_explosion"] = bool(
            float(np.max(np.abs(affinity), initial=0.0)) > 50.0 or saturated > 0.25
        )
        affinity_values = {
            "affinity_frobenius_norm": float(np.linalg.norm(affinity)),
            "affinity_spectral_norm": float(np.linalg.norm(affinity, ord=2)),
            "affinity_max_abs": float(np.max(np.abs(affinity), initial=0.0)),
            "affinity_mean": float(affinity.mean()),
            "bias": float(bias),
            "logit_quantiles": (
                np.quantile(scores, [0.0, 0.05, 0.5, 0.95, 1.0]).tolist()
                if scores.size
                else []
            ),
            "logit_saturated_fraction": saturated,
            "affinity_training_auc": _affinity_auc(positive_scores, negative_scores),
        }
    return {
        "component_mass": component_mass.tolist(),
        "effective_component_count": effective_components,
        "mean_normalized_spot_entropy": float(np.mean(normalized_entropy)),
        "spot_variance": variance,
        "base_spot_variance": base_variance,
        "spot_variance_ratio": variance / max(base_variance, EPS),
        "distance_from_v2_frobenius": float(np.linalg.norm(proportions - base_proportions)),
        "profile_max_offdiagonal_cosine": profile_max_cosine,
        "profile_minimum": float(profiles.min()),
        "profile_row_sum_max_error": float(np.max(np.abs(profiles.sum(axis=1) - 1.0))),
        "collapse_flags": flags,
        "collapse_detected": bool(any(flags.values())),
        **affinity_values,
    }


def _identity_fit(
    base_fit: V2Fit,
    graph: SpatialGraph,
    negative_edges: np.ndarray,
    config: DualGeometryConfig,
    reason: str,
    counts: np.ndarray,
    elapsed: float,
) -> DualGeometryFit:
    composition = row_normalize(counts)
    data = conditional_kl(composition, base_fit.proportions, base_fit.profiles)
    diagnostics = {
        "method": config.method,
        "identity_return": True,
        "termination_reason": reason,
        "iterations": 0,
        "conditional_kl": data,
        "raw_count_poisson_deviance_per_mass": raw_count_poisson_deviance(
            counts, base_fit.proportions, base_fit.profiles
        ),
        "spatial_lambda": 0.0,
        "graph": graph.diagnostics,
        "base_v2": base_fit.diagnostics,
        "runtime_sec": float(elapsed),
        "truth_used": False,
        "gene_side_penalty_active": False,
        "solvability_penalty_active": False,
    }
    return DualGeometryFit(
        proportions=base_fit.proportions.copy(),
        profiles=base_fit.profiles.copy(),
        affinity=None,
        bias=None,
        positive_edges=graph.edges.copy(),
        negative_edges=np.asarray(negative_edges, dtype=int).copy(),
        history=[],
        diagnostics=diagnostics,
        config=config.to_dict(),
        graph=graph,
        base_fit=base_fit,
    )


def fit_spgsnmf_v3_dual_geometry(
    counts: np.ndarray,
    n_components: int,
    coordinates: np.ndarray,
    section_labels: Sequence[object] | np.ndarray | None = None,
    config: DualGeometryConfig | Mapping[str, object] | None = None,
    initial_result: V2Fit | DualGeometryFit | None = None,
) -> DualGeometryFit:
    """Fit an explicitly selected M0--M3 dual-geometry configuration."""

    started = time.perf_counter()
    counts, coordinates, n_components, section_labels = _validate_inputs(
        counts, coordinates, n_components, section_labels
    )
    validation_finished = time.perf_counter()
    if config is None:
        config = DualGeometryConfig()
    elif isinstance(config, Mapping):
        config = DualGeometryConfig(**dict(config))
    if not isinstance(config, DualGeometryConfig):
        raise TypeError("config must be DualGeometryConfig, a mapping, or None")
    config.validate()

    v2_started = time.perf_counter()
    if initial_result is None:
        base_fit = fit_spgsnmf_v2(
            counts,
            n_components,
            coords=coordinates,
            random_state=config.random_state,
            max_iter=config.v2_max_iter,
            tolerance=config.v2_tolerance,
        )
    elif isinstance(initial_result, DualGeometryFit):
        base_fit = initial_result.base_fit
    elif isinstance(initial_result, V2Fit):
        base_fit = initial_result
    else:
        raise TypeError("initial_result must be a V2Fit or DualGeometryFit")
    if base_fit.proportions.shape != (counts.shape[0], n_components):
        raise ValueError("initial_result proportions have an incompatible shape")
    if base_fit.profiles.shape != (n_components, counts.shape[1]):
        raise ValueError("initial_result profiles have an incompatible shape")
    v2_finished = time.perf_counter()

    # M0 is an exact runtime as well as numerical baseline.  It must not pay
    # for graph construction or negative sampling that frozen v2 never uses.
    if config.method == "baseline_v2" or config.gradient_ratio == 0.0:
        unused_graph = SpatialGraph(
            n_spots=counts.shape[0],
            edges=np.empty((0, 2), dtype=int),
            distances=np.empty(0, dtype=float),
            weights=np.empty(0, dtype=float),
            sections=np.zeros(counts.shape[0], dtype=int),
            diagnostics={
                "construction": "not_built_for_exact_identity_baseline",
                "n_spots": int(counts.shape[0]),
                "n_edges": 0,
                "cross_section_edges": 0,
            },
        )
        reason = (
            "exact_frozen_v2_baseline"
            if config.method == "baseline_v2"
            else "gradient_ratio_zero"
        )
        identity = _identity_fit(
            base_fit,
            unused_graph,
            np.empty((0, 2), dtype=int),
            config,
            reason,
            counts,
            time.perf_counter() - started,
        )
        identity.diagnostics["timing"] = {
            "validation_sec": float(validation_finished - started),
            "v2_fit_sec": float(v2_finished - v2_started),
            "graph_preprocessing_sec": 0.0,
            "affinity_prefit_sec": 0.0,
            "alternating_optimization_sec": 0.0,
            "total_sec": float(time.perf_counter() - started),
        }
        return identity

    graph_started = time.perf_counter()
    if config.method == "rna_refined_control":
        canonical_graph = SpatialGraph(
            n_spots=counts.shape[0],
            edges=np.empty((0, 2), dtype=int),
            distances=np.empty(0, dtype=float),
            weights=np.empty(0, dtype=float),
            sections=np.zeros(counts.shape[0], dtype=int),
            diagnostics={
                "construction": "not_built_for_rna_refined_control",
                "n_spots": int(counts.shape[0]),
                "n_edges": 0,
                "cross_section_edges": 0,
            },
        )
        graph = canonical_graph
        positive_edges = np.empty((0, 2), dtype=int)
        negative_edges = np.empty((0, 2), dtype=int)
        negative_diagnostics = {
            "candidate_pool": "not_used_by_rna_refined_control",
            "n_positive_edges": 0,
            "n_negative_edges": 0,
        }
    else:
        canonical_graph = build_spatial_graph(
            coordinates,
            section_labels=section_labels,
            n_neighbors=config.n_neighbors,
            local_scale_neighbor_rank=config.local_scale_neighbor_rank,
        )
        if config.null_model == "coordinate_permutation":
            graph = permuted_coordinate_graph(
                coordinates,
                section_labels=section_labels,
                n_neighbors=config.n_neighbors,
                local_scale_neighbor_rank=config.local_scale_neighbor_rank,
                random_state=config.random_state + 100003,
            )
        elif config.null_model == "degree_preserving":
            graph = degree_preserving_randomized_graph(
                canonical_graph, random_state=config.random_state + 200003
            )
        else:
            graph = canonical_graph

        positive_edges = graph.edges.copy()
        if config.method == "affinity_v3":
            negative_edges, negative_diagnostics = sample_local_negative_edges(
                graph,
                ratio=config.negative_ratio,
                local_radius_multiplier=config.local_radius_multiplier,
                random_state=config.random_state + 300007,
                return_diagnostics=True,
            )
            if config.null_model == "edge_label_permutation":
                positive_edges, negative_edges = permute_edge_labels(
                    positive_edges,
                    negative_edges,
                    random_state=config.random_state + 400009,
                )
        else:
            negative_edges = np.empty((0, 2), dtype=int)
            negative_diagnostics = {
                "candidate_pool": f"not_used_by_{config.method}",
                "n_positive_edges": int(positive_edges.shape[0]),
                "n_negative_edges": 0,
                "negative_ratio": 0.0,
            }
    graph_finished = time.perf_counter()

    if config.method != "rna_refined_control" and graph.edges.size == 0:
        return _identity_fit(
            base_fit,
            graph,
            negative_edges,
            config,
            "graph_has_no_edges",
            counts,
            time.perf_counter() - started,
        )

    composition = row_normalize(counts)
    proportions = project_rows_to_lower_simplex(base_fit.proportions)
    profiles = project_rows_to_lower_simplex(base_fit.profiles)
    base_proportions = proportions.copy()
    initial_rna = conditional_kl(composition, proportions, profiles)
    affinity: np.ndarray | None = None
    bias: float | None = None
    prefit: dict[str, object] | None = None
    affinity_prefit_started = time.perf_counter()
    if config.method == "affinity_v3":
        affinity, bias, prefit = fit_affinity_block(
            proportions,
            positive_edges,
            negative_edges,
            config.affinity_ridge,
            max_iter=config.affinity_prefit_max_iter,
            tolerance=config.affinity_optimizer_gradient_tolerance,
            function_tolerance=config.affinity_optimizer_function_tolerance,
        )
    affinity_prefit_finished = time.perf_counter()

    rna_reference = _rna_reference(composition, proportions, profiles)
    if config.method == "rna_refined_control":
        spatial_reference = {
            "reference_rms": 0.0,
            "raw_rms": 0.0,
            "tangent_rms": 0.0,
        }
        spatial_lambda = 0.0
        calibration: dict[str, float | str] = {
            "status": "rna_only_optimization_matched_control",
            "lambda": 0.0,
        }
    else:
        spatial_reference = _spatial_reference(
            config.method,
            proportions,
            graph,
            delta=config.huber_delta,
            affinity=affinity,
            bias=bias,
            positive_edges=positive_edges,
            negative_edges=negative_edges,
            ridge=config.affinity_ridge,
        )
        spatial_lambda, calibration = calibrate_spatial_lambda(
            config.gradient_ratio, rna_reference, spatial_reference
        )
    if spatial_lambda == 0.0 and config.method != "rna_refined_control":
        identity = _identity_fit(
            base_fit,
            graph,
            negative_edges,
            config,
            str(calibration["status"]),
            counts,
            time.perf_counter() - started,
        )
        identity.diagnostics["gradient_calibration"] = calibration
        identity.diagnostics["affinity_prefit"] = prefit
        return identity

    objective, data_value, spatial_value, pieces = _objective(
        composition,
        proportions,
        profiles,
        config.method,
        graph,
        spatial_lambda,
        delta=config.huber_delta,
        affinity=affinity,
        bias=bias,
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        ridge=config.affinity_ridge,
    )
    initial_objective = objective
    history: list[dict[str, object]] = []
    # Both simplex blocks use the same monotone projected-gradient contract.
    # Euclidean projection is intentional: unlike multiplicative or entropic
    # updates it can move mass into a coordinate currently on the lower bound.
    # Carried/BB step sizes are proposals only; the actual complete objective
    # and RNA-sacrifice guard decide every accepted step through Armijo.
    step_p = math.nan
    step_b = math.nan
    backtracks = 0
    failed_steps = 0
    plateau = 0
    converged = False
    termination_reason = "maximum_iterations_reached"
    rna_limit = initial_rna * (1.0 + config.rna_sacrifice_cap) + 1e-12

    _, initial_gradient_p_rna, initial_gradient_b = conditional_kl_and_gradients(
        composition, proportions, profiles
    )
    _, initial_gradient_p_spatial, _ = _unit_spatial_objective(
        config.method,
        proportions,
        graph,
        delta=config.huber_delta,
        affinity=affinity,
        bias=bias,
        positive_edges=positive_edges,
        negative_edges=negative_edges,
        ridge=config.affinity_ridge,
    )
    initial_full_gradient_p = (
        initial_gradient_p_rna + spatial_lambda * initial_gradient_p_spatial
    )
    initial_mapping_p = projected_gradient_mapping(
        proportions, initial_full_gradient_p
    )
    initial_mapping_b = projected_gradient_mapping(profiles, initial_gradient_b)
    initial_mapping_b_rms = _rms(initial_mapping_b)
    initial_mapping_b_max = float(np.max(np.abs(initial_mapping_b), initial=0.0))
    initial_natural_b = _mirror_stationarity(profiles, initial_gradient_b)
    initial_mapping_p_rms = _rms(initial_mapping_p)
    initial_mapping_p_max = float(np.max(np.abs(initial_mapping_p), initial=0.0))
    initial_mapping_rms = float(
        np.sqrt(
            (initial_mapping_p_rms**2 + initial_mapping_b_rms**2) / 2.0
        )
    )
    optimization_started = time.perf_counter()

    for iteration in range(1, config.max_iter + 1):
        iteration_start = objective
        moved = False
        b_accepted = False
        p_accepted = False
        a_accepted: bool | None = None

        # Optional legacy B block: monotone lower-simplex projected gradient.
        # The default logit L-BFGS block is applied *after* P below, so its KKT
        # residual is measured against the current proportions rather than the
        # proportions from the previous outer cycle.
        profile_block_info: dict[str, object] | None = None
        _, _, gradient_b = conditional_kl_and_gradients(composition, proportions, profiles)
        block_start_b_mapping_rms = _rms(
            projected_gradient_mapping(profiles, gradient_b)
        )
        accepted_inner_b = 0
        previous_b: np.ndarray | None = None
        previous_gradient_b: np.ndarray | None = None
        profile_pg_steps = (
            config.profile_pg_inner_steps
            if config.profile_solver == "projected_gradient"
            else 0
        )
        for inner in range(profile_pg_steps):
            _, _, gradient_b = conditional_kl_and_gradients(
                composition, proportions, profiles
            )
            current_b_mapping_rms = _rms(
                projected_gradient_mapping(profiles, gradient_b)
            )
            if (
                inner > 0
                and current_b_mapping_rms
                <= config.inner_stationarity_fraction
                * max(block_start_b_mapping_rms, EPS)
            ):
                break
            if previous_b is not None and previous_gradient_b is not None:
                proposal_b = _safe_bb_step(
                    previous_b,
                    profiles,
                    previous_gradient_b,
                    gradient_b,
                    fallback=(
                        step_b * config.projected_gradient_step_growth
                        if np.isfinite(step_b)
                        else _initial_projected_step(
                            gradient_b,
                            config.projected_gradient_initial_change,
                            config.projected_gradient_min_step,
                            config.projected_gradient_max_step,
                        )
                    ),
                    minimum=config.projected_gradient_min_step,
                    maximum=config.projected_gradient_max_step,
                )
            elif np.isfinite(step_b):
                proposal_b = min(
                    step_b * config.projected_gradient_step_growth,
                    config.projected_gradient_max_step,
                )
            else:
                proposal_b = _initial_projected_step(
                    gradient_b,
                    config.projected_gradient_initial_change,
                    config.projected_gradient_min_step,
                    config.projected_gradient_max_step,
                )
            local_step_b = float(proposal_b)
            inner_accepted = False
            old_b = profiles.copy()
            old_gradient_b = gradient_b.copy()
            for _ in range(config.projected_gradient_max_backtracks):
                candidate_b = project_rows_to_lower_simplex(
                    profiles - local_step_b * gradient_b,
                    lower_bound=SIMPLEX_LOWER_BOUND,
                )
                displacement_b = candidate_b - profiles
                directional_b = float(np.sum(gradient_b * displacement_b))
                candidate_data = conditional_kl(
                    composition, proportions, candidate_b
                )
                candidate_objective = float(
                    candidate_data + spatial_lambda * spatial_value
                )
                if (
                    directional_b <= 0.0
                    and candidate_objective
                    <= objective
                    + config.projected_gradient_armijo * directional_b
                    + 1e-12 * max(abs(objective), 1.0)
                    and candidate_data <= rna_limit
                ):
                    moved |= bool(
                        np.max(np.abs(displacement_b), initial=0.0) > 1e-12
                    )
                    profiles = candidate_b
                    data_value = float(candidate_data)
                    objective = candidate_objective
                    step_b = local_step_b
                    accepted_inner_b += 1
                    inner_accepted = True
                    previous_b = old_b
                    previous_gradient_b = old_gradient_b
                    break
                local_step_b *= 0.5
                backtracks += 1
            if not inner_accepted:
                break
        b_accepted = accepted_inner_b > 0

        # P block: the same monotone exact-lower-simplex scheme against the
        # complete RNA + spatial objective.
        _, gradient_p_rna, _ = conditional_kl_and_gradients(composition, proportions, profiles)
        _, gradient_p_spatial, _ = _unit_spatial_objective(
            config.method, proportions, graph, delta=config.huber_delta,
            affinity=affinity, bias=bias, positive_edges=positive_edges,
            negative_edges=negative_edges, ridge=config.affinity_ridge,
        )
        full_gradient_p = gradient_p_rna + spatial_lambda * gradient_p_spatial
        block_start_p_mapping_rms = _rms(
            projected_gradient_mapping(proportions, full_gradient_p)
        )
        accepted_inner_p = 0
        previous_p: np.ndarray | None = None
        previous_gradient_p: np.ndarray | None = None
        for inner in range(config.proportion_pg_inner_steps):
            _, gradient_p_rna, _ = conditional_kl_and_gradients(
                composition, proportions, profiles
            )
            _, gradient_p_spatial, _ = _unit_spatial_objective(
                config.method,
                proportions,
                graph,
                delta=config.huber_delta,
                affinity=affinity,
                bias=bias,
                positive_edges=positive_edges,
                negative_edges=negative_edges,
                ridge=config.affinity_ridge,
            )
            full_gradient_p = gradient_p_rna + spatial_lambda * gradient_p_spatial
            current_p_mapping_rms = _rms(
                projected_gradient_mapping(proportions, full_gradient_p)
            )
            if (
                inner > 0
                and current_p_mapping_rms
                <= config.inner_stationarity_fraction
                * max(block_start_p_mapping_rms, EPS)
            ):
                break
            if previous_p is not None and previous_gradient_p is not None:
                proposal_p = _safe_bb_step(
                    previous_p,
                    proportions,
                    previous_gradient_p,
                    full_gradient_p,
                    fallback=(
                        step_p * config.projected_gradient_step_growth
                        if np.isfinite(step_p)
                        else _initial_projected_step(
                            full_gradient_p,
                            config.projected_gradient_initial_change,
                            config.projected_gradient_min_step,
                            config.projected_gradient_max_step,
                        )
                    ),
                    minimum=config.projected_gradient_min_step,
                    maximum=config.projected_gradient_max_step,
                )
            elif np.isfinite(step_p):
                proposal_p = min(
                    step_p * config.projected_gradient_step_growth,
                    config.projected_gradient_max_step,
                )
            else:
                proposal_p = _initial_projected_step(
                    full_gradient_p,
                    config.projected_gradient_initial_change,
                    config.projected_gradient_min_step,
                    config.projected_gradient_max_step,
                )
            local_step_p = float(proposal_p)
            inner_accepted = False
            old_p = proportions.copy()
            old_gradient_p = full_gradient_p.copy()
            for _ in range(config.projected_gradient_max_backtracks):
                trial_p = project_rows_to_lower_simplex(
                    proportions - local_step_p * full_gradient_p,
                    lower_bound=SIMPLEX_LOWER_BOUND,
                )
                displacement_p = trial_p - proportions
                trial_objective, trial_data, trial_spatial, trial_pieces = _objective(
                    composition,
                    trial_p,
                    profiles,
                    config.method,
                    graph,
                    spatial_lambda,
                    delta=config.huber_delta,
                    affinity=affinity,
                    bias=bias,
                    positive_edges=positive_edges,
                    negative_edges=negative_edges,
                    ridge=config.affinity_ridge,
                )
                directional_p = float(np.sum(full_gradient_p * displacement_p))
                if (
                    directional_p <= 0.0
                    and trial_objective
                    <= objective
                    + config.projected_gradient_armijo * directional_p
                    + 1e-12 * max(abs(objective), 1.0)
                    and trial_data <= rna_limit
                ):
                    moved |= bool(
                        np.max(np.abs(displacement_p), initial=0.0) > 1e-12
                    )
                    proportions = trial_p
                    objective, data_value, spatial_value, pieces = (
                        trial_objective,
                        trial_data,
                        trial_spatial,
                        trial_pieces,
                    )
                    step_p = local_step_p
                    accepted_inner_p += 1
                    inner_accepted = True
                    previous_p = old_p
                    previous_gradient_p = old_gradient_p
                    break
                local_step_p *= 0.5
                backtracks += 1
            if not inner_accepted:
                break
        p_accepted = accepted_inner_p > 0
        if not p_accepted:
            failed_steps += 1

        # Default B block: for fixed current P this is a convex problem.  The
        # diagonal-Newton weighted projection handles active bounds directly;
        # logit L-BFGS remains an explicit diagnostic alternative.  Post-P
        # placement is essential for a meaningful final B KKT check.
        if config.profile_solver == "projected_gradient":
            before_b = profiles.copy()
            trial_b, profile_block_info = fit_profile_block_projected_gradient(
                composition,
                proportions,
                before_b,
                max_iter=config.profile_pg_post_inner_steps,
                initial_change=config.projected_gradient_initial_change,
                step_growth=config.projected_gradient_step_growth,
                minimum_step=config.projected_gradient_min_step,
                maximum_step=config.projected_gradient_max_step,
                armijo=config.projected_gradient_armijo,
                max_backtracks=config.projected_gradient_max_backtracks,
            )
            trial_data = conditional_kl(composition, proportions, trial_b)
            trial_objective = float(trial_data + spatial_lambda * spatial_value)
            post_b_accepted = bool(
                profile_block_info["accepted"]
                and trial_objective
                <= objective + 1e-12 * max(abs(objective), 1.0)
                and trial_data <= rna_limit
            )
            if post_b_accepted:
                displacement_b = trial_b - before_b
                moved |= bool(
                    np.max(np.abs(displacement_b), initial=0.0) > 1e-12
                )
                profiles = trial_b
                data_value = float(trial_data)
                objective = trial_objective
                accepted_inner_b += int(profile_block_info["iterations"])
                step_b = float(profile_block_info["final_step"])
                b_accepted = b_accepted or bool(
                    int(profile_block_info["iterations"]) > 0
                )
            else:
                failed_steps += 1
        elif config.profile_solver in ("lbfgs_logits", "diagonal_newton"):
            before_b = profiles.copy()
            _, _, before_gradient_b = conditional_kl_and_gradients(
                composition, proportions, before_b
            )
            if config.profile_solver == "lbfgs_logits":
                trial_b, profile_block_info = fit_profile_block_logits(
                    composition,
                    proportions,
                    before_b,
                    max_iter=config.profile_lbfgs_max_iter,
                    gradient_tolerance=config.profile_lbfgs_gradient_tolerance,
                    function_tolerance=config.profile_lbfgs_function_tolerance,
                    seed_mass=config.profile_logit_seed_mass,
                )
            else:
                trial_b, profile_block_info = fit_profile_block_diagonal_newton(
                    composition,
                    proportions,
                    before_b,
                    max_iter=config.profile_diagonal_newton_max_iter,
                    curvature_floor=config.profile_curvature_floor,
                    initial_change=config.projected_gradient_initial_change,
                    armijo=config.projected_gradient_armijo,
                    max_backtracks=config.projected_gradient_max_backtracks,
                    stationarity_fraction=config.profile_block_stationarity_fraction,
                )
            trial_data = conditional_kl(composition, proportions, trial_b)
            trial_objective = float(trial_data + spatial_lambda * spatial_value)
            b_accepted = bool(
                profile_block_info["accepted"]
                and trial_objective
                <= objective + 1e-12 * max(abs(objective), 1.0)
                and trial_data <= rna_limit
            )
            if b_accepted:
                displacement_b = trial_b - before_b
                moved |= bool(
                    np.max(np.abs(displacement_b), initial=0.0) > 1e-12
                )
                profiles = trial_b
                data_value = float(trial_data)
                objective = trial_objective
                accepted_inner_b = 1
                gradient_norm = float(np.linalg.norm(before_gradient_b))
                step_b = float(
                    np.linalg.norm(displacement_b) / max(gradient_norm, EPS)
                )
            else:
                failed_steps += 1

        if config.profile_solver == "projected_gradient" and not b_accepted:
            failed_steps += 1

        # A/b block is convex for fixed P.  The candidate is kept only if it
        # decreases the same total objective used by the P/B line searches.
        if (
            config.method == "affinity_v3"
            and affinity is not None
            and bias is not None
            and (
                iteration % config.affinity_update_interval == 0
                or iteration == config.max_iter
            )
        ):
            trial_a, trial_bias, a_info = fit_affinity_block(
                proportions,
                positive_edges,
                negative_edges,
                config.affinity_ridge,
                initial_affinity=affinity,
                initial_bias=bias,
                max_iter=config.affinity_block_max_iter,
                tolerance=config.affinity_optimizer_gradient_tolerance,
                function_tolerance=config.affinity_optimizer_function_tolerance,
            )
            trial_objective, trial_data, trial_spatial, trial_pieces = _objective(
                composition,
                proportions,
                profiles,
                config.method,
                graph,
                spatial_lambda,
                delta=config.huber_delta,
                affinity=trial_a,
                bias=trial_bias,
                positive_edges=positive_edges,
                negative_edges=negative_edges,
                ridge=config.affinity_ridge,
            )
            a_accepted = bool(
                trial_objective <= objective + 1e-12 * max(abs(objective), 1.0)
            )
            if a_accepted:
                moved |= bool(
                    np.max(np.abs(trial_a - affinity)) > 1e-12
                    or abs(trial_bias - bias) > 1e-12
                )
                affinity, bias = trial_a, trial_bias
                objective, data_value, spatial_value, pieces = (
                    trial_objective,
                    trial_data,
                    trial_spatial,
                    trial_pieces,
                )
            else:
                failed_steps += 1
        else:
            a_info = None

        # Recompute the KKT residual at the *current* P/B/A state.  This is a
        # true projected-gradient mapping on the lower-bounded simplexes and
        # is unaffected by adding a constant (such as 2 log 2) to the loss.
        _, current_gradient_p_rna, current_gradient_b = conditional_kl_and_gradients(
            composition, proportions, profiles
        )
        _, current_gradient_p_spatial, _ = _unit_spatial_objective(
            config.method,
            proportions,
            graph,
            delta=config.huber_delta,
            affinity=affinity,
            bias=bias,
            positive_edges=positive_edges,
            negative_edges=negative_edges,
            ridge=config.affinity_ridge,
        )
        current_full_gradient_p = (
            current_gradient_p_rna + spatial_lambda * current_gradient_p_spatial
        )
        mapping_p = projected_gradient_mapping(
            proportions, current_full_gradient_p
        )
        mapping_p_rms = _rms(mapping_p)
        mapping_p_max = float(np.max(np.abs(mapping_p), initial=0.0))
        mapping_b = projected_gradient_mapping(profiles, current_gradient_b)
        mapping_b_rms = _rms(mapping_b)
        mapping_b_max = float(np.max(np.abs(mapping_b), initial=0.0))
        natural_b_rms = _mirror_stationarity(profiles, current_gradient_b)
        mapping_p_ratio = mapping_p_rms / max(initial_mapping_p_rms, EPS)
        mapping_b_ratio = mapping_b_rms / max(initial_mapping_b_rms, EPS)
        mapping_p_max_ratio = mapping_p_max / max(initial_mapping_p_max, EPS)
        mapping_b_max_ratio = mapping_b_max / max(initial_mapping_b_max, EPS)
        natural_b_ratio = natural_b_rms / max(initial_natural_b, EPS)
        mapping_ratio = max(mapping_p_ratio, mapping_b_ratio)
        projected_gradient_rms = float(
            np.sqrt((mapping_p_rms**2 + mapping_b_rms**2) / 2.0)
        )
        natural_gradient_rms = float(
            np.sqrt(
                (
                    _mirror_stationarity(proportions, current_full_gradient_p) ** 2
                    * proportions.size
                    + _mirror_stationarity(profiles, current_gradient_b) ** 2
                    * profiles.size
                )
                / (proportions.size + profiles.size)
            )
        )
        if affinity is not None and bias is not None:
            _, _, current_gradient_a, current_gradient_bias, _ = affinity_loss_gradients(
                proportions,
                affinity,
                bias,
                positive_edges,
                negative_edges,
                ridge=config.affinity_ridge,
            )
            affinity_gradient_norm = float(
                np.linalg.norm(
                    _pack_symmetric_gradient(
                        current_gradient_a, current_gradient_bias
                    )
                )
            )
        else:
            affinity_gradient_norm = 0.0
        p_kkt_rms_pass = bool(
            mapping_p_rms
            <= max(
                config.p_kkt_absolute_tolerance,
                config.stationarity_fraction * initial_mapping_p_rms,
            )
        )
        p_kkt_max_pass = bool(
            mapping_p_max
            <= max(
                config.p_kkt_max_absolute_tolerance,
                config.stationarity_fraction * initial_mapping_p_max,
            )
        )
        b_kkt_rms_pass = bool(
            mapping_b_rms
            <= max(
                config.b_kkt_absolute_tolerance,
                config.stationarity_fraction * initial_mapping_b_rms,
            )
        )
        b_kkt_max_pass = bool(
            mapping_b_max
            <= max(
                config.b_kkt_max_absolute_tolerance,
                config.stationarity_fraction * initial_mapping_b_max,
            )
        )
        p_kkt_pass = p_kkt_rms_pass and p_kkt_max_pass
        b_kkt_pass = b_kkt_rms_pass and b_kkt_max_pass
        a_kkt_pass = bool(
            affinity_gradient_norm <= config.affinity_gradient_tolerance
        )
        all_kkt_pass = p_kkt_pass and b_kkt_pass and a_kkt_pass
        relative_change = abs(iteration_start - objective) / max(initial_rna, EPS)
        plateau = plateau + 1 if relative_change <= config.tolerance else 0
        positive_l1, positive_l2 = _mean_edge_distance(proportions, positive_edges)
        negative_l1, negative_l2 = _mean_edge_distance(proportions, negative_edges)
        component_mass = proportions.mean(axis=0)
        history.append(
            {
                "iteration": int(iteration),
                "objective": float(objective),
                "conditional_rna_kl": float(data_value),
                "spatial_unit_loss": float(spatial_value),
                "weighted_spatial_loss": float(spatial_lambda * spatial_value),
                "relative_objective_change": float(relative_change),
                "projected_gradient_rms": float(projected_gradient_rms),
                "projected_gradient_ratio_to_initial": float(mapping_ratio),
                "P_projected_gradient_mapping_rms": float(mapping_p_rms),
                "P_projected_gradient_ratio_to_initial": float(mapping_p_ratio),
                "P_projected_gradient_mapping_max": float(mapping_p_max),
                "P_projected_gradient_max_ratio_to_initial": float(
                    mapping_p_max_ratio
                ),
                "B_projected_gradient_mapping_rms": float(mapping_b_rms),
                "B_projected_gradient_ratio_to_initial": float(mapping_b_ratio),
                "B_projected_gradient_mapping_max": float(mapping_b_max),
                "B_projected_gradient_max_ratio_to_initial": float(
                    mapping_b_max_ratio
                ),
                "B_natural_gradient_rms": float(natural_b_rms),
                "B_natural_gradient_ratio_to_initial": float(natural_b_ratio),
                "affinity_packed_gradient_norm": float(affinity_gradient_norm),
                "P_KKT_pass": bool(p_kkt_pass),
                "P_KKT_RMS_pass": bool(p_kkt_rms_pass),
                "P_KKT_max_pass": bool(p_kkt_max_pass),
                "B_KKT_pass": bool(b_kkt_pass),
                "B_KKT_RMS_pass": bool(b_kkt_rms_pass),
                "B_KKT_max_pass": bool(b_kkt_max_pass),
                "A_KKT_pass": bool(a_kkt_pass),
                "all_KKT_pass": bool(all_kkt_pass),
                "natural_gradient_rms": float(natural_gradient_rms),
                "step_p": float(step_p),
                "step_b": float(step_b),
                "accepted_p": bool(p_accepted),
                "accepted_b": bool(b_accepted),
                "accepted_inner_p": int(accepted_inner_p),
                "accepted_inner_b": int(accepted_inner_b),
                "profile_solver": str(config.profile_solver),
                "profile_block_iterations": (
                    int(profile_block_info["iterations"])
                    if profile_block_info is not None
                    else int(accepted_inner_b)
                ),
                "profile_block_function_evaluations": (
                    int(profile_block_info["function_evaluations"])
                    if profile_block_info is not None
                    else int(accepted_inner_b)
                ),
                "profile_block_projected_mapping_rms": (
                    float(profile_block_info["projected_mapping_rms"])
                    if profile_block_info is not None
                    else float(mapping_b_rms)
                ),
                "profile_block_projected_mapping_max": (
                    float(profile_block_info["projected_mapping_max"])
                    if profile_block_info is not None
                    else float(mapping_b_max)
                ),
                "accepted_a": a_accepted,
                "cumulative_backtracks": int(backtracks),
                "positive_edge_tvd": positive_l1,
                "negative_edge_tvd": negative_l1,
                "positive_edge_l2": positive_l2,
                "negative_edge_l2": negative_l2,
                "spot_variance": float(
                    np.mean(np.square(proportions - proportions.mean(axis=0)))
                ),
                "component_mass_min": float(component_mass.min()),
                "component_mass_max": float(component_mass.max()),
                "affinity_frobenius_norm": (
                    float(np.linalg.norm(affinity)) if affinity is not None else math.nan
                ),
                "affinity_bias": float(bias) if bias is not None else math.nan,
                **{f"spatial_{key}": float(value) for key, value in pieces.items()},
                "affinity_block_iterations": (
                    int(a_info["iterations"]) if a_info is not None else 0
                ),
            }
        )
        affinity_current_for_stop = bool(
            config.method != "affinity_v3" or a_info is not None
        )
        if not moved:
            # Do not terminate with an A block that predates the final P step.
            # If necessary, carry the stationary P/B state to the next due A
            # update (or the forced max-iteration update above).
            if not affinity_current_for_stop and iteration < config.max_iter:
                continue
            converged = bool(all_kkt_pass)
            termination_reason = (
                "projected_stationarity"
                if all_kkt_pass
                else "projected_line_search_failed_before_stationarity"
            )
            break
        if (
            iteration >= config.minimum_iterations
            and plateau >= config.patience
            and all_kkt_pass
            and affinity_current_for_stop
        ):
            converged = True
            termination_reason = "objective_and_projected_kkt_tolerance_met"
            break

    final_rna = conditional_kl(composition, proportions, profiles)
    positive_l1, positive_l2 = _mean_edge_distance(proportions, positive_edges)
    negative_l1, negative_l2 = _mean_edge_distance(proportions, negative_edges)
    collapse = _collapse_diagnostics(
        proportions,
        profiles,
        base_proportions,
        initial_rna,
        final_rna,
        config.rna_sacrifice_cap,
        affinity,
        bias,
        positive_edges,
        negative_edges,
    )
    diagnostics: dict[str, object] = {
        "method": config.method,
        "model": {
            "rna_refined_control": "optimization-matched conditional KL control",
            "laplacian_v3": "conditional KL plus weighted graph Laplacian",
            "huber_tv_v3": "conditional KL plus component-wise pseudo-Huber graph TV",
            "affinity_v3": "conditional KL plus learned symmetric signed edge affinity",
        }[config.method],
        "identity_return": False,
        "iterations": int(len(history)),
        "converged": bool(converged),
        "termination_reason": termination_reason,
        "initial_objective": float(initial_objective),
        "final_objective": float(objective),
        "initial_conditional_kl": float(initial_rna),
        "conditional_kl": float(final_rna),
        "conditional_kl_relative_change": float(
            (final_rna - initial_rna) / max(initial_rna, EPS)
        ),
        "raw_count_poisson_deviance_per_mass": raw_count_poisson_deviance(
            counts, proportions, profiles
        ),
        "spatial_lambda": float(spatial_lambda),
        "affinity_ridge_inside_spatial_term": float(config.affinity_ridge),
        "equivalent_absolute_lambda_A": float(
            spatial_lambda * config.affinity_ridge
        ),
        "gradient_calibration": calibration,
        "affinity_prefit": prefit,
        "graph": graph.diagnostics,
        "canonical_graph": canonical_graph.diagnostics,
        "negative_sampling": negative_diagnostics,
        "positive_edge_tvd": positive_l1,
        "negative_edge_tvd": negative_l1,
        "positive_edge_l2": positive_l2,
        "negative_edge_l2": negative_l2,
        "neighbor_nonedge_tvd_gap": negative_l1 - positive_l1,
        "line_search_backtracks": int(backtracks),
        "failed_block_steps": int(failed_steps),
        "maximum_relative_objective_increase": float(
            max(
                [
                    (right["objective"] - left["objective"])
                    / max(abs(float(left["objective"])), EPS)
                    for left, right in zip(history, history[1:])
                ]
                or [0.0]
            )
        ),
        "runtime_sec": float(time.perf_counter() - started),
        "timing": {
            "validation_sec": float(validation_finished - started),
            "v2_fit_sec": float(v2_finished - v2_started),
            "graph_preprocessing_sec": float(graph_finished - graph_started),
            "affinity_prefit_sec": float(
                affinity_prefit_finished - affinity_prefit_started
            ),
            "alternating_optimization_sec": float(
                time.perf_counter() - optimization_started
            ),
            "total_sec": float(time.perf_counter() - started),
        },
        "initial_projected_gradient_mapping_rms": float(initial_mapping_rms),
        "initial_P_projected_gradient_mapping_rms": float(initial_mapping_p_rms),
        "initial_P_projected_gradient_mapping_max": float(initial_mapping_p_max),
        "initial_B_projected_gradient_mapping_rms": float(initial_mapping_b_rms),
        "initial_B_projected_gradient_mapping_max": float(initial_mapping_b_max),
        "final_projected_gradient_mapping_rms": (
            float(history[-1]["projected_gradient_rms"]) if history else math.nan
        ),
        "final_projected_gradient_ratio_to_initial": (
            float(history[-1]["projected_gradient_ratio_to_initial"])
            if history
            else math.nan
        ),
        "final_P_projected_gradient_mapping_rms": (
            float(history[-1]["P_projected_gradient_mapping_rms"])
            if history
            else math.nan
        ),
        "final_P_projected_gradient_mapping_max": (
            float(history[-1]["P_projected_gradient_mapping_max"])
            if history
            else math.nan
        ),
        "final_B_projected_gradient_mapping_rms": (
            float(history[-1]["B_projected_gradient_mapping_rms"])
            if history
            else math.nan
        ),
        "final_B_projected_gradient_mapping_max": (
            float(history[-1]["B_projected_gradient_mapping_max"])
            if history
            else math.nan
        ),
        "final_P_KKT_pass": bool(history[-1]["P_KKT_pass"]) if history else False,
        "final_B_KKT_pass": bool(history[-1]["B_KKT_pass"]) if history else False,
        "final_A_KKT_pass": bool(history[-1]["A_KKT_pass"]) if history else False,
        "final_all_KKT_pass": bool(history[-1]["all_KKT_pass"]) if history else False,
        "base_v2": base_fit.diagnostics,
        "truth_used": False,
        "annotation_used": False,
        "single_cell_reference_used": False,
        "other_method_prediction_used": False,
        "gene_side_penalty_active": False,
        "solvability_penalty_active": False,
        **collapse,
    }
    return DualGeometryFit(
        proportions=proportions,
        profiles=profiles,
        affinity=affinity,
        bias=bias,
        positive_edges=np.asarray(positive_edges, dtype=int).copy(),
        negative_edges=np.asarray(negative_edges, dtype=int).copy(),
        history=history,
        diagnostics=diagnostics,
        config=config.to_dict(),
        graph=graph,
        base_fit=base_fit,
    )


# Short alias for interactive use; the long name is kept for benchmark clarity.
fit_spgsnmf_v3 = fit_spgsnmf_v3_dual_geometry


__all__ = [
    "DualGeometryFit",
    "EPS",
    "affinity_loss_gradients",
    "affinity_scores",
    "calibrate_spatial_lambda",
    "center_affinity",
    "conditional_kl",
    "conditional_kl_and_gradients",
    "entropic_mirror_candidate",
    "fit_affinity_block",
    "fit_profile_block_diagonal_newton",
    "fit_profile_block_logits",
    "fit_spgsnmf_v3",
    "fit_spgsnmf_v3_dual_geometry",
    "huber_tv_loss_gradient",
    "laplacian_loss_gradient",
    "projected_gradient_mapping",
    "profile_logits_objective_gradient",
    "project_rows_to_lower_simplex",
    "project_rows_to_weighted_lower_simplex",
    "raw_count_poisson_deviance",
]
