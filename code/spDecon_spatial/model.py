"""Core spDecon optimizer: exposure-aware Poisson/KL factorization with TV.

The spot-level workflow calls ``fit_spdecon`` to jointly update component
weights P and expression profiles B, with explicit graph-TV regularization.
The shared numerical routines also retain an optional HMRF mode; that mode
is disabled by the spot-level workflow's ``rna_only_poisson`` configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.stats import qmc
import torch

try:
    from v3_dual_geometry.graph import build_spatial_graph
except ModuleNotFoundError:
    from ..v3_dual_geometry.graph import build_spatial_graph  # type: ignore

from .config import PoissonHMRFConfig


EPS = 1e-12


def _tv_energy_gradient(p, tv):
    """Weighted smoothed graph TV; return its value and unweighted gradient."""
    left, right, weights, strength, epsilon = tv
    difference = p[left] - p[right]
    length = torch.sqrt(torch.sum(difference * difference, dim=1) + epsilon**2)
    value = torch.sum(weights * (length - epsilon))
    edge_gradient = (weights / length)[:, None] * difference
    gradient = torch.zeros_like(p)
    gradient.index_add_(0, left, edge_gradient)
    gradient.index_add_(0, right, -edge_gradient)
    return value, gradient


@dataclass
class PoissonHMRFFit:
    proportions: np.ndarray
    profiles: np.ndarray
    affinity: np.ndarray | None
    edges: np.ndarray
    history: list[dict[str, object]]
    diagnostics: dict[str, object]
    config: dict[str, object]


def _validate(
    counts: np.ndarray,
    coordinates: np.ndarray,
    n_components: int,
    initial_proportions: np.ndarray,
    initial_profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    p = np.asarray(initial_proportions, dtype=float)
    b = np.asarray(initial_profiles, dtype=float)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite nonnegative matrix")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("counts contain an empty spot")
    if coordinates.ndim != 2 or coordinates.shape[0] != counts.shape[0] or coordinates.shape[1] < 2:
        raise ValueError("coordinates must provide two columns per spot")
    if np.any(~np.isfinite(coordinates[:, :2])):
        raise ValueError("coordinates contain nonfinite values")
    if p.shape != (counts.shape[0], n_components):
        raise ValueError("initial proportions have an incompatible shape")
    if b.shape != (n_components, counts.shape[1]):
        raise ValueError("initial profiles have an incompatible shape")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(~np.isfinite(b)) or np.any(b < 0):
        raise ValueError("initial factors must be finite and nonnegative")
    p = p / np.maximum(p.sum(axis=1, keepdims=True), EPS)
    b = b / np.maximum(b.sum(axis=1, keepdims=True), EPS)
    return counts, coordinates[:, :2], p, b


def _torch_context(config: PoissonHMRFConfig) -> tuple[torch.device, torch.dtype]:
    requested = config.device
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    return torch.device(requested), dtype


def project_rows_lower_simplex(
    values: torch.Tensor, lower_bound: float
) -> torch.Tensor:
    """Exact Euclidean row projection onto x>=lower_bound, sum(x)=1."""

    width = values.shape[1]
    available = 1.0 - width * lower_bound
    if available <= 0:
        raise ValueError("lower bound is incompatible with simplex width")
    shifted = (values - lower_bound) / available
    ordered = torch.sort(shifted, dim=1, descending=True).values
    cumulative = torch.cumsum(ordered, dim=1) - 1.0
    indices = torch.arange(1, width + 1, device=values.device, dtype=values.dtype)[None]
    support = ordered - cumulative / indices > 0
    support_size = torch.clamp(support.sum(dim=1), min=1)
    theta = cumulative.gather(1, (support_size - 1)[:, None]) / support_size[:, None]
    projected = torch.clamp(shifted - theta, min=0.0)
    return lower_bound + available * projected


def entropic_rows_lower_simplex(
    values: torch.Tensor,
    gradient: torch.Tensor,
    step: float,
    lower_bound: float,
) -> torch.Tensor:
    """Mirror-descent candidate that remains able to leave the profile floor."""

    width = values.shape[1]
    available = 1.0 - width * lower_bound
    interior = torch.clamp((values - lower_bound) / available, min=1e-300)
    interior = interior / interior.sum(dim=1, keepdim=True)
    logits = torch.log(interior) - float(step) * available * gradient
    logits = logits - logits.max(dim=1, keepdim=True).values
    candidate = torch.exp(torch.clamp(logits, min=-745.0))
    candidate = candidate / candidate.sum(dim=1, keepdim=True)
    return lower_bound + available * candidate


def _raw_poisson_kl_and_gradients(
    counts: torch.Tensor,
    exposure: torch.Tensor,
    total_count: torch.Tensor,
    entropy_constant: torch.Tensor,
    p: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw-count generalized KL and gradients, normalized per observed count.

    The count-only ``sum Y log(Y / s)`` contribution is cached by the caller.
    This avoids recomputing a masked logarithm over the full 360 x 20k matrix
    at every line-search/EM step.
    """

    mixture = torch.clamp(p @ b, min=EPS)
    # Keep the linear generalized-KL term explicit.  It cancels analytically
    # for exact row-simplex factors, but retaining it prevents float32
    # accumulation error across 20k features from biasing the convergence log.
    linear = torch.sum(exposure[:, None] * mixture) / total_count - 1.0
    loss = linear + entropy_constant - torch.sum(counts * torch.log(mixture)) / total_count
    residual = (exposure[:, None] - counts / mixture) / total_count
    return loss, residual @ b.T, p.T @ residual


def raw_poisson_kl_per_count(
    counts: np.ndarray, proportions: np.ndarray, profiles: np.ndarray
) -> float:
    counts = np.asarray(counts, dtype=float)
    exposure = counts.sum(axis=1)
    q = counts / exposure[:, None]
    mixture = np.maximum(np.asarray(proportions) @ np.asarray(profiles), EPS)
    terms = mixture - q
    positive = q > 0
    terms[positive] += q[positive] * np.log(q[positive] / mixture[positive])
    return float(np.sum(exposure[:, None] * terms) / exposure.sum())


def pairwise_energy_gradient_numpy(
    proportions: np.ndarray, affinity: np.ndarray, edges: np.ndarray
) -> tuple[float, np.ndarray]:
    p = np.asarray(proportions, dtype=float)
    a = np.asarray(affinity, dtype=float)
    e = np.asarray(edges, dtype=int)
    if len(e) == 0:
        return 0.0, np.zeros_like(p)
    left, right = e[:, 0], e[:, 1]
    value = float(np.einsum("ik,kl,il->", p[left], a, p[right]) / len(e))
    gradient = np.zeros_like(p)
    np.add.at(gradient, left, p[right] @ a.T / len(e))
    np.add.at(gradient, right, p[left] @ a / len(e))
    return value, gradient


def _pairwise_energy_gradient_torch(
    p: torch.Tensor, affinity: torch.Tensor, edges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if edges.numel() == 0:
        return torch.zeros((), device=p.device, dtype=p.dtype), torch.zeros_like(p)
    left, right = edges[:, 0], edges[:, 1]
    value = torch.sum((p[left] @ affinity) * p[right]) / len(edges)
    gradient = torch.zeros_like(p)
    gradient.index_add_(0, left, p[right] @ affinity.T / len(edges))
    gradient.index_add_(0, right, p[left] @ affinity / len(edges))
    return value, gradient


def _adjacency(edges: np.ndarray, n_spots: int) -> csr_matrix:
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_spots, n_spots))


def _coordinate_permuted_edges(
    edges: np.ndarray, n_spots: int, random_state: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Relabel a fixed coordinate graph after permuting spot-to-location assignment.

    Rebuilding kNN after permuting a regular grid changes tie-breaking and can
    change the edge count.  Relabeling the already-fixed graph is the exact
    coordinate-permutation null: topology, degree multiset, and edge count are
    preserved while the biological spot assigned to each location is random.
    """

    rng = np.random.default_rng(int(random_state))
    source_for_destination = rng.permutation(n_spots)
    if np.array_equal(source_for_destination, np.arange(n_spots)):
        source_for_destination = np.roll(source_for_destination, 1)
    destination_for_source = np.empty(n_spots, dtype=int)
    destination_for_source[source_for_destination] = np.arange(n_spots)
    permuted = destination_for_source[np.asarray(edges, dtype=int)]
    permuted.sort(axis=1)
    order = np.lexsort((permuted[:, 1], permuted[:, 0]))
    permuted = permuted[order]
    return permuted, {
        "null_model": "coordinate_assignment_permutation_on_fixed_graph",
        "random_state": int(random_state),
        "coordinate_permutation_moved": int(
            np.sum(source_for_destination != np.arange(n_spots))
        ),
        "coordinate_source_indices": source_for_destination.tolist(),
        "edge_count_preserved_exactly": True,
        "degree_multiset_preserved_exactly": True,
    }


def _quadrature_samples(k: int, power: int, seed: int) -> np.ndarray:
    uniforms = qmc.Sobol(d=k, scramble=True, seed=int(seed)).random_base2(power)
    exponentials = -np.log(np.clip(uniforms, 1e-12, 1.0))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _unpack_symmetric_centered(vector: np.ndarray, k: int) -> np.ndarray:
    upper = np.triu_indices(k)
    matrix = np.zeros((k, k), dtype=float)
    matrix[upper] = vector
    matrix[(upper[1], upper[0])] = vector
    return matrix - matrix.mean()


def _pack_symmetric_gradient(gradient: np.ndarray) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=float)
    gradient = gradient - gradient.mean()
    upper = np.triu_indices(len(gradient))
    packed = gradient[upper].copy()
    off = upper[0] != upper[1]
    packed[off] += gradient[(upper[1][off], upper[0][off])]
    return packed


def affinity_pseudolikelihood_value_gradient(
    vector: np.ndarray,
    proportions: np.ndarray,
    adjacency: csr_matrix,
    samples: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """SpiceMix node-conditional negative log pseudo-likelihood.

    ``log Z_i`` is evaluated as a deterministic uniform-simplex quadrature of
    ``exp(-z.T A nu_i)``.  No non-neighbor or negative edge is introduced.
    """

    p = np.asarray(proportions, dtype=float)
    k = p.shape[1]
    affinity = _unpack_symmetric_centered(np.asarray(vector, dtype=float), k)
    neighbor_sum = np.asarray(adjacency @ p, dtype=float)
    directed_edges = max(float(adjacency.sum()), 1.0)
    eta = neighbor_sum @ affinity
    logits = -(samples @ eta.T)
    maximum = logits.max(axis=0, keepdims=True)
    weights = np.exp(logits - maximum)
    normalizers = weights.sum(axis=0, keepdims=True)
    log_z = maximum.ravel() + np.log(normalizers.ravel()) - math.log(len(samples))
    expected_z = (weights.T @ samples) / normalizers.T
    observed = float(np.sum(p * eta))
    data_value = (observed + float(log_z.sum())) / directed_edges
    ridge_value = float(ridge * np.square(affinity).sum())
    gradient = (p.T @ neighbor_sum - expected_z.T @ neighbor_sum) / directed_edges
    gradient += 2.0 * ridge * affinity
    packed = _pack_symmetric_gradient(gradient)
    return data_value + ridge_value, packed, {
        "conditional_data_term": data_value,
        "ridge_term": ridge_value,
        "packed_gradient_max": float(np.max(np.abs(packed), initial=0.0)),
    }


def initialize_affinity(proportions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Official SpiceMix sign: negative standardized neighbor correlation."""

    directed = np.vstack((edges, edges[:, ::-1]))
    x = proportions[directed[:, 0]]
    y = proportions[directed[:, 1]]
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    sx = np.maximum(x.std(axis=0, keepdims=True), 1e-8)
    sy = np.maximum(y.std(axis=0, keepdims=True), 1e-8)
    corr = (y / sy).T @ (x / sx) / len(x)
    affinity = -5.0 * (corr + corr.T) / 2.0
    return affinity - affinity.mean()


def fit_affinity_pseudolikelihood(
    proportions: np.ndarray,
    edges: np.ndarray,
    initial_affinity: np.ndarray | None,
    config: PoissonHMRFConfig,
    samples: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    k = proportions.shape[1]
    adjacency = _adjacency(edges, len(proportions))
    if initial_affinity is None:
        initial_affinity = initialize_affinity(proportions, edges)
    upper = np.triu_indices(k)
    initial = np.asarray(initial_affinity, dtype=float)[upper]
    evaluations = 0
    last_details: dict[str, float] = {}

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations, last_details
        evaluations += 1
        value, gradient, last_details = affinity_pseudolikelihood_value_gradient(
            vector, proportions, adjacency, samples, config.affinity_ridge
        )
        return value, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(config.affinity_max_iterations),
            "gtol": float(config.affinity_gradient_tolerance),
            "ftol": 1e-12,
            "maxls": 50,
        },
    )
    affinity = _unpack_symmetric_centered(result.x, k)
    value, gradient, details = affinity_pseudolikelihood_value_gradient(
        result.x, proportions, adjacency, samples, config.affinity_ridge
    )
    return affinity, {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(evaluations),
        "value": float(value),
        "packed_gradient_max": float(np.max(np.abs(gradient), initial=0.0)),
        "conditional_data_term": details["conditional_data_term"],
        "ridge_term": details["ridge_term"],
        "affinity_frobenius_norm": float(np.linalg.norm(affinity)),
        "affinity_mean": float(affinity.mean()),
        "affinity_min": float(affinity.min()),
        "affinity_max": float(affinity.max()),
    }


def _mapping_max(values: torch.Tensor, gradient: torch.Tensor, lower: float) -> float:
    mapping = values - project_rows_lower_simplex(values - gradient, lower)
    return float(torch.max(torch.abs(mapping)).item())


def _initial_step(gradient: torch.Tensor, target: float, minimum: float, maximum: float) -> float:
    tangent = gradient - gradient.mean(dim=1, keepdim=True)
    scale = float(torch.max(torch.abs(tangent)).item())
    return float(np.clip(target / max(scale, EPS), minimum, maximum))


def _gradient_ratio_lambda(
    rna_gradient: torch.Tensor,
    hmrf_gradient: torch.Tensor,
    target_ratio: float,
) -> tuple[float, float, float]:
    """Calibrate the HMRF coefficient on simplex-tangent RMS gradients."""

    rna_tangent = rna_gradient - rna_gradient.mean(dim=1, keepdim=True)
    hmrf_tangent = hmrf_gradient - hmrf_gradient.mean(dim=1, keepdim=True)
    rna_rms = float(torch.sqrt(torch.mean(rna_tangent.square())).item())
    hmrf_rms = float(torch.sqrt(torch.mean(hmrf_tangent.square())).item())
    coefficient = float(target_ratio * rna_rms / max(hmrf_rms, EPS))
    return coefficient, rna_rms, hmrf_rms


def _optimize_block(
    block: str,
    counts: torch.Tensor,
    exposure: torch.Tensor,
    total_count: torch.Tensor,
    entropy_constant: torch.Tensor,
    p: torch.Tensor,
    b: torch.Tensor,
    affinity: torch.Tensor | None,
    edges: torch.Tensor,
    spatial_lambda: float,
    step: float | None,
    config: PoissonHMRFConfig,
    tv: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float | None, dict[str, object]]:
    if block not in ("P", "B"):
        raise ValueError("block must be P or B")
    maximum_steps = config.max_p_steps if block == "P" else config.max_b_steps
    tolerance = config.p_kkt_tolerance if block == "P" else config.b_kkt_tolerance
    x = p if block == "P" else b
    accepted = 0
    backtracks_total = 0
    last_gradient: torch.Tensor | None = None
    last_x: torch.Tensor | None = None

    def evaluate(
        candidate_p: torch.Tensor, candidate_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rna, gp, gb = _raw_poisson_kl_and_gradients(
            counts, exposure, total_count, entropy_constant, candidate_p, candidate_b
        )
        spatial = torch.zeros((), device=p.device, dtype=p.dtype)
        if affinity is not None and spatial_lambda > 0:
            spatial, spatial_gradient = _pairwise_energy_gradient_torch(
                candidate_p, affinity, edges
            )
            gp = gp + spatial_lambda * spatial_gradient
        objective = rna + spatial_lambda * spatial
        if tv is not None:
            tv_value, tv_gradient = _tv_energy_gradient(candidate_p, tv)
            objective = objective + tv[3] * tv_value
            gp = gp + tv[3] * tv_gradient
        return objective, rna, gp, gb

    if block == "B":
        # With P fixed, the raw-Poisson/simplex profile block is a mixture
        # maximum-likelihood problem.  This responsibility/EM update is
        # monotone and avoids the singular Euclidean gradient created by rare
        # ATAC peaks at the profile floor.
        objective, rna, gp, gb = evaluate(p, b)
        initial_kkt = _mapping_max(b, gb, config.lower_bound)
        final_kkt = initial_kkt
        accepted = 0
        available = 1.0 - b.shape[1] * config.lower_bound
        for iteration in range(maximum_steps):
            mixture = torch.clamp(p @ b, min=EPS)
            responsibility_mass = p.T @ (counts / mixture)
            raw = torch.clamp(b * responsibility_mass, min=1e-300)
            raw = raw / raw.sum(dim=1, keepdim=True)
            candidate = config.lower_bound + available * raw
            # Accumulation across 20k features in float32 can otherwise leave
            # a 1e-4 row-sum error, large enough to contaminate the KL/KKT
            # audit.  Renormalization is deterministic and preserves the
            # numerical positivity floor to float precision.
            candidate = candidate / candidate.sum(dim=1, keepdim=True)
            b = candidate
            accepted += 1
            if iteration < 10 or (iteration + 1) % config.kkt_check_interval == 0:
                objective, rna, gp, gb = evaluate(p, b)
                final_kkt = _mapping_max(b, gb, config.lower_bound)
                if final_kkt <= tolerance:
                    break
        objective, rna, gp, gb = evaluate(p, b)
        final_kkt = _mapping_max(b, gb, config.lower_bound)
        return p, b, None, {
            "accepted_steps": int(accepted),
            "backtracks": 0,
            "initial_kkt_max": float(initial_kkt),
            "final_kkt_max": float(final_kkt),
            "objective": float(objective.item()),
            "rna_kl_per_count": float(rna.item()),
            "solver": "monotone_poisson_mixture_EM",
        }

    objective, rna, gp, gb = evaluate(p, b)
    gradient = gp if block == "P" else gb
    if step is None or step <= config.minimum_step * 10.0:
        step = _initial_step(
            gradient,
            config.initial_projected_change,
            config.minimum_step,
            config.maximum_step,
        )
    initial_kkt = _mapping_max(x, gradient, config.lower_bound)
    final_kkt = initial_kkt
    for _ in range(maximum_steps):
        final_kkt = _mapping_max(x, gradient, config.lower_bound)
        if final_kkt <= tolerance:
            break
        trial_step = float(step)
        accepted_candidate = None
        for backtrack in range(config.max_backtracks + 1):
            candidate = (
                project_rows_lower_simplex(
                    x - trial_step * gradient, config.lower_bound
                )
                if block == "P"
                else entropic_rows_lower_simplex(
                    x, gradient, trial_step, config.lower_bound
                )
            )
            displacement = candidate - x
            if float(torch.max(torch.abs(displacement)).item()) <= 10.0 * torch.finfo(x.dtype).eps:
                restarted = _initial_step(
                    gradient,
                    config.initial_projected_change,
                    config.minimum_step,
                    config.maximum_step,
                )
                if restarted > trial_step * 2.0:
                    trial_step = restarted
                    continue
                break
            cp, cb = (candidate, b) if block == "P" else (p, candidate)
            candidate_values = evaluate(cp, cb)
            candidate_objective = candidate_values[0]
            armijo_bound = objective + config.armijo * torch.sum(gradient * displacement)
            if torch.isfinite(candidate_objective) and candidate_objective <= armijo_bound + 1e-14:
                accepted_candidate = (candidate, candidate_values)
                backtracks_total += backtrack
                break
            trial_step *= 0.5
            if trial_step < config.minimum_step:
                break
        if accepted_candidate is None:
            # A BB proposal can collapse to a numerically immobile step at a
            # high-dimensional simplex boundary.  Returning None forces a
            # scale-aware restart in the next alternating block rather than
            # falsely declaring a plateau.
            step = None
            break
        candidate, candidate_values = accepted_candidate
        new_gradient = candidate_values[2] if block == "P" else candidate_values[3]
        if block == "P" and last_gradient is not None and last_x is not None:
            s = candidate - x
            y = new_gradient - gradient
            denominator = float(torch.sum(s * y).item())
            if denominator > 0:
                step = float(
                    np.clip(
                        float(torch.sum(s * s).item()) / denominator,
                        config.minimum_step,
                        config.maximum_step,
                    )
                )
            else:
                step = min(trial_step * 1.5, config.maximum_step)
        else:
            step = min(trial_step * 1.5, config.maximum_step)
        last_x = x
        last_gradient = gradient
        x = candidate
        objective, rna, gp, gb = candidate_values
        gradient = new_gradient
        if block == "P":
            p = x
        else:
            b = x
        accepted += 1
    final_kkt = _mapping_max(x, gradient, config.lower_bound)
    return p, b, (None if step is None else float(step)), {
        "accepted_steps": int(accepted),
        "backtracks": int(backtracks_total),
        "initial_kkt_max": float(initial_kkt),
        "final_kkt_max": float(final_kkt),
        "objective": float(objective.item()),
        "rna_kl_per_count": float(rna.item()),
    }


def fit_spdecon(
    counts: np.ndarray,
    coordinates: np.ndarray,
    n_components: int,
    initial_proportions: np.ndarray,
    initial_profiles: np.ndarray,
    config: PoissonHMRFConfig | dict[str, Any] | None = None,
    *,
    tv_edges: np.ndarray | None = None,
    tv_weights: np.ndarray | None = None,
    tv_lambda: float = 0.0,
    tv_epsilon: float = 0.02,
) -> PoissonHMRFFit:
    """Jointly fit P and B with optional graph TV or HMRF regularization."""

    started = time.perf_counter()
    if config is None:
        config = PoissonHMRFConfig()
    elif isinstance(config, dict):
        config = PoissonHMRFConfig(**config)
    config.validate()
    counts, coordinates, p0, b0 = _validate(
        counts, coordinates, n_components, initial_proportions, initial_profiles
    )
    device, dtype = _torch_context(config)
    exposure_np = counts.sum(axis=1)
    composition_np = counts / exposure_np[:, None]
    p = project_rows_lower_simplex(
        torch.as_tensor(p0, device=device, dtype=dtype), config.lower_bound
    )
    b = project_rows_lower_simplex(
        torch.as_tensor(b0, device=device, dtype=dtype), config.lower_bound
    )
    counts_tensor = torch.as_tensor(counts, device=device, dtype=dtype)
    # Zero weight bypasses TV entirely and preserves the original numerical path.
    if not math.isfinite(tv_lambda) or tv_lambda < 0:
        raise ValueError("tv_lambda must be finite and nonnegative")
    tv = None
    if tv_lambda > 0:
        if config.method != "rna_only_poisson":
            raise ValueError("This experiment adds TV to the original RNA-only optimizer")
        te, tw = np.asarray(tv_edges), np.asarray(tv_weights, dtype=float)
        if te.ndim != 2 or te.shape[1] != 2 or not np.issubdtype(te.dtype, np.integer):
            raise ValueError("TV edges must be an integer (n_edges, 2) array")
        if tw.shape != (len(te),) or not len(te) or not np.isfinite(tw).all() or (tw < 0).any() or tw.sum() <= 0:
            raise ValueError("TV weights must be finite, nonnegative and have positive sum")
        if (te < 0).any() or (te >= len(p)).any() or (te[:, 0] == te[:, 1]).any():
            raise ValueError("Invalid TV edge indices")
        if not math.isfinite(tv_epsilon) or tv_epsilon <= 0:
            raise ValueError("tv_epsilon must be finite and positive")
        edge_tensor = torch.as_tensor(te, device=device, dtype=torch.long)
        tv = (edge_tensor[:, 0], edge_tensor[:, 1],
              torch.as_tensor(tw / tw.sum(), device=device, dtype=dtype), float(tv_lambda), float(tv_epsilon))
    exposure = torch.as_tensor(exposure_np, device=device, dtype=dtype)
    total_count = torch.as_tensor(float(exposure_np.sum()), device=device, dtype=dtype)
    positive = composition_np > 0
    entropy_constant = torch.as_tensor(
        float(np.sum(counts[positive] * np.log(composition_np[positive])) / exposure_np.sum()),
        device=device,
        dtype=dtype,
    )

    edges_np = np.empty((0, 2), dtype=int)
    graph_diagnostics: dict[str, object] = {"construction": "not_used_by_rna_only"}
    affinity_np: np.ndarray | None = None
    affinity_info: dict[str, object] | None = None
    samples: np.ndarray | None = None
    if config.method == "poisson_hmrf":
        graph = build_spatial_graph(
            coordinates,
            n_neighbors=config.n_neighbors,
            local_scale_neighbor_rank=config.local_scale_neighbor_rank,
        )
        edges_np = np.asarray(graph.edges, dtype=int)
        graph_diagnostics = dict(graph.diagnostics)
        if config.coordinate_permutation:
            edges_np, null_diagnostics = _coordinate_permuted_edges(
                edges_np, len(coordinates), config.random_state + 100003
            )
            graph_diagnostics.update(null_diagnostics)
        # A common deterministic quadrature rule is used for every biological
        # seed and for the coordinate-null arm.  Seed-specific quadrature
        # noise would otherwise masquerade as affinity instability.
        samples = _quadrature_samples(n_components, config.quadrature_power, 700001)
        affinity_np, affinity_info = fit_affinity_pseudolikelihood(
            p0, edges_np, None, config, samples
        )
    edges = torch.as_tensor(edges_np, device=device, dtype=torch.long)

    rna0, gp0, _ = _raw_poisson_kl_and_gradients(
        counts_tensor, exposure, total_count, entropy_constant, p, b
    )
    spatial_lambda = 0.0
    calibration: dict[str, object] = {"status": "rna_only", "gradient_ratio": 0.0}
    if affinity_np is not None:
        affinity = torch.as_tensor(affinity_np, device=device, dtype=dtype)
        _, gs0 = _pairwise_energy_gradient_torch(p, affinity, edges)
        spatial_lambda, rna_rms, spatial_rms = _gradient_ratio_lambda(
            gp0, gs0, config.gradient_ratio
        )
        calibration = {
            "status": "calibrated_at_warm_start_and_after_each_A_update",
            "gradient_ratio": float(config.gradient_ratio),
            "rna_tangent_gradient_rms": rna_rms,
            "unit_hmrf_tangent_gradient_rms": spatial_rms,
            "spatial_lambda": float(spatial_lambda),
        }
    else:
        affinity = None

    history: list[dict[str, object]] = []
    p_step: float | None = None
    b_step: float | None = None
    stable = 0
    previous_map: float | None = None
    converged = False
    termination = "maximum_outer_iterations_reached"
    for outer in range(1, config.max_outer_iterations + 1):
        p, b, p_step, p_info = _optimize_block(
            "P", counts_tensor, exposure, total_count, entropy_constant, p, b, affinity, edges,
            spatial_lambda, p_step, config, tv
        )
        p, b, b_step, b_info = _optimize_block(
            "B", counts_tensor, exposure, total_count, entropy_constant, p, b, affinity, edges,
            spatial_lambda, b_step, config, tv
        )
        if affinity_np is not None and samples is not None:
            affinity_np, affinity_info = fit_affinity_pseudolikelihood(
                p.detach().cpu().numpy(), edges_np, affinity_np, config, samples
            )
            affinity = torch.as_tensor(affinity_np, device=device, dtype=dtype)

        rna, gp, gb = _raw_poisson_kl_and_gradients(
            counts_tensor, exposure, total_count, entropy_constant, p, b
        )
        spatial = torch.zeros((), device=device, dtype=dtype)
        if affinity is not None:
            spatial, gs = _pairwise_energy_gradient_torch(p, affinity, edges)
            # A pseudo-likelihood updates can change the scale of A sharply,
            # especially on an unstructured coordinate-null graph.  Reapply
            # the declared 0.05 gradient-ratio calibration after every A
            # update so that scale drift cannot turn a weak prior into a
            # self-reinforcing dominating term.
            spatial_lambda, rna_rms, spatial_rms = _gradient_ratio_lambda(
                gp, gs, config.gradient_ratio
            )
            gp = gp + spatial_lambda * gs
        else:
            rna_rms = 0.0
            spatial_rms = 0.0
        ridge_term = (
            float(config.affinity_ridge * np.square(affinity_np).sum())
            if affinity_np is not None
            else 0.0
        )
        map_state_objective = float((rna + spatial_lambda * spatial).item())
        tv_value = 0.0
        if tv is not None:
            tv_tensor, tv_gradient = _tv_energy_gradient(p, tv)
            gp = gp + tv[3] * tv_gradient
            tv_value = float(tv_tensor.item())
            map_state_objective = float((rna + spatial_lambda * spatial + tv[3] * tv_tensor).item())
        # The P/B block and the A pseudo-likelihood block are two alternating
        # approximations, not one globally monotone scalar objective.  Keep
        # the state MAP objective separate from the A pseudo-likelihood/ridge
        # rather than adding incomparable quantities.
        map_objective = map_state_objective
        p_kkt = _mapping_max(p, gp, config.lower_bound)
        b_kkt = _mapping_max(b, gb, config.lower_bound)
        a_kkt = (
            float(affinity_info["packed_gradient_max"])
            if affinity_info is not None
            else 0.0
        )
        relative = (
            math.inf
            if previous_map is None
            else abs(previous_map - map_objective) / max(abs(previous_map), EPS)
        )
        all_kkt = (
            p_kkt <= config.p_kkt_tolerance
            and b_kkt <= config.b_kkt_tolerance
            and a_kkt <= config.affinity_gradient_tolerance
        )
        stable = stable + 1 if all_kkt and relative <= config.relative_tolerance else 0
        history.append(
            {
                "outer_iteration": outer,
                "map_objective": map_objective,
                "map_state_objective": map_state_objective,
                "poisson_kl_per_count": float(rna.item()),
                "TV": tv_value,
                "weighted_TV": float(tv_lambda * tv_value),
                "hmrf_pairwise_energy_mean": float(spatial.item()),
                "spatial_lambda": float(spatial_lambda),
                "rna_tangent_gradient_rms": float(rna_rms),
                "unit_hmrf_tangent_gradient_rms": float(spatial_rms),
                "effective_gradient_ratio": (
                    float(spatial_lambda * spatial_rms / max(rna_rms, EPS))
                    if affinity is not None
                    else 0.0
                ),
                "weighted_hmrf_term": float(spatial_lambda * spatial.item()),
                "affinity_pseudolikelihood": (
                    float(affinity_info["value"]) if affinity_info is not None else 0.0
                ),
                "affinity_ridge_term": (
                    ridge_term
                ),
                "p_kkt_max": p_kkt,
                "b_kkt_max": b_kkt,
                "affinity_gradient_max": a_kkt,
                "relative_map_change": relative,
                "all_kkt_pass": bool(all_kkt),
                "p_steps": int(p_info["accepted_steps"]),
                "b_steps": int(b_info["accepted_steps"]),
                "p_backtracks": int(p_info["backtracks"]),
                "b_backtracks": int(b_info["backtracks"]),
            }
        )
        if config.verbose:
            print(
                f"[{config.method} outer {outer}] obj={map_objective:.9g} "
                f"KL={float(rna.item()):.9g} P-KKT={p_kkt:.3g} "
                f"B-KKT={b_kkt:.3g} A-grad={a_kkt:.3g} "
                f"steps=({p_info['accepted_steps']},{b_info['accepted_steps']})",
                flush=True,
            )
        previous_map = map_objective
        if (
            outer >= config.minimum_outer_iterations
            and stable >= config.convergence_patience
        ):
            converged = True
            termination = "projected_kkt_and_relative_change_met"
            break

    p_np = p.detach().cpu().numpy().astype(float, copy=False)
    b_np = b.detach().cpu().numpy().astype(float, copy=False)
    p_np /= np.maximum(p_np.sum(axis=1, keepdims=True), EPS)
    b_np /= np.maximum(b_np.sum(axis=1, keepdims=True), EPS)
    final_kl_audit = raw_poisson_kl_per_count(counts, p_np, b_np)
    final = history[-1]
    diagnostics = {
        "method": config.method,
        "model": "Y_ig ~ Poisson(library_i * (P B)_ig); observed-edge HMRF MAP energy",
        "warm_start": "caller_supplied; benchmark uses frozen Diffuse v2",
        "gene_side_penalty_active": False,
        "truth_used": False,
        "annotation_used": False,
        "negative_or_non_neighbor_edges_used": False,
        "coordinate_permutation": bool(config.coordinate_permutation),
        "graph": graph_diagnostics,
        "n_edges": int(len(edges_np)),
        "gradient_calibration": calibration,
        "final_spatial_lambda": float(spatial_lambda),
        "final_effective_gradient_ratio": float(final["effective_gradient_ratio"]),
        "affinity_update": affinity_info,
        "converged": bool(converged),
        "termination_reason": termination,
        "outer_iterations": int(len(history)),
        "final_P_KKT_pass": bool(final["p_kkt_max"] <= config.p_kkt_tolerance),
        "final_B_KKT_pass": bool(final["b_kkt_max"] <= config.b_kkt_tolerance),
        "final_A_KKT_pass": bool(
            affinity_np is None
            or final["affinity_gradient_max"] <= config.affinity_gradient_tolerance
        ),
        "final_all_KKT_pass": bool(final["all_kkt_pass"]),
        "initial_poisson_kl_per_count": float(rna0.item()),
        "final_poisson_kl_per_count": float(final["poisson_kl_per_count"]),
        "final_poisson_kl_per_count_float64_audit": float(final_kl_audit),
        "final_map_objective": float(final["map_objective"]),
        "runtime_sec": float(time.perf_counter() - started),
        "device": str(device),
        "dtype": str(dtype),
        "proportion_row_sum_max_error": float(np.max(np.abs(p_np.sum(axis=1) - 1))),
        "profile_row_sum_max_error": float(np.max(np.abs(b_np.sum(axis=1) - 1))),
        "minimum_proportion": float(p_np.min()),
        "minimum_profile": float(b_np.min()),
        "component_mass": p_np.mean(axis=0).tolist(),
        "collapse_detected": bool(np.any(p_np.mean(axis=0) < 1e-4)),
        "tv_lambda": float(tv_lambda),
        "tv_epsilon": float(tv_epsilon),
        "tv_active": tv is not None,
        "tv_n_edges": 0 if tv is None else len(tv[0]),
        "final_TV": float(final["TV"]),
    }
    return PoissonHMRFFit(
        proportions=p_np,
        profiles=b_np,
        affinity=affinity_np,
        edges=edges_np,
        history=history,
        diagnostics=diagnostics,
        config=config.to_dict(),
    )
