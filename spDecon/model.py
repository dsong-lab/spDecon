"""Section 1 simplex-constrained generalized KL + spatial TV optimizer."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from .config import SolverConfig

EPS = 1e-12


@dataclass
class SpDeconFit:
    proportions: np.ndarray
    profiles: np.ndarray
    history: list[dict[str, object]]
    diagnostics: dict[str, object]
    config: dict[str, object]


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


def _validate(
    counts: np.ndarray,
    n_components: int,
    initial_proportions: np.ndarray,
    initial_profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    p = np.asarray(initial_proportions, dtype=float)
    b = np.asarray(initial_profiles, dtype=float)
    if counts.ndim != 2 or min(counts.shape) < 1 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite nonnegative matrix")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("counts contain an empty spot")
    if p.shape != (counts.shape[0], n_components):
        raise ValueError("initial proportions have an incompatible shape")
    if b.shape != (n_components, counts.shape[1]):
        raise ValueError("initial profiles have an incompatible shape")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(~np.isfinite(b)) or np.any(b < 0):
        raise ValueError("initial factors must be finite and nonnegative")
    p = p / np.maximum(p.sum(axis=1, keepdims=True), EPS)
    b = b / np.maximum(b.sum(axis=1, keepdims=True), EPS)
    return counts, p, b


def _torch_context(config: SolverConfig) -> tuple[torch.device, torch.dtype]:
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


def _kl_and_gradients(
    counts: torch.Tensor,
    exposure: torch.Tensor,
    total_count: torch.Tensor,
    entropy_constant: torch.Tensor,
    p: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw-count generalized KL and gradients, normalized per observed count.

    The count-only ``sum Y log(Y / s)`` contribution is cached by the caller.
    This avoids recomputing a masked logarithm over the full count matrix
    at every line-search/EM step.
    """

    mixture = torch.clamp(p @ b, min=EPS)
    # Keep the linear generalized-KL term explicit.  It cancels analytically
    # for exact row-simplex factors, but retaining it prevents float32
    # accumulation error across many genes from biasing the convergence log.
    linear = torch.sum(exposure[:, None] * mixture) / total_count - 1.0
    loss = linear + entropy_constant - torch.sum(counts * torch.log(mixture)) / total_count
    residual = (exposure[:, None] - counts / mixture) / total_count
    return loss, residual @ b.T, p.T @ residual


def kl_per_count(
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


def _mapping_max(values: torch.Tensor, gradient: torch.Tensor, lower: float) -> float:
    mapping = values - project_rows_lower_simplex(values - gradient, lower)
    return float(torch.max(torch.abs(mapping)).item())


def _initial_step(gradient: torch.Tensor, target: float, minimum: float, maximum: float) -> float:
    tangent = gradient - gradient.mean(dim=1, keepdim=True)
    scale = float(torch.max(torch.abs(tangent)).item())
    return float(np.clip(target / max(scale, EPS), minimum, maximum))


def _optimize_block(
    block: str,
    counts: torch.Tensor,
    exposure: torch.Tensor,
    total_count: torch.Tensor,
    entropy_constant: torch.Tensor,
    p: torch.Tensor,
    b: torch.Tensor,
    step: float | None,
    config: SolverConfig,
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
        rna, gp, gb = _kl_and_gradients(
            counts, exposure, total_count, entropy_constant, candidate_p, candidate_b
        )
        objective = rna
        if tv is not None:
            tv_value, tv_gradient = _tv_energy_gradient(candidate_p, tv)
            objective = objective + tv[3] * tv_value
            gp = gp + tv[3] * tv_gradient
        return objective, rna, gp, gb

    if block == "B":
        # With P fixed, the KL profile block admits a responsibility/EM
        # update. This avoids large Euclidean gradients near the profile floor.
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
            # Accumulation across many genes in float32 can otherwise leave
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
            "solver": "KL_mixture_EM",
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
            candidate = project_rows_lower_simplex(
                x - trial_step * gradient, config.lower_bound
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
    n_components: int,
    initial_proportions: np.ndarray,
    initial_profiles: np.ndarray,
    config: SolverConfig | dict[str, object] | None = None,
    *,
    tv_edges: np.ndarray | None = None,
    tv_weights: np.ndarray | None = None,
    tv_lambda: float = 0.0,
    tv_epsilon: float = 0.02,
) -> SpDeconFit:
    """Jointly fit P and B under generalized KL, optionally with spatial TV.

    Branch fitting uses tv_lambda=0; final fitting starts from the aligned
    consensus with the calibrated spatial coefficient.
    """
    started = time.perf_counter()
    if config is None:
        config = SolverConfig()
    elif isinstance(config, dict):
        config = SolverConfig(**config)
    config.validate()
    counts, p0, b0 = _validate(
        counts, n_components, initial_proportions, initial_profiles
    )
    device, dtype = _torch_context(config)
    if config.verbose:
        print(f"[spDecon] optimization started | spots={len(counts)} features={counts.shape[1]} "
              f"K={n_components} seed={config.random_state} device={device} "
              f"max_outer={config.max_outer_iterations}", flush=True)
    exposure_np = counts.sum(axis=1)
    composition_np = counts / exposure_np[:, None]
    p = project_rows_lower_simplex(
        torch.as_tensor(p0, device=device, dtype=dtype), config.lower_bound
    )
    b = project_rows_lower_simplex(
        torch.as_tensor(b0, device=device, dtype=dtype), config.lower_bound
    )
    counts_tensor = torch.as_tensor(counts, device=device, dtype=dtype)
    if not math.isfinite(tv_lambda) or tv_lambda < 0:
        raise ValueError("tv_lambda must be finite and nonnegative")
    tv = None
    if tv_lambda > 0:
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
              torch.as_tensor(tw / tw.sum(), device=device, dtype=dtype),
              float(tv_lambda), float(tv_epsilon))
    exposure = torch.as_tensor(exposure_np, device=device, dtype=dtype)
    total_count = torch.as_tensor(float(exposure_np.sum()), device=device, dtype=dtype)
    positive = composition_np > 0
    entropy_constant = torch.as_tensor(
        float(np.sum(counts[positive] * np.log(composition_np[positive])) / exposure_np.sum()),
        device=device, dtype=dtype,
    )
    initial_kl, _, _ = _kl_and_gradients(
        counts_tensor, exposure, total_count, entropy_constant, p, b
    )
    history = []
    p_step = b_step = None
    stable = 0
    previous_objective = None
    converged = False
    for outer in range(1, config.max_outer_iterations + 1):
        p, b, p_step, p_info = _optimize_block(
            "P", counts_tensor, exposure, total_count, entropy_constant,
            p, b, p_step, config, tv,
        )
        p, b, b_step, b_info = _optimize_block(
            "B", counts_tensor, exposure, total_count, entropy_constant,
            p, b, b_step, config, tv,
        )
        kl, gp, gb = _kl_and_gradients(
            counts_tensor, exposure, total_count, entropy_constant, p, b
        )
        objective = float(kl.item())
        tv_value = 0.0
        if tv is not None:
            tv_tensor, tv_gradient = _tv_energy_gradient(p, tv)
            gp = gp + tv[3] * tv_gradient
            tv_value = float(tv_tensor.item())
            objective = float((kl + tv[3] * tv_tensor).item())
        p_kkt = _mapping_max(p, gp, config.lower_bound)
        b_kkt = _mapping_max(b, gb, config.lower_bound)
        relative = (
            math.inf if previous_objective is None else
            abs(previous_objective - objective) / max(abs(previous_objective), EPS)
        )
        all_kkt = p_kkt <= config.p_kkt_tolerance and b_kkt <= config.b_kkt_tolerance
        stable = stable + 1 if all_kkt and relative <= config.relative_tolerance else 0
        history.append({
            "outer_iteration": outer, "objective": objective,
            "kl_per_count": float(kl.item()), "TV": tv_value,
            "weighted_TV": float(tv_lambda * tv_value),
            "p_kkt_max": p_kkt, "b_kkt_max": b_kkt,
            "relative_objective_change": relative, "all_kkt_pass": bool(all_kkt),
            "p_steps": int(p_info["accepted_steps"]),
            "b_steps": int(b_info["accepted_steps"]),
            "p_backtracks": int(p_info["backtracks"]),
            "b_backtracks": int(b_info["backtracks"]),
        })
        if config.verbose and (outer == 1 or outer % config.log_every == 0):
            print(f"[spDecon] iteration={outer} objective={objective:.9g} "
                  f"P-KKT={p_kkt:.3g} B-KKT={b_kkt:.3g} relative_change={relative:.3g} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        previous_objective = objective
        if outer >= config.minimum_outer_iterations and stable >= config.convergence_patience:
            converged = True
            break

    p_np = p.detach().cpu().numpy().astype(float, copy=False)
    b_np = b.detach().cpu().numpy().astype(float, copy=False)
    p_np /= np.maximum(p_np.sum(axis=1, keepdims=True), EPS)
    b_np /= np.maximum(b_np.sum(axis=1, keepdims=True), EPS)
    final = history[-1]
    diagnostics = {
        "method": "spDecon_KL_TV" if tv is not None else "spDecon_KL",
        "converged": bool(converged),
        "termination_reason": "projected_kkt_and_relative_change_met" if converged else "maximum_outer_iterations_reached",
        "outer_iterations": len(history),
        "final_P_KKT_pass": bool(final["p_kkt_max"] <= config.p_kkt_tolerance),
        "final_B_KKT_pass": bool(final["b_kkt_max"] <= config.b_kkt_tolerance),
        "final_all_KKT_pass": bool(final["all_kkt_pass"]),
        "initial_kl_per_count": float(initial_kl.item()),
        "final_kl_per_count": final["kl_per_count"],
        "final_kl_per_count_float64_audit": kl_per_count(counts, p_np, b_np),
        "final_objective": final["objective"],
        "runtime_sec": time.perf_counter() - started,
        "device": str(device), "dtype": str(dtype),
        "proportion_row_sum_max_error": float(np.max(np.abs(p_np.sum(axis=1) - 1))),
        "profile_row_sum_max_error": float(np.max(np.abs(b_np.sum(axis=1) - 1))),
        "minimum_proportion": float(p_np.min()), "minimum_profile": float(b_np.min()),
        "component_mass": p_np.mean(axis=0).tolist(),
        "collapse_detected": bool(np.any(p_np.mean(axis=0) < 1e-4)),
        "tv_lambda": float(tv_lambda), "tv_epsilon": float(tv_epsilon),
        "tv_active": tv is not None, "tv_n_edges": 0 if tv is None else len(tv[0]),
        "final_TV": final["TV"],
    }
    if config.verbose:
        status = 'converged' if converged else 'iteration limit reached; not converged'
        print(f"[spDecon] finished | {status} | iterations={len(history)} "
              f"objective={final['objective']:.9g} P-KKT={final['p_kkt_max']:.3g} "
              f"B-KKT={final['b_kkt_max']:.3g} elapsed={diagnostics['runtime_sec']:.1f}s", flush=True)
    return SpDeconFit(p_np, b_np, history, diagnostics, config.to_dict())
