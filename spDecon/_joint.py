"""Shared validation, spatial loss and safeguarded MM/SQUAREM optimization."""
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd
import torch

from ._numerics import FLOOR, simplex, em_simplex, quadratic_log_simplex


@dataclass
class JointFit:
    proportions: np.ndarray
    profiles: np.ndarray
    history: list
    diagnostics: dict
    extra: dict = field(default_factory=dict)


def device_name(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select device='cpu' or 'auto'")
    torch.device(device)
    return str(device)


def validate_counts(values):
    if isinstance(values, pd.DataFrame):
        if not values.index.is_unique or not values.columns.is_unique:
            raise ValueError("Spot and feature IDs must be unique")
    y = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or min(y.shape) < 1 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Measurements must be a nonempty, finite, nonnegative matrix")
    if (y.sum(1) <= 0).any():
        raise ValueError("Measurements contain an empty spot")
    return y


def positive_number(value, name, allow_zero=False):
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def initial_factors(y, k, p0=None, b0=None, seed=1, device="auto"):
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not 1 <= k <= min(y.shape):
        raise ValueError("K must be an integer between 1 and min(counts.shape)")
    if (p0 is None) != (b0 is None):
        raise ValueError("Supply both initial_proportions and initial_profiles, or neither")
    if p0 is None:
        from .config import SolverConfig
        from .consensus import build_consensus
        fit = build_consensus(y, k, random_state=seed,
                              solver_config=SolverConfig(device=device_name(device), random_state=seed))
        if not all(d['converged'] for d in fit.diagnostics['branch_diagnostics'].values()):
            raise RuntimeError("An initialization branch did not converge")
        p0, b0 = fit.consensus_proportions, fit.consensus_profiles
    p0, b0 = np.asarray(p0, float), np.asarray(b0, float)
    if p0.shape != (len(y), k) or b0.shape != (k, y.shape[1]):
        raise ValueError("Initial factor dimensions do not match the measurements and K")
    for a in (p0, b0):
        if not np.isfinite(a).all() or (a < 0).any() or (a.sum(1) <= 0).any():
            raise ValueError("Initial factors must be finite, nonnegative and have nonempty rows")
        if a.shape[1] * FLOOR >= 1:
            raise ValueError("Too many features/components for the numerical simplex floor")
    return p0 / p0.sum(1, keepdims=True), b0 / b0.sum(1, keepdims=True)


def validate_edges(edges, n, sections=None):
    if edges is None:
        return np.empty((0, 2), dtype=np.int64)
    e = np.asarray(edges)
    if e.ndim != 2 or e.shape[1] != 2 or not np.issubdtype(e.dtype, np.integer):
        raise ValueError("Edges must be an integer (n_edges, 2) array")
    if (e < 0).any() or (e >= n).any() or (e[:, 0] == e[:, 1]).any():
        raise ValueError("Invalid spatial edge indices")
    if len(np.unique(np.sort(e, axis=1), axis=0)) != len(e):
        raise ValueError("Store each undirected edge once")
    if sections is not None and np.any(np.asarray(sections)[e[:, 0]] != np.asarray(sections)[e[:, 1]]):
        raise ValueError("Spatial edges cannot cross physical sections")
    return e.astype(np.int64, copy=False)


class SpatialLoss:
    def __init__(self, edges, weights, n, device, strength, epsilon=0.02, normalizer=None):
        edges = validate_edges(edges, n)
        self.strength = positive_number(strength, "spatial_weight", True)
        self.epsilon = positive_number(epsilon, "TV epsilon")
        if self.strength and not len(edges):
            raise ValueError("Nonzero spatial_weight requires spatial edges")
        weights = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
        if weights.shape != (len(edges),) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("One finite nonnegative weight is required per edge")
        norm = weights.sum() if normalizer is None else float(normalizer)
        if len(edges) and (not np.isfinite(norm) or norm <= 0):
            raise ValueError("The spatial normalizer must be positive")
        self.i, self.j = torch.as_tensor(edges.T, dtype=torch.long, device=device)
        self.w = torch.as_tensor(weights / norm if len(edges) else weights, dtype=torch.float64, device=device)

    def value(self, p):
        d = p[self.i] - p[self.j]
        return (self.w * (torch.sqrt(d.square().sum(1) + self.epsilon**2) - self.epsilon)).sum()

    def gradient_curvature(self, p):
        d = p[self.i] - p[self.j]
        coefficient = self.strength * self.w / torch.sqrt(d.square().sum(1) + self.epsilon**2)
        grad, degree = torch.zeros_like(p), p.new_zeros(len(p))
        grad.index_add_(0, self.i, coefficient[:, None] * d)
        grad.index_add_(0, self.j, -coefficient[:, None] * d)
        degree.index_add_(0, self.i, coefficient)
        degree.index_add_(0, self.j, coefficient)
        return grad, 2 * degree

    def step(self, p, counts):
        grad, curvature = self.gradient_curvature(p)
        curvature = len(p) * curvature
        linear = len(p) * grad - curvature[:, None] * p
        return quadratic_log_simplex(counts, curvature, linear) if self.strength else em_simplex(counts)


class MolecularLoss:
    def __init__(self, y, device):
        self.y = torch.as_tensor(y, dtype=torch.float64, device=device)
        self.s = self.y.sum(1, keepdim=True)
        self.mass = self.y.sum()
        self.const = (self.y * (self.y / self.s).clamp_min(1e-300).log()).sum() / self.mass

    def value(self, p, b):
        rate = (p @ b).clamp_min(1e-300)
        return self.const - (self.y * rate.log()).sum() / self.mass + (self.s * rate).sum() / self.mass - 1

    def gradients(self, p, b):
        residual = (self.s - self.y / (p @ b).clamp_min(1e-300)) / self.mass
        return residual @ b.T, p.T @ residual


def clean_floor(p):
    removed = torch.where(p <= FLOOR + 1e-12, p - FLOOR, 0.)
    result = p - removed
    result.scatter_add_(1, p.argmax(1, keepdim=True), removed.sum(1, keepdim=True))
    return result


def diagnostics(model, state):
    value, parts = model.value(state)
    gradients = model.gradients(state)
    proposal = model.project([v - scale * g for v, g, scale in zip(state, gradients, model.scales)])
    residuals = [float((a - b).abs().max()) for a, b in zip(state, proposal)]
    p, g = state[0], len(state[0]) * gradients[0]
    gap = float((g.masked_fill(p <= FLOOR + 1e-14, -torch.inf).max(1).values - g.min(1).values).clamp_min(0).max())
    return dict(objective=float(value), **{k: float(v) for k, v in parts.items()},
                P_KKT=residuals[0], profile_KKT=max(residuals[1:], default=0.),
                P_pairwise_KKT=gap, stopping_KKT=max(residuals + [gap]))


def polish_p(model, state, steps=5):
    """Feasible pair transfers with full-objective Armijo safeguards."""
    state = list(state)
    for _ in range(steps):
        p = state[0]
        g = len(p) * model.gradients(state)[0]
        row = torch.arange(len(p), device=p.device)
        enter = g.argmin(1)
        leave = g.masked_fill(p <= FLOOR + 1e-14, -torch.inf).argmax(1)
        gap = (g[row, leave] - g[row, enter]).clamp_min(0)
        curvature = model.p_curvature(state, enter, leave).clamp_min(1e-18)
        amount = torch.minimum(gap / curvature, (p[row, leave] - FLOOR).clamp_min(0))
        amount = torch.where(enter == leave, 0., amount)
        direction = torch.zeros_like(p)
        direction[row, enter] += amount
        direction[row, leave] -= amount
        slope = (g * direction).sum() / len(p)
        if float(slope) >= 0:
            break
        before = model.value(state)[0]
        alpha = 1.
        for _ in range(50):
            candidate = [clean_floor(p + alpha * direction)] + state[1:]
            value = model.value(candidate)[0]
            if bool(torch.isfinite(value)) and value <= before + 1e-4 * alpha * slope + 1e-14:
                state = candidate
                break
            alpha *= .5
        else:
            break
    return state


def spatial_pair_curvature(graph, p):
    return 2 * len(p) * graph.gradient_curvature(p)[1]


@torch.no_grad()
def optimize(model, initial, *, max_iter=20000, tolerance=1e-4,
             relative_tolerance=1e-7, check_every=25, minimum_iterations=25, progress=None):
    for value, name in ((max_iter, "max_iter"), (check_every, "check_every"), (minimum_iterations, "minimum_iterations")):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    positive_number(tolerance, "tolerance")
    positive_number(relative_tolerance, "relative_tolerance")
    started = time.perf_counter()
    state = model.project(initial)
    before = previous = float(model.value(state)[0])
    history, converged, maximum_increase = [], False, 0.
    for iteration in range(max_iter + 1):
        if iteration % check_every == 0 or iteration == max_iter:
            row = diagnostics(model, state)
            change = abs(row['objective'] - previous) / max(1., abs(previous))
            converged = iteration >= minimum_iterations and row['stopping_KKT'] <= tolerance and change <= relative_tolerance
            row.update(iteration=iteration, relative_change=change)
            history.append(row)
            if progress is not None:
                progress(dict(row))
            previous = row['objective']
            if converged:
                break
        if iteration == max_iter:
            break
        first = model.step(state)
        second = model.step(first)
        chosen, value = second, float(model.value(second)[0])
        if not np.isfinite(value) or value > before + 1e-9:
            raise RuntimeError(f"MM objective increased: {before} -> {value}")
        displacement = [a - b for a, b in zip(first, state)]
        change = [a - 2*b + c for a, b, c in zip(second, first, state)]
        distance = lambda xs: sum(float(x.square().sum() / np.prod(x.shape[:-1])) for x in xs)
        length = float(np.clip(np.sqrt(distance(displacement) / max(distance(change), 1e-30)), 1, 64))
        if length > 1.05:
            for _ in range(6):
                trial = model.project([a + 2*length*r + length**2*v for a, r, v in zip(state, displacement, change)])
                candidate = float(model.value(trial)[0])
                if np.isfinite(candidate) and candidate <= value + 1e-13:
                    chosen, value = trial, candidate
                    break
                length = 1 + (length - 1) / 2
        if (iteration + 1) % 50 == 0:
            chosen = polish_p(model, chosen)
        chosen[0] = clean_floor(chosen[0])
        value = float(model.value(chosen)[0])
        if not np.isfinite(value) or value > before + 1e-9:
            raise RuntimeError("Accelerated/polished update increased the complete objective")
        maximum_increase = max(maximum_increase, value - before)
        state, before = chosen, value
    diag = dict(history[-1], converged=converged,
                termination_reason='joint_KKT_and_relative_change' if converged else 'maximum_iterations',
                runtime_sec=time.perf_counter() - started, max_objective_increase=maximum_increase,
                optimizer='MM_SQUAREM', device=str(state[0].device), dtype='float64',
                tolerance=tolerance, relative_tolerance=relative_tolerance, max_iter=max_iter)
    return state, history, diag
