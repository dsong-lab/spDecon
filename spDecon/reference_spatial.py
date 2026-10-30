"""Extend the fixed-reference projected-gradient/BB solver with graph TV."""
import time
import numpy as np
import torch
from . import reference as native


def tv_value_gradient(p, edges, weights, epsilon=.02):
    i, j = edges
    difference = p[i] - p[j]
    lengths = (difference.square().sum(1) + epsilon**2).sqrt()
    value = (weights * (lengths - epsilon)).sum()
    contribution = (weights / lengths)[:, None] * difference
    gradient = torch.zeros_like(p)
    gradient.index_add_(0, i, contribution)
    gradient.index_add_(0, j, -contribution)
    return value, gradient


class Objective:
    def __init__(self, counts, reference, edge_i, edge_j, weights, strength):
        y, r, _, _ = native._validate(counts, len(reference), reference, None, None, 1e-12)
        self.y, self.b = [torch.as_tensor(a, dtype=torch.float64) for a in (y, r)]
        self.s = self.y.sum(1)
        self.total = self.s.sum()
        self.mass = self.s / self.total
        self.q = self.y / self.s[:, None]
        self.entropy = torch.where(self.y > 0, self.y * self.q.clamp(min=1e-300).log(), 0.).sum() / self.total
        self.edges = tuple(torch.as_tensor(a, dtype=torch.long) for a in (edge_i, edge_j))
        self.weights = torch.as_tensor(weights / np.sum(weights), dtype=torch.float64)
        self.strength = float(strength)

    def __call__(self, p):
        mixture = p @ self.b
        rna = self.entropy - (self.y * mixture.log()).sum() / self.total
        rna = rna + (self.s[:, None] * mixture).sum() / self.total - 1.
        gradient = ((self.s[:, None] - self.y / mixture) / self.total) @ self.b.T
        tv, tv_gradient = tv_value_gradient(p, self.edges, self.weights)
        return rna + self.strength * tv, gradient + self.strength * tv_gradient, rna, tv


def fit(counts, reference, edge_i, edge_j, weights, strength, max_steps=3000, tolerance=1e-6,
        *, verbose=True, log_every=1000):
    """Same uniform start, simplex, row BB scaling and KKT; coupled Armijo.

    TV links rows, so line search must accept the complete update against the
    joint objective. Individual row acceptance is no longer a descent guarantee.
    The lambda=0 run through this same code is retained as a solver-only control.
    """
    started = time.perf_counter()
    if not isinstance(log_every, int) or log_every < 1:
        raise ValueError('log_every must be a positive integer')
    if verbose:
        print(f"[spDecon:reference] optimization started | spots={len(counts)} "
              f"K={len(reference)} max_steps={max_steps} KKT_tolerance={tolerance:g}", flush=True)
    obj = Objective(counts, reference, edge_i, edge_j, weights, strength)
    p = torch.full((len(counts), len(reference)), 1. / len(reference), dtype=torch.float64)
    value, gradient, rna, tv = obj(p)
    conditional = gradient / obj.mass[:, None]
    scale = (conditional - conditional.mean(1, keepdim=True)).abs().amax(1)
    steps = (.1 / scale.clamp(min=1e-8)).clamp(min=1e-10, max=1e6)
    history, backtracks, increases = [], 0, 0.
    for iteration in range(max_steps + 1):
        mappings = (p - native._project_simplex(p - conditional)).abs().amax(1)
        kkt = float(mappings.max())
        if verbose and iteration % log_every == 0:
            print(f"[spDecon:reference] iteration={iteration} objective={float(value):.9g} "
                  f"KKT={kkt:.3g} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if iteration % 20 == 0 or kkt <= tolerance or iteration == max_steps:
            history.append(dict(iteration=iteration, objective=float(value), RNA_KL=float(rna),
                                TV=float(tv), conditional_KKT=kkt, global_KKT=native._mapping(p, gradient)))
        if kkt <= tolerance or iteration == max_steps:
            break
        active = mappings > tolerance
        trial = steps.clone()
        for backtrack in range(45):
            candidate = native._project_simplex(p - trial[:, None] * conditional)
            candidate[~active] = p[~active]
            candidate_value, candidate_gradient, candidate_rna, candidate_tv = obj(candidate)
            slope = (gradient * (candidate - p)).sum()
            if torch.isfinite(candidate_value) and candidate_value <= value + 1e-4 * slope + 2e-14:
                break
            trial *= .5
            backtracks += 1
        else:
            raise RuntimeError('Coupled Armijo line search failed')
        new_conditional = candidate_gradient / obj.mass[:, None]
        displacement = candidate - p
        denominator = (displacement * (new_conditional - conditional)).sum(1)
        bb = displacement.square().sum(1) / denominator.clamp(min=1e-300)
        steps = torch.where(denominator > 1e-24, bb, trial * 1.5).clamp(min=1e-10, max=1e6)
        increases = max(increases, float(candidate_value - value))
        p, value, gradient, conditional = candidate, candidate_value, candidate_gradient, new_conditional
        rna, tv = candidate_rna, candidate_tv
    result = p.numpy()
    assert np.isfinite(result).all() and result.min() >= 0
    np.testing.assert_allclose(result.sum(1), 1, atol=1e-12, rtol=0)
    diag = dict(converged=kkt <= tolerance, iterations=iteration,
        conditional_KKT=kkt, global_KKT=native._mapping(p, gradient), objective=float(value),
        RNA_KL=float(rna), TV=float(tv), spatial_weight=float(strength), max_steps=max_steps,
        KKT_tolerance=tolerance, backtracks=backtracks, max_objective_increase=increases,
        runtime_sec=time.perf_counter()-started, reference_fixed=True,
        initialization='uniform', solver='native projected-gradient/row BB extended with joint Armijo',
        dtype='float64', device='cpu', truth_used=False, histology_used=False)
    if verbose:
        status = 'converged' if diag['converged'] else 'iteration limit reached; not converged'
        print(f"[spDecon:reference] finished | {status} | iterations={iteration} "
              f"objective={diag['objective']:.9g} KKT={kkt:.3g} "
              f"elapsed={diag['runtime_sec']:.1f}s", flush=True)
    return result, obj.b.numpy(), history, diag
