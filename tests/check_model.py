"""Check KL/TV derivatives against autograd and fit the constrained model on CPU."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spDecon.config import SolverConfig
from spDecon.model import _kl_and_gradients, _tv_energy_gradient, fit_spdecon


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    counts = rng.poisson(3, size=(12, 24)).astype(float)
    p0 = rng.dirichlet(np.ones(3), size=12)
    b0 = rng.dirichlet(np.ones(24), size=3)
    y = torch.tensor(counts, dtype=torch.float64)
    p = torch.tensor(p0, requires_grad=True)
    b = torch.tensor(b0, requires_grad=True)
    exposure = y.sum(1)
    positive = y > 0
    entropy = (y[positive] * torch.log((y / exposure[:, None])[positive])).sum() / y.sum()
    value, gp, gb = _kl_and_gradients(y, exposure, y.sum(), entropy, p, b)
    mean = exposure[:, None] * (p @ b)
    direct_kl = ((y[positive] * torch.log(y[positive] / mean[positive])).sum()
                 + (mean - y).sum()) / y.sum()
    expected_gp, expected_gb = torch.autograd.grad(direct_kl, (p, b))
    for actual, expected in ((value, direct_kl), (gp, expected_gp), (gb, expected_gb)):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    edges = np.column_stack((np.arange(11), np.arange(1, 12)))
    weights = np.linspace(0.5, 1, len(edges))
    normalized = torch.tensor(weights / weights.sum())
    left, right = torch.tensor(edges.T)
    epsilon = 0.02
    tv, gradient = _tv_energy_gradient(p, (left, right, normalized, 1, epsilon))
    difference = p[left] - p[right]
    direct_tv = (normalized * (torch.sqrt(difference.square().sum(1) + epsilon**2) - epsilon)).sum()
    expected_gradient, = torch.autograd.grad(direct_tv, p)
    torch.testing.assert_close(tv, direct_tv, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)

    for strength in (0.0, 0.07):
        fit = fit_spdecon(counts, 3, p0, b0, SolverConfig(device="cpu", dtype="float64"),
                         tv_edges=edges, tv_weights=weights, tv_lambda=strength)
        assert fit.diagnostics["converged"], fit.diagnostics
        for matrix in (fit.proportions, fit.profiles):
            assert np.isfinite(matrix).all() and (matrix > 0).all()
            np.testing.assert_allclose(matrix.sum(1), 1, atol=1e-12, rtol=0)
        fitted_mean = counts.sum(1)[:, None] * (fit.proportions @ fit.profiles)
        mask = counts > 0
        terms = fitted_mean - counts
        terms[mask] += counts[mask] * np.log(counts[mask] / fitted_mean[mask])
        delta = fit.proportions[edges[:, 0]] - fit.proportions[edges[:, 1]]
        fitted_tv = np.average(np.sqrt((delta**2).sum(1) + epsilon**2) - epsilon, weights=weights)
        objective = terms.sum() / counts.sum() + strength * fitted_tv
        np.testing.assert_allclose(fit.diagnostics["final_objective"], objective, atol=1e-12, rtol=1e-12)
        assert objective <= float(direct_kl.detach() + strength * direct_tv.detach()) + 1e-10
    print("PASS: KL/TV values and gradients, converged simplex fits, and final objective audit.")


if __name__ == "__main__":
    main()
