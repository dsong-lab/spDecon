"""The three Section 1 initializations and their full-feature KL fits."""
import time

import numpy as np
from sklearn.decomposition import NMF, PCA

from .clustering import louvain_labels
from .config import SolverConfig
from .model import fit_spdecon
from .preprocessing import exposure_balanced_input, exposure_null_residuals, row_normalize

EPS = 1e-10


def _full_profiles(balanced, weights, iterations=75):
    gram = weights.T @ weights
    ridge = 1e-8 * max(float(np.trace(gram)) / max(len(gram), 1), 1.0)
    profiles = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]), weights.T @ balanced
    )
    profiles = np.maximum(profiles, EPS)
    denominator = np.maximum(weights.sum(axis=0)[:, None], EPS)
    for _ in range(iterations):
        fitted = np.maximum(weights @ profiles, EPS)
        profiles *= (weights.T @ (balanced / fitted)) / denominator
        profiles = np.maximum(profiles, EPS)
    return profiles


def _kl_nmf(values, k, seed):
    estimator = NMF(
        n_components=k, init="nndsvda", solver="mu",
        beta_loss="kullback-leibler", max_iter=200, tol=1e-4,
        random_state=seed,
    )
    return estimator.fit_transform(values), estimator


def _refine(counts, weights, profiles, seed, scale_floor, started, details, solver_config=None):
    component_scale = np.maximum(profiles.sum(axis=1), scale_floor)
    initial_p = row_normalize(weights * component_scale[None, :])
    initial_b = row_normalize(profiles)
    fit = fit_spdecon(
        counts, weights.shape[1], initial_p, initial_b,
        solver_config or SolverConfig(random_state=seed),
    )
    fit.diagnostics.update(details)
    fit.diagnostics["standalone_runtime_sec"] = time.perf_counter() - started
    return fit


def fit_initializations(counts, n_components, random_state=0, *, solver_config=None):
    """Fit all-feature, residual-feature, and expression-neighborhood branches.

    Initialization uses balanced/transformed measurements. Each branch's
    final KL fit uses all features of the original input counts.
    """
    counts = np.asarray(counts, dtype=float)
    balanced, exposure = exposure_balanced_input(counts)
    k, seed = int(n_components), int(random_state)
    if not 1 <= k <= min(counts.shape):
        raise ValueError("n_components must be between 1 and min(counts.shape)")
    results = {}

    started = time.perf_counter()
    weights, nmf = _kl_nmf(balanced, k, seed)
    profiles = np.maximum(nmf.components_, 0.0)
    results["all_features"] = _refine(
        counts, weights, profiles, seed, 1e-12, started,
        {"initialization": "all-feature NNDSVDa + KL-NMF",
         "initialization_nmf_iterations": int(nmf.n_iter_)}, solver_config,
    )

    started = time.perf_counter()
    feature_count = min(5000, max(k, int(np.ceil(0.25 * counts.shape[1]))))
    residuals = exposure_null_residuals(counts, exposure)
    selected = np.argsort(-residuals.var(axis=0), kind="mergesort")[:feature_count]
    weights, nmf = _kl_nmf(balanced[:, selected], k, seed)
    profiles = _full_profiles(balanced, weights)
    results["variable_features"] = _refine(
        counts, weights, profiles, seed, EPS, started,
        {"initialization": "exposure-residual features + KL-NMF",
         "n_initialization_features": feature_count,
         "initialization_nmf_iterations": int(nmf.n_iter_)}, solver_config,
    )

    started = time.perf_counter()
    selected = np.flatnonzero((counts > 0).mean(axis=0) > 0.05)
    if len(selected) < k:
        raise ValueError("Prevalence filter retained fewer than K features")
    retained = counts[:, selected]
    log_selected = np.log1p(
        1e4 * retained / np.maximum(retained.sum(axis=1, keepdims=True), EPS)
    )
    n_pcs = min(50, len(counts) - 1, len(selected))
    if n_pcs < 1:
        raise ValueError("Not enough observations/features for PCA")
    representation = PCA(
        n_components=n_pcs, svd_solver="randomized", random_state=seed
    ).fit_transform(log_selected)
    labels, neighbors = louvain_labels(representation, k, initial_neighbors=20)
    hard_weights = np.full((len(labels), k), EPS)
    hard_weights[np.arange(len(labels)), labels] = 1.0
    cluster_profiles = np.maximum(
        np.stack([log_selected[labels == j].mean(axis=0) for j in range(k)]), EPS
    )
    smoother = NMF(
        n_components=k, init="custom", solver="mu", beta_loss="frobenius",
        max_iter=10, tol=0.0, random_state=seed,
    )
    weights = smoother.fit_transform(
        log_selected, W=hard_weights.copy(), H=cluster_profiles.copy()
    )
    profiles = _full_profiles(balanced, weights)
    results["expression_clusters"] = _refine(
        counts, weights, profiles, seed, EPS, started,
        {"initialization": "log-CPM + PCA + fixed-K Louvain + Frobenius-NMF",
         "selected_feature_count": len(selected), "pca_components": n_pcs,
         "louvain_neighbors": neighbors,
         "cluster_sizes": np.bincount(labels, minlength=k).tolist()}, solver_config,
    )
    return results
