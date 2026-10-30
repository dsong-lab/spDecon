"""Standalone spGSNMF v2: exposure-balanced all-gene KL factorization.

The estimator starts from the count matrix on every call.  It does not fit,
read, or blend a result from another spGSNMF version.  Generalized-KL NMF on
exposure-normalized counts gives an equal-spot conditional expression
factorization; component scales are then fixed so that spot rows are
proportions and profile rows are gene distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


EPS = 1e-10


@dataclass
class V2Fit:
    proportions: np.ndarray
    profiles: np.ndarray
    marker_indices: dict[int, list[int]]
    gene_scores: pd.DataFrame
    diagnostics: dict[str, object]


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize nonnegative rows; an empty row becomes uniform."""

    matrix = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    totals = matrix.sum(axis=1, keepdims=True)
    result = matrix / np.maximum(totals, EPS)
    empty = totals[:, 0] <= EPS
    if np.any(empty):
        result[empty] = 1.0 / matrix.shape[1]
    return result


def compute_exposure(counts: np.ndarray) -> np.ndarray:
    """Library size relative to the median positive library size."""

    libraries = np.maximum(np.asarray(counts, dtype=float), 0.0).sum(axis=1)
    positive = libraries[libraries > 0]
    reference = float(np.median(positive)) if positive.size else 1.0
    return np.maximum(libraries / max(reference, EPS), EPS)


def _column_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    numerator = left.T @ right
    denominator = np.sqrt(
        np.maximum(np.square(left).sum(axis=0)[:, None], EPS)
        * np.maximum(np.square(right).sum(axis=0)[None, :], EPS)
    )
    return np.nan_to_num(numerator / denominator)


def exposure_null_residuals(
    counts: np.ndarray, exposure: np.ndarray
) -> np.ndarray:
    """Clipped Pearson residuals from an exposure-only gene model."""

    counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
    exposure = np.maximum(np.asarray(exposure, dtype=float), EPS)
    abundance = counts.sum(axis=0) / max(float(exposure.sum()), EPS)
    expected = np.maximum(exposure[:, None] * abundance[None, :], EPS)
    residuals = (counts - expected) / np.sqrt(expected)
    bound = np.sqrt(max(counts.shape[0], 1))
    return np.clip(residuals, -bound, bound)


def interleaved_folds(
    n_spots: int,
    coords: np.ndarray | None,
    n_folds: int,
    random_state: int,
) -> np.ndarray:
    """Make folds spread across the tissue, or balanced folds without coords."""

    n_folds = min(max(int(n_folds), 2), int(n_spots))
    if coords is None:
        labels = np.arange(n_spots, dtype=int) % n_folds
        np.random.default_rng(int(random_state) + 1901).shuffle(labels)
        return labels
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] != n_spots or coords.shape[1] < 2:
        raise ValueError("coords must have one row and at least two columns per spot")
    ranks = [
        np.argsort(np.argsort(coords[:, axis], kind="mergesort"), kind="mergesort")
        for axis in range(2)
    ]
    return ((ranks[0] + 2 * ranks[1]) % n_folds).astype(int)


def select_v2_markers(
    counts: np.ndarray,
    exposure: np.ndarray,
    proportions: np.ndarray,
    profiles: np.ndarray,
    coords: np.ndarray | None = None,
    n_folds: int = 4,
    max_markers: int = 30,
    random_state: int = 0,
) -> tuple[dict[int, list[int]], pd.DataFrame]:
    """Report strong component markers learned by v2 itself.

    Markers are an interpretation output in v2; they do not reset or constrain
    its fitted proportions.
    """

    counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
    P = row_normalize(proportions)
    C = row_normalize(profiles)
    residuals = exposure_null_residuals(counts, exposure)
    folds = interleaved_folds(
        counts.shape[0], coords, n_folds=n_folds, random_state=random_state
    )
    fold_association = np.stack(
        [
            np.maximum(_column_corr(residuals[folds == fold], P[folds == fold]), 0.0)
            for fold in sorted(np.unique(folds))
        ]
    )
    robust = np.quantile(fold_association, 0.25, axis=0)
    median = np.median(fold_association, axis=0)
    fold_winner = np.argmax(fold_association, axis=2)
    owner = np.argmax(median, axis=1)
    profile_share = C / np.maximum(C.sum(axis=0, keepdims=True), EPS)
    detected = (counts > 0).sum(axis=0) >= max(
        5, int(np.ceil(0.01 * counts.shape[0]))
    )

    marker_indices: dict[int, list[int]] = {}
    rows: list[dict[str, float | int | bool]] = []
    for component in range(P.shape[1]):
        if P.shape[1] > 1:
            other_association = np.max(
                np.delete(median, component, axis=1), axis=1
            )
            other_share = np.max(
                np.delete(profile_share, component, axis=0), axis=0
            )
        else:
            other_association = np.zeros(counts.shape[1])
            other_share = np.zeros(counts.shape[1])
        association_margin = median[:, component] - other_association
        profile_gap = profile_share[component] - other_share
        consistency = np.mean(fold_winner == component, axis=0)
        eligible = (
            detected
            & (owner == component)
            & (robust[:, component] >= 0.05)
            & (association_margin >= 0.02)
            & (profile_gap > 0.0)
            & (consistency >= 0.75)
        )
        score = (
            np.maximum(robust[:, component], 0.0)
            * np.maximum(profile_gap, 0.0)
            * consistency
        )
        genes = np.flatnonzero(eligible)
        ranked = genes[np.lexsort((genes, -score[genes]))]
        selected = ranked[: int(max_markers)].astype(int).tolist()
        marker_indices[component] = selected
        for gene in ranked.tolist():
            rows.append(
                {
                    "component": int(component),
                    "gene_index": int(gene),
                    "score": float(score[gene]),
                    "robust_association": float(robust[gene, component]),
                    "association_margin": float(association_margin[gene]),
                    "profile_share": float(profile_share[component, gene]),
                    "profile_gap": float(profile_gap[gene]),
                    "fold_consistency": float(consistency[gene]),
                    "selected": bool(gene in selected),
                }
            )
    gene_scores = pd.DataFrame(rows)
    if not gene_scores.empty:
        gene_scores = gene_scores.sort_values(
            ["component", "score", "gene_index"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    return marker_indices, gene_scores


def generalized_kl_per_mass(observed: np.ndarray, expected: np.ndarray) -> float:
    observed = np.maximum(np.asarray(observed, dtype=float), 0.0)
    expected = np.maximum(np.asarray(expected, dtype=float), EPS)
    term = expected - observed
    positive = observed > 0
    term[positive] += observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return float(2.0 * term.sum() / max(float(observed.sum()), 1.0))


def fit_spgsnmf_v2(
    counts: np.ndarray,
    n_components: int,
    exposure: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    random_state: int = 0,
    max_iter: int = 200,
    tolerance: float = 1e-4,
    n_folds: int = 4,
    max_markers: int = 30,
) -> V2Fit:
    """Fit standalone v2 from ``counts`` and known ``n_components``."""

    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite nonnegative spot-by-gene matrix")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("counts must not contain empty spots")
    n_components = int(n_components)
    if not 1 <= n_components <= min(counts.shape):
        raise ValueError("n_components must be between 1 and min(counts.shape)")
    if exposure is None:
        exposure = compute_exposure(counts)
    exposure = np.maximum(np.asarray(exposure, dtype=float), EPS)
    if exposure.shape != (counts.shape[0],):
        raise ValueError("exposure must contain one value per spot")

    rates = counts / exposure[:, None]
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=int(max_iter),
        tol=float(tolerance),
        random_state=int(random_state),
    )
    W = model.fit_transform(rates)
    H = np.maximum(model.components_, 0.0)
    component_scale = np.maximum(H.sum(axis=1), EPS)
    proportions = row_normalize(W * component_scale[None, :])
    profiles = row_normalize(H)

    marker_indices, gene_scores = select_v2_markers(
        counts,
        exposure,
        proportions,
        profiles,
        coords=coords,
        n_folds=n_folds,
        max_markers=max_markers,
        random_state=random_state,
    )
    rate_scale = float(rates.sum() / max(counts.shape[0], 1))
    expected_rates = np.maximum(
        rate_scale * (proportions @ profiles), EPS
    )
    diagnostics = {
        "method": "standalone_exposure_balanced_allgene_kl_nmf",
        "nmf_iterations": int(model.n_iter_),
        "nmf_reconstruction_error": float(model.reconstruction_err_),
        "rate_kl_deviance_per_mass": generalized_kl_per_mass(
            rates, expected_rates
        ),
        "selected_marker_counts": {
            str(component): int(len(marker_indices[component]))
            for component in range(n_components)
        },
        "truth_used": False,
        "cross_version_input_used": False,
    }
    return V2Fit(
        proportions=proportions,
        profiles=profiles,
        marker_indices=marker_indices,
        gene_scores=gene_scores,
        diagnostics=diagnostics,
    )
