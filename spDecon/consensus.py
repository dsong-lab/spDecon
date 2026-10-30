"""Profile alignment and equal-weight consensus of the three fitted branches."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .initialization import fit_initializations

EPS = 1e-12
BRANCH_NAMES = ("all_features", "variable_features", "expression_clusters")


@dataclass
class ConsensusFit:
    consensus_proportions: np.ndarray
    consensus_profiles: np.ndarray
    branch_proportions: dict[str, np.ndarray]
    branch_profiles: dict[str, np.ndarray]
    alignments_to_all_gene: dict[str, np.ndarray]
    diagnostics: dict[str, object]


def _row_normalize(values):
    values = np.maximum(np.asarray(values, dtype=float), EPS)
    return values / np.maximum(values.sum(axis=1, keepdims=True), EPS)


def align_to_reference_profiles(reference_profiles, proportions, profiles):
    """Hungarian-align a branch using profile cosine similarity."""
    reference_profiles = np.asarray(reference_profiles, dtype=float)
    proportions = np.asarray(proportions, dtype=float)
    profiles = np.asarray(profiles, dtype=float)
    if profiles.shape != reference_profiles.shape:
        raise ValueError("Branch and reference profiles have incompatible shapes")
    if proportions.ndim != 2 or proportions.shape[1] != len(profiles):
        raise ValueError("Branch proportions have an incompatible shape")
    reference_unit = reference_profiles / np.maximum(
        np.linalg.norm(reference_profiles, axis=1, keepdims=True), EPS
    )
    branch_unit = profiles / np.maximum(
        np.linalg.norm(profiles, axis=1, keepdims=True), EPS
    )
    similarity = reference_unit @ branch_unit.T
    reference_index, branch_index = linear_sum_assignment(-similarity)
    order = np.empty(len(reference_profiles), dtype=int)
    order[reference_index] = branch_index
    return proportions[:, order], profiles[order], order, similarity[np.arange(len(order)), order]


def build_consensus(counts, n_components, *, random_state=0, solver_config=None):
    """Fit three branches and return the unrefined consensus for the TV fit."""
    fits = fit_initializations(counts, n_components, random_state, solver_config=solver_config)
    reference = fits["all_features"]
    reference_p = np.asarray(reference.proportions, dtype=float)
    reference_b = np.asarray(reference.profiles, dtype=float)
    aligned_p = {"all_features": reference_p}
    aligned_b = {"all_features": reference_b}
    alignments = {"all_features": np.arange(n_components, dtype=int)}
    similarities = {"all_features": np.ones(n_components).tolist()}
    for name in BRANCH_NAMES[1:]:
        fit = fits[name]
        p, b, order, similarity = align_to_reference_profiles(
            reference_b, fit.proportions, fit.profiles
        )
        aligned_p[name], aligned_b[name] = p, b
        alignments[name], similarities[name] = order, similarity.tolist()
    p0 = _row_normalize(np.mean([aligned_p[name] for name in BRANCH_NAMES], axis=0))
    b0 = _row_normalize(np.mean([aligned_b[name] for name in BRANCH_NAMES], axis=0))
    return ConsensusFit(
        p0, b0, aligned_p, aligned_b, alignments,
        {"branch_diagnostics": {name: fit.diagnostics for name, fit in fits.items()},
         "profile_alignment_cosines": similarities,
         "branch_weights": {name: 1 / 3 for name in BRANCH_NAMES},
         "truth_used": False},
    )
