"""Unsupervised consensus of complementary raw-count Poisson basins."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from louvain_init_poisson import fit_louvain_init_poisson
from poisson_hmrf import fit_poisson_hmrf
from rare_init_poisson import fit_rare_init_poisson
from rare_init_poisson.model import _raw_config
from raw_count_poisson import FROZEN_CONFIG as RAW_FROZEN_CONFIG
from raw_count_poisson import fit_raw_count_poisson


EPS = 1e-12
BRANCH_NAMES = ("all_gene", "rare_5000", "louvain")


@dataclass
class ConsensusRawPoissonFit:
    proportions: np.ndarray
    profiles: np.ndarray
    consensus_proportions: np.ndarray
    consensus_profiles: np.ndarray
    branch_proportions: dict[str, np.ndarray]
    branch_profiles: dict[str, np.ndarray]
    alignments_to_all_gene: dict[str, np.ndarray]
    history: list[dict[str, object]]
    diagnostics: dict[str, object]
    config: dict[str, object]


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), EPS)
    return values / np.maximum(values.sum(axis=1, keepdims=True), EPS)


def align_to_reference_profiles(
    reference_profiles: np.ndarray,
    proportions: np.ndarray,
    profiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian-align a branch using profile cosine only (never truth)."""

    reference_profiles = np.asarray(reference_profiles, dtype=float)
    proportions = np.asarray(proportions, dtype=float)
    profiles = np.asarray(profiles, dtype=float)
    if profiles.shape != reference_profiles.shape:
        raise ValueError("branch and reference profiles have incompatible shapes")
    if proportions.ndim != 2 or proportions.shape[1] != len(profiles):
        raise ValueError("branch proportions have an incompatible shape")
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
    return (
        proportions[:, order],
        profiles[order],
        order,
        similarity[np.arange(len(order)), order],
    )


def fit_consensus_raw_poisson(
    counts: np.ndarray,
    n_components: int,
    *,
    random_state: int = 0,
) -> ConsensusRawPoissonFit:
    """Fit three fixed count-only basins, average them, then reconverge KL.

    The branches are fixed in advance: frozen all-gene raw Poisson, the
    5,000-feature exposure-residual initializer, and the SpiceMix-style
    nonspatial Louvain initializer.  Equal weights are used after profile-only
    Hungarian alignment.  No metric, annotation, coordinate, or other modality
    is available to the estimator.
    """

    started = time.perf_counter()
    counts = np.asarray(counts, dtype=float)
    n_components = int(n_components)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite nonnegative matrix")
    # The gold-standard primary used 5,000/20,000 = 25% of features.  Preserve
    # that fraction for smaller panels while retaining the preregistered 5,000
    # cap on larger matrices.
    rare_feature_count = min(
        5000,
        max(n_components, int(np.ceil(0.25 * counts.shape[1]))),
    )

    branch_started = time.perf_counter()
    all_gene = fit_raw_count_poisson(
        counts, n_components, random_state=int(random_state)
    )
    rare = fit_rare_init_poisson(
        counts,
        n_components,
        n_initialization_features=rare_feature_count,
        random_state=int(random_state),
    )
    louvain = fit_louvain_init_poisson(
        counts, n_components, random_state=int(random_state)
    )
    branch_runtime = time.perf_counter() - branch_started

    reference_p = np.asarray(all_gene.proportions, dtype=float)
    reference_b = np.asarray(all_gene.profiles, dtype=float)
    aligned_p: dict[str, np.ndarray] = {"all_gene": reference_p}
    aligned_b: dict[str, np.ndarray] = {"all_gene": reference_b}
    alignments: dict[str, np.ndarray] = {
        "all_gene": np.arange(n_components, dtype=int)
    }
    similarities: dict[str, list[float]] = {
        "all_gene": np.ones(n_components).tolist()
    }
    for name, fit in (("rare_5000", rare), ("louvain", louvain)):
        branch_p, branch_b, order, similarity = align_to_reference_profiles(
            reference_b, fit.proportions, fit.profiles
        )
        aligned_p[name] = branch_p
        aligned_b[name] = branch_b
        alignments[name] = order
        similarities[name] = similarity.tolist()

    consensus_p = _row_normalize(
        np.mean([aligned_p[name] for name in BRANCH_NAMES], axis=0)
    )
    consensus_b = _row_normalize(
        np.mean([aligned_b[name] for name in BRANCH_NAMES], axis=0)
    )
    refinement_started = time.perf_counter()
    refined = fit_poisson_hmrf(
        counts,
        np.zeros((counts.shape[0], 2), dtype=float),
        n_components,
        consensus_p,
        consensus_b,
        _raw_config(int(random_state)),
    )
    refinement_runtime = time.perf_counter() - refinement_started
    total_runtime = time.perf_counter() - started
    branch_diagnostics = {
        "all_gene": {
            "converged": bool(all_gene.diagnostics["converged"]),
            "outer_iterations": int(all_gene.diagnostics["outer_iterations"]),
            "runtime_sec": float(all_gene.diagnostics["standalone_runtime_sec"]),
        },
        "rare_5000": {
            "converged": bool(rare.diagnostics["converged"]),
            "outer_iterations": int(rare.diagnostics["outer_iterations"]),
            "runtime_sec": float(rare.diagnostics["standalone_runtime_sec"]),
        },
        "louvain": {
            "converged": bool(louvain.diagnostics["converged"]),
            "outer_iterations": int(louvain.diagnostics["outer_iterations"]),
            "runtime_sec": float(louvain.diagnostics["standalone_runtime_sec"]),
        },
    }
    diagnostics = dict(refined.diagnostics)
    diagnostics.update(
        {
            "method": "multi-basin consensus raw-count Poisson",
            "branch_names": list(BRANCH_NAMES),
            "rare_branch_feature_count": int(rare_feature_count),
            "branch_weights": {name: 1.0 / len(BRANCH_NAMES) for name in BRANCH_NAMES},
            "branch_diagnostics": branch_diagnostics,
            "profile_alignment_cosines": similarities,
            "branch_runtime_sec": float(branch_runtime),
            "consensus_refinement_runtime_sec": float(refinement_runtime),
            "standalone_runtime_sec": float(total_runtime),
            "coordinates_used": False,
            "spatial_graph_used": False,
            "feature_knn_graph_used_by_louvain_initializer": True,
            "external_warm_start_used": False,
            "truth_used": False,
            "annotation_used": False,
            "other_modality_used": False,
            "consensus_component_mass": consensus_p.mean(axis=0).tolist(),
            "final_component_mass": refined.proportions.mean(axis=0).tolist(),
            "initial_raw_poisson_kl_per_count": float(
                refined.diagnostics["initial_poisson_kl_per_count"]
            ),
            "final_raw_poisson_kl_per_count": float(
                refined.diagnostics["final_poisson_kl_per_count"]
            ),
        }
    )
    config = {
        "input": "literal raw counts",
        "branches": {
            "all_gene": "frozen raw-count Poisson v1.0.0",
            "rare_5000": (
                "top 25% exposure-residual features, capped at 5,000 "
                f"(used {rare_feature_count})"
            ),
            "louvain": "fixed SpiceMix-style count-only Louvain basin",
        },
        "alignment": "Hungarian maximum profile cosine to all-gene branch",
        "branch_weights": "equal fixed one-third weights",
        "primary_output": "consensus followed by converged raw-count Poisson refinement",
        "secondary_output": "unrefined equal-weight consensus",
        "final_objective": RAW_FROZEN_CONFIG["model"],
    }
    return ConsensusRawPoissonFit(
        proportions=refined.proportions,
        profiles=refined.profiles,
        consensus_proportions=consensus_p,
        consensus_profiles=consensus_b,
        branch_proportions=aligned_p,
        branch_profiles=aligned_b,
        alignments_to_all_gene=alignments,
        history=refined.history,
        diagnostics=diagnostics,
        config=config,
    )
