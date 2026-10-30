"""Optional reference interface inside the original three-initializer core.

Original initial factors are retained in soft mode. Reference information
aligns their component identities and constrains every subsequent B update.
No standalone reference-regression solver is imported by this module.
"""
from dataclasses import dataclass, replace
import time

import numpy as np
import torch

from spDecon._atac_reference.poisson_hmrf import PoissonHMRFConfig, fit_poisson_hmrf
from spDecon._atac_reference.poisson_hmrf.model import project_rows_lower_simplex
from spDecon._atac_reference.poisson_hmrf.independent_raw import initialize_raw_count_poisson
from spDecon._atac_reference.rare_init_poisson.model import initialize_rare_poisson, _raw_config
from spDecon._atac_reference.louvain_init_poisson.model import initialize_louvain_poisson
from .model import BRANCH_NAMES, ConsensusRawPoissonFit, align_to_reference_profiles, _row_normalize


@dataclass
class ReferenceConsensusFit(ConsensusRawPoissonFit):
    reference_profiles: np.ndarray
    branch_initial_proportions: dict[str, np.ndarray]
    branch_initial_profiles: dict[str, np.ndarray]
    alignments_to_reference: dict[str, np.ndarray]
    branch_initialization_diagnostics: dict[str, dict]
    branch_histories: dict[str, list]


def _ids(values, n, name):
    values = list(values)
    if len(values) != n or len(set(values)) != n:
        raise ValueError(f"{name} must contain one unique ID per gene")
    return values


def _prepare(counts, n_components, reference, gene_ids, reference_gene_ids, config):
    if gene_ids is None and hasattr(counts, 'columns'):
        gene_ids = counts.columns
    if reference_gene_ids is None and hasattr(reference, 'columns'):
        reference_gene_ids = reference.columns
    if hasattr(reference, 'columns') and reference.index.has_duplicates:
        raise ValueError("reference type identifiers must be unique")
    y = np.asarray(counts.toarray() if hasattr(counts, 'toarray') else counts, dtype=float)
    r = np.asarray(reference.toarray() if hasattr(reference, 'toarray') else reference, dtype=float)
    if y.ndim != 2 or min(y.shape) < 1 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("counts must be a nonempty finite nonnegative matrix")
    if (y.sum(axis=1) <= 0).any():
        raise ValueError("counts contain an empty sample")
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise ValueError("n_components must be an integer")
    if not 1 <= n_components <= min(y.shape):
        raise ValueError("n_components is incompatible with counts shape")
    if r.shape != (n_components, y.shape[1]):
        raise ValueError("reference_profiles must have shape (K, genes)")
    if not np.isfinite(r).all() or (r < 0).any() or (r.sum(axis=1) <= 0).any():
        raise ValueError("reference must be nonnegative and finite with nonempty type profiles")
    if (gene_ids is None) != (reference_gene_ids is None):
        raise ValueError("supply both gene_ids and reference_gene_ids, or neither")
    if gene_ids is not None:
        left = _ids(gene_ids, y.shape[1], 'gene_ids')
        right = _ids(reference_gene_ids, y.shape[1], 'reference_gene_ids')
        if left != right:
            raise ValueError("counts and reference gene IDs/order differ; align them explicitly")
    if config.lower_bound * max(n_components, y.shape[1]) >= 1:
        raise ValueError("simplex lower_bound is incompatible with K or gene count")
    normalized = r / r.sum(axis=1, keepdims=True)
    effective = project_rows_lower_simplex(torch.as_tensor(normalized, dtype=torch.float64), config.lower_bound).numpy()
    return y, effective, {'gene_order_checked': gene_ids is not None,
                          'reference_preparation': 'row normalization then original lower-simplex projection',
                          'reference_max_adjustment_after_normalization': float(np.abs(effective - normalized).max())}


def fit_reference_consensus(counts, n_components, *, random_state,
                            reference_profiles, reference_mode, reference_strength,
                            gene_ids, reference_gene_ids, reference_config):
    started = time.perf_counter()
    if reference_mode not in ('soft', 'fixed'):
        raise ValueError("reference_mode must be soft or fixed")
    if not np.isfinite(reference_strength) or reference_strength < 0:
        raise ValueError("reference_strength must be finite and nonnegative")
    if reference_mode == 'soft' and reference_strength <= 0:
        raise ValueError("soft reference_strength must be positive")
    cfg = replace(_raw_config(random_state), device='cpu', dtype='float64',
                  lower_bound=1e-12, p_kkt_tolerance=1e-6, b_kkt_tolerance=1e-6,
                  relative_tolerance=1e-9)
    if reference_config is not None:
        cfg = replace(cfg, **reference_config) if isinstance(reference_config, dict) else reference_config
        if not isinstance(cfg, PoissonHMRFConfig):
            raise TypeError("reference_config must be a PoissonHMRFConfig or a dictionary")
        cfg = replace(cfg, random_state=random_state)
    cfg.validate()
    if cfg.method != 'rna_only_poisson' or cfg.coordinate_permutation:
        raise ValueError("the reference consensus interface is count-only")
    y, r, preparation = _prepare(counts, n_components, reference_profiles, gene_ids, reference_gene_ids, cfg)
    k = int(n_components)
    rare_count = min(5000, max(k, int(np.ceil(.25 * y.shape[1]))))
    initializers = (
        ('all_gene', initialize_raw_count_poisson, {}),
        ('rare_5000', initialize_rare_poisson, {'n_initialization_features': rare_count}),
        ('louvain', initialize_louvain_poisson, {}),
    )
    branch_p, branch_b, initial_p, initial_b = {}, {}, {}, {}
    alignments, cosines, initial_diagnostics, branch_diagnostics, histories = {}, {}, {}, {}, {}
    coordinates = np.zeros((len(y), 2), dtype=float)
    branch_started = time.perf_counter()
    for name, initialize, extra in initializers:
        one_started = time.perf_counter()
        initialization = initialize(y, k, random_state=random_state, **extra)
        p0, b0, order, similarity = align_to_reference_profiles(r, initialization.proportions, initialization.profiles)
        initial_p[name], initial_b[name] = p0.copy(), b0.copy()
        alignments[name], cosines[name] = order, similarity.tolist()
        initial_diagnostics[name] = dict(initialization.diagnostics)
        initial_diagnostics[name]['selected_feature_indices'] = (
            None if initialization.selected_feature_indices is None else initialization.selected_feature_indices.tolist())
        branch = fit_poisson_hmrf(y, coordinates, k, p0, r if reference_mode == 'fixed' else b0,
                                 cfg, reference_profiles=r, reference_mode=reference_mode,
                                 reference_strength=reference_strength)
        branch_p[name], branch_b[name] = branch.proportions, branch.profiles
        branch_diagnostics[name] = {**branch.diagnostics, 'runtime_sec': time.perf_counter() - one_started}
        histories[name] = branch.history
    branch_runtime = time.perf_counter() - branch_started
    effective_r = r.astype(np.float64 if cfg.dtype == 'float64' else np.float32).astype(float)
    consensus_p = _row_normalize(np.mean([branch_p[name] for name in BRANCH_NAMES], axis=0))
    consensus_b = (effective_r.copy() if reference_mode == 'fixed' else
                   _row_normalize(np.mean([branch_b[name] for name in BRANCH_NAMES], axis=0)))
    final_started = time.perf_counter()
    final = fit_poisson_hmrf(y, coordinates, k, consensus_p, consensus_b, cfg,
                            reference_profiles=r, reference_mode=reference_mode,
                            reference_strength=reference_strength)
    final_runtime = time.perf_counter() - final_started
    diagnostics = {**final.diagnostics, **preparation,
                   'method': 'reference-guided three-initializer core consensus',
                   'branch_names': list(BRANCH_NAMES), 'rare_branch_feature_count': rare_count,
                   'branch_weights': {name: 1/3 for name in BRANCH_NAMES},
                   'branch_diagnostics': branch_diagnostics, 'profile_alignment_cosines': cosines,
                   'branch_runtime_sec': branch_runtime, 'consensus_refinement_runtime_sec': final_runtime,
                   'standalone_runtime_sec': time.perf_counter() - started,
                   'original_three_initializers_used': True,
                   'free_PB_refinement_before_reference_used': False,
                   'original_initial_B_preserved_in_soft_mode': reference_mode == 'soft',
                   'component_order': 'external reference row order',
                   'all_branches_converged': all(d['converged'] for d in branch_diagnostics.values()),
                   'final_refinement_converged': final.diagnostics['converged'],
                   'feature_knn_graph_used_by_louvain_initializer': True,
                   'coordinates_used': False, 'spatial_graph_used': False,
                   'truth_used': False}
    diagnostics['converged'] = bool(diagnostics['all_branches_converged'] and diagnostics['final_refinement_converged'])
    if not diagnostics['all_branches_converged']:
        diagnostics['termination_reason'] = 'one_or_more_reference_branches_did_not_converge'
    return ReferenceConsensusFit(
        proportions=final.proportions, profiles=final.profiles,
        consensus_proportions=consensus_p, consensus_profiles=consensus_b,
        branch_proportions=branch_p, branch_profiles=branch_b,
        alignments_to_all_gene={}, history=final.history, diagnostics=diagnostics,
        config={'reference_mode': reference_mode,
                'reference_strength': reference_strength if reference_mode == 'soft' else 0.,
                'initialization': 'unaltered original all-gene/residual/Louvain P0 and B0, reference-aligned',
                'alignment': 'initial-profile cosine/Hungarian to reference; no subsequent relabeling',
                'branch_weights': 'equal fixed one-third weights',
                'final_objective': 'KL(Y||diag(s)PB)/sum(Y) + lambda/K * sum KL(R||B)' if reference_mode == 'soft' else 'KL(Y||diag(s)PR)/sum(Y)',
                'refinement': cfg.to_dict()},
        reference_profiles=effective_r, branch_initial_proportions=initial_p,
        branch_initial_profiles=initial_b, alignments_to_reference=alignments,
        branch_initialization_diagnostics=initial_diagnostics, branch_histories=histories)
