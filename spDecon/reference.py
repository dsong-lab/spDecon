"""Fixed-reference KL fitting and reference preparation."""
import time
import numpy as np
import pandas as pd
import torch

from ._joint import JointFit, device_name, validate_counts


def fit_reference_consensus(counts, reference_profiles, *, random_state=0):
    """Manuscript ATAC: three raw initializers, fixed-profile fits and consensus.

    CPU float64; original 500-outer-iteration solver; P KKT tolerance 1e-10
    and relative-objective tolerance 1e-12.
    """
    from ._atac_reference.consensus_raw_poisson import fit_consensus_raw_poisson
    return fit_consensus_raw_poisson(counts, len(reference_profiles), random_state=random_state,
                                    reference_profiles=reference_profiles, reference_mode='fixed',
                                    reference_config=dict(p_kkt_tolerance=1e-10, relative_tolerance=1e-12))


def _array(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def _ids(ids, n, name):
    values = list(ids)
    if len(values) != n or len(set(values)) != n:
        raise ValueError(f"{name} must have one unique ID per gene")
    return values


def _validate(counts, n_components, reference_profiles, gene_ids,
              reference_gene_ids, reference_floor):
    for frame in (counts, reference_profiles):
        if isinstance(frame, pd.DataFrame) and (not frame.index.is_unique or not frame.columns.is_unique):
            raise ValueError("Spot, type and feature IDs must be unique")
    # DataFrame inputs can carry their own gene identities. Never reorder a
    # reference silently: the caller must explicitly establish the shared panel.
    if gene_ids is None and hasattr(counts, "columns"):
        gene_ids = counts.columns
    if reference_gene_ids is None and hasattr(reference_profiles, "columns"):
        reference_gene_ids = reference_profiles.columns
    y, r = _array(counts), _array(reference_profiles)
    if y.ndim != 2 or min(y.shape) < 1 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("counts must be a nonempty finite nonnegative spot-by-gene matrix")
    if (y.sum(axis=1) <= 0).any():
        raise ValueError("counts contain an empty spot")
    if isinstance(n_components, bool) or int(n_components) != n_components or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    k = int(n_components)
    if r.shape != (k, y.shape[1]):
        raise ValueError("reference_profiles must have shape (n_components, counts.shape[1])")
    if not np.isfinite(r).all() or (r < 0).any():
        raise ValueError("reference_profiles must be finite and nonnegative")
    if (r.sum(axis=1) <= 0).any():
        raise ValueError("reference_profiles contain an empty component spectrum")
    if (gene_ids is None) != (reference_gene_ids is None):
        raise ValueError("provide both gene_ids and reference_gene_ids, or neither")
    if gene_ids is not None:
        left = _ids(gene_ids, y.shape[1], "gene_ids")
        right = _ids(reference_gene_ids, y.shape[1], "reference_gene_ids")
        if left != right:
            raise ValueError("counts and reference gene IDs/order differ; align explicitly before fitting")
    if not np.isfinite(reference_floor) or not 0 < reference_floor < 1 / y.shape[1]:
        raise ValueError("reference_floor must be positive and smaller than 1 / n_genes")
    r = r / r.sum(axis=1, keepdims=True)
    smoothed_entries = int((r < reference_floor).sum())
    r = np.maximum(r, reference_floor)
    r /= r.sum(axis=1, keepdims=True)
    return y, r, smoothed_entries, gene_ids is not None


def _project_simplex(x: torch.Tensor) -> torch.Tensor:
    """Exact Euclidean projection onto x >= 0, sum(x)=1, independently by row."""
    ordered = torch.sort(x, dim=1, descending=True).values
    cumulative = ordered.cumsum(dim=1) - 1.0
    j = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)[None]
    support = (ordered - cumulative / j > 0).sum(dim=1).clamp(min=1)
    threshold = cumulative.gather(1, (support - 1)[:, None]) / support[:, None]
    return (x - threshold).clamp(min=0.0)


def _mapping(x, gradient):
    return float((x - _project_simplex(x - gradient)).abs().max().item())


def _p_block(q, p, b, maximum_steps, tolerance, steps=None):
    """Rowwise projected gradient with BB proposals and Armijo safeguards.

    Each row minimizes -sum_g (Y_ig / s_i) log((PB)_ig). This positive
    rescaling removes spot-library imbalance from the numerical stopping
    criterion but leaves each spot's MLE and global-objective descent intact.
    """
    mixture = p @ b
    objective = -(q * mixture.log()).sum(dim=1)
    gradient = -(q / mixture) @ b.T
    n = len(p)
    if steps is None:
        scale = (gradient - gradient.mean(dim=1, keepdim=True)).abs().amax(dim=1)
        steps = (0.1 / scale.clamp(min=1e-8)).clamp(min=1e-10, max=1e6)
    accepted_total = 0
    backtracks_total = 0
    for _ in range(maximum_steps):
        mapping = (p - _project_simplex(p - gradient)).abs().amax(dim=1)
        active = mapping > tolerance
        if not bool(active.any()):
            break
        trial = steps.clone()
        accepted = ~active
        new_p = p.clone()
        new_obj = objective.clone()
        for backtrack in range(45):
            candidate = _project_simplex(p - trial[:, None] * gradient)
            candidate_obj = -(q * (candidate @ b).log()).sum(dim=1)
            armijo = objective + 1e-4 * (gradient * (candidate - p)).sum(dim=1)
            valid = (~accepted) & torch.isfinite(candidate_obj) & (candidate_obj <= armijo + 2e-14)
            new_p[valid] = candidate[valid]
            new_obj[valid] = candidate_obj[valid]
            accepted |= valid
            if bool(accepted.all()):
                break
            trial[~accepted] *= 0.5
            backtracks_total += int((~accepted).sum().item())
        new_gradient = -(q / (new_p @ b)) @ b.T
        displacement = new_p - p
        gradient_change = new_gradient - gradient
        denom = (displacement * gradient_change).sum(dim=1)
        bb = displacement.square().sum(dim=1) / denom.clamp(min=1e-300)
        steps = torch.where(denom > 1e-24, bb, trial * 1.5).clamp(min=1e-10, max=1e6)
        moved = displacement.abs().amax(dim=1) > 1e-15
        accepted_total += int(moved.sum().item())
        p, objective, gradient = new_p, new_obj, new_gradient
        if not bool(moved.any()):
            break
    return p, steps, {
        "p_conditional_kkt": _mapping(p, gradient),
        "p_accepted_row_steps": accepted_total,
        "p_backtracks": backtracks_total,
    }


def fit_reference(counts, reference_profiles, *, gene_ids=None, reference_gene_ids=None,
                  initial_proportions=None, reference_floor=1e-12, max_steps=3000,
                  tolerance=1e-6, device='auto'):
    """Fix the supplied reference profiles and optimize only the spot weights.

    Defaults use the single uniform start and rowwise BB/Armijo solver of the
    independent STARmap control. The three primary RNA top100 recipes instead
    use the strict solver in reference_spatial.py. No smoothing is applied here.
    """
    started = time.perf_counter()
    if isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be finite and positive')
    k = len(reference_profiles)
    y, r, smoothed, checked = _validate(counts, k, reference_profiles, gene_ids,
                                       reference_gene_ids, reference_floor)
    device = device_name(device)
    p0 = np.full((len(y), k), 1/k) if initial_proportions is None else np.asarray(initial_proportions, float)
    if p0.shape != (len(y), k) or not np.isfinite(p0).all() or (p0 < 0).any() or (p0.sum(1) <= 0).any():
        raise ValueError('Invalid initial spot weights')
    with torch.no_grad():
        yt = torch.as_tensor(y, device=device)
        rt = torch.as_tensor(r, device=device)
        p = _project_simplex(torch.as_tensor(p0/p0.sum(1, keepdims=True), device=device))
        q = yt/yt.sum(1, keepdim=True)
        def loss(a):
            mean = yt.sum(1, keepdim=True)*(a@rt)
            mask = yt > 0
            return float(((mean-yt).sum()+(yt[mask]*torch.log(yt[mask]/mean[mask])).sum())/yt.sum())
        history = [dict(iteration=0, objective=loss(p))]
        p, _, info = _p_block(q, p, rt, max_steps, tolerance)
        history.append(dict(iteration=1, objective=loss(p), **info))
    diag = dict(mode='fixed_reference', converged=info['p_conditional_kkt'] <= tolerance,
                objective=history[-1]['objective'], stopping_KKT=info['p_conditional_kkt'],
                reference_profiles_fixed=True, initialization='uniform' if initial_proportions is None else 'supplied',
                reference_floor=reference_floor, floored_entries=smoothed, feature_ids_checked=checked,
                max_steps=max_steps, tolerance=tolerance, device=device, dtype='float64',
                runtime_sec=time.perf_counter()-started)
    diag['termination_reason'] = 'conditional_P_KKT' if diag['converged'] else 'iteration_limit_or_stalled'
    return JointFit(p.cpu().numpy(), r.copy(), history, diag)


def cell_fractions(signal_weights, reference_signal_per_cell):
    """Optional reference-yield conversion; use yields from the fitted panel."""
    p, yields = np.asarray(signal_weights, float), np.asarray(reference_signal_per_cell, float)
    if p.ndim != 2 or yields.shape != (p.shape[1],) or not np.isfinite(yields).all() or (yields <= 0).any():
        raise ValueError('One finite positive reference yield is required per type')
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any():
        raise ValueError('Invalid signal weights')
    values = p/yields
    return values/values.sum(1, keepdims=True)


def prepare_reference(reference_cells, cell_types, *, markers=None, balanced=True):
    """Construct type means on the full shared panel, then subset markers.

    Pass raw cell counts as a DataFrame. `markers` may come from COSG or a
    user-defined panel. Returned unnormalized row sums are panel-specific yields.
    """
    if not isinstance(reference_cells, pd.DataFrame):
        raise ValueError('reference_cells must be a cell-by-feature DataFrame')
    x = validate_counts(reference_cells)
    if isinstance(cell_types, pd.Series) and not reference_cells.index.equals(cell_types.index):
        raise ValueError('Reference cells and type labels must have identical IDs/order')
    labels = np.asarray(cell_types)
    if labels.shape != (len(x),) or pd.isna(labels).any() or (labels.astype(str) == '').any():
        raise ValueError('One nonempty cell-type label is required per reference cell')
    types = pd.unique(labels)
    libraries = x.sum(1)
    means = np.stack([((x[labels == t]/libraries[labels == t, None]).mean(0)*libraries[labels == t].mean()
                       if balanced else x[labels == t].mean(0)) for t in types])
    frame = pd.DataFrame(means, index=types, columns=reference_cells.columns)
    if markers is not None:
        markers = list(markers)
        if not markers or len(set(markers)) != len(markers) or not set(markers).issubset(frame.columns):
            raise ValueError('Markers must be unique features from the shared panel')
        frame = frame.loc[:, markers]
    if (frame.sum(1) <= 0).any():
        raise ValueError('The fitting panel leaves a reference type without signal')
    return frame, frame.sum(1).rename('reference_signal_per_cell')
