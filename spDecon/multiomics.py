"""Paired multi-omics: shared spot weights, free or fixed assay profiles."""
import numpy as np
import pandas as pd
import torch

from ._numerics import simplex, em_simplex
from ._joint import (JointFit, SpatialLoss, MolecularLoss, device_name, validate_counts,
                     initial_factors, positive_number, optimize, spatial_pair_curvature, validate_edges)


def diagonal_log_simplex(a, curvature, linear):
    """Solve the separable quadratic/log auxiliary with a row-sum constraint."""
    from ._numerics import FLOOR
    width = a.shape[1]
    multipliers = a*width-curvature/width-linear
    lo, hi = multipliers.amin(1, keepdim=True), multipliers.amax(1, keepdim=True)
    def roots(lam):
        b = linear+lam
        disc = torch.sqrt(b.square()+4*curvature*a)
        positive = 2*a/(b+disc).clamp_min(1e-300)
        negative = (disc-b)/(2*curvature).clamp_min(1e-300)
        return torch.where(b >= 0, positive, negative).clamp_min(FLOOR)
    for _ in range(60):
        mid = (lo+hi)/2
        above = roots(mid).sum(1, keepdim=True) > 1
        lo, hi = torch.where(above, mid, lo), torch.where(above, hi, mid)
    return roots((lo+hi)/2)


class MultiOmicsModel:
    def __init__(self, y, graph, k, links, eta, device, fixed=None):
        self.loss, self.graph = MolecularLoss(y, device), graph
        self.k, self.eta, self.fixed = k, eta, fixed
        self.links = [torch.as_tensor(ix, device=device) for ix in links]
        self.q = self.loss.y.sum(0)/self.loss.mass
        self.scales = [len(y), 1.]

    def comparable(self, b):
        return torch.stack([b[:, ix]/self.q[ix] for ix in self.links])

    def value(self, state):
        p, b = state
        kl, tv = self.loss.value(p, b), self.graph.value(p)
        penalty = b.new_zeros(())
        if self.links and self.eta:
            z = self.comparable(b)
            penalty = self.eta/(2*z.numel())*(z-z.mean(0)).square().sum()
        return kl+self.graph.strength*tv+penalty, dict(KL=kl, TV=tv, modality_penalty=penalty)

    def gradients(self, state):
        p, b = state
        gp, gb = self.loss.gradients(p, b)
        if self.links and self.eta:
            z = self.comparable(b)
            for j, ix in enumerate(self.links):
                gb[:, ix] += self.eta/z.numel()*(z[j]-z.mean(0))/self.q[ix]
        if self.fixed is not None:
            gb = torch.zeros_like(b)
        return [gp+self.graph.gradient_curvature(p)[0], gb]

    def project(self, state):
        return [simplex(state[0]), self.fixed if self.fixed is not None else simplex(state[1])]

    def step(self, state):
        p, b = state
        y, mass = self.loss.y, self.loss.mass
        if self.fixed is None:
            counts = b*(p.T@(y/(p@b).clamp_min(1e-300)))/mass
            if self.links and self.eta:
                z = self.comparable(b)
                curvature, linear = torch.zeros_like(b), torch.zeros_like(b)
                factor = self.eta/z.numel()
                for ix in self.links:
                    curvature[:, ix] = factor/self.q[ix].square()
                    linear[:, ix] = -factor*z.mean(0)/self.q[ix]
                b = diagonal_log_simplex(counts, curvature, linear)
            else:
                b = em_simplex(counts)
        counts_p = p*((y/(p@b).clamp_min(1e-300))@b.T)*len(p)/mass
        return [self.graph.step(p, counts_p), b]

    def p_curvature(self, state, enter, leave):
        p, b = state
        db = b[enter]-b[leave]
        h = (self.loss.y*(db/(p@b).clamp_min(1e-300)).square()).sum(1)*len(p)/self.loss.mass
        return h+spatial_pair_curvature(self.graph, p)


def fit_multiomics(modalities, n_components=None, *, references=None, links=None,
                   edges=None, coordinates=None, sections=None, spatial_weight=.05, eta=None,
                   assay_weights=None, initial_proportions=None, initial_profiles=None,
                   random_state=1, device='auto', **optimizer):
    """Fit paired assays ordered by identical spot IDs.

    `modalities` and optional `references` are dictionaries keyed by assay.
    `links` maps assay names to equally sized lists of corresponding feature IDs
    (DataFrames) or zero-based column indices (arrays). Coupling uses only these
    features; reconstruction retains every column. Fixed references disable delta.
    """
    if not isinstance(modalities, dict) or not modalities:
        raise ValueError('Supply at least one named modality')
    names = list(modalities)
    ys = [validate_counts(modalities[name]) for name in names]
    n = len(ys[0])
    if any(len(y) != n for y in ys):
        raise ValueError('Paired modalities must have the same number of spots')
    frames = [isinstance(modalities[name], pd.DataFrame) for name in names]
    if any(frames) and not all(frames):
        raise ValueError('Use DataFrames for every assay or arrays for every assay')
    if all(frames) and any(not modalities[name].index.equals(modalities[names[0]].index) for name in names):
        raise ValueError('Paired assay spot IDs/order differ')
    weights = np.ones(len(names))/len(names) if assay_weights is None else np.asarray(assay_weights, float)
    if weights.shape != (len(names),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('One positive balancing weight is required per assay')
    weights = weights/weights.sum()
    scales = weights/np.array([y.sum() for y in ys])
    scales = scales/scales[0]
    y = np.concatenate([scale*a for scale, a in zip(scales, ys)], axis=1)
    offsets = np.r_[0, np.cumsum([a.shape[1] for a in ys])]
    fixed, indices = None, []
    device = device_name(device)
    if references is not None:
        if set(references) != set(names):
            raise ValueError('Supply reference profiles for every modality')
        if eta is not None and eta != 0:
            raise ValueError('Fixed-reference multi-omics has no learned delta; eta must be zero')
        from .reference import _validate
        k = len(references[names[0]])
        if n_components is not None and n_components != k:
            raise ValueError('K must match the number of reference types')
        bs = []
        for name, weight in zip(names, weights):
            if isinstance(references[name], pd.DataFrame) and isinstance(references[names[0]], pd.DataFrame):
                if not references[name].index.equals(references[names[0]].index):
                    raise ValueError('Reference type IDs/order must agree across modalities')
            _, b, _, _ = _validate(modalities[name], k, references[name], None, None, 1e-12)
            bs.append(weight*b)
        b0 = np.concatenate(bs, axis=1)
        p0 = np.full((n, k), 1/k) if initial_proportions is None else np.asarray(initial_proportions, float)
        if initial_profiles is not None:
            raise ValueError('Fixed mode takes profiles from references')
        if p0.shape != (n, k) or not np.isfinite(p0).all() or (p0 < 0).any() or (p0.sum(1) <= 0).any():
            raise ValueError('Invalid initial proportions')
        fixed = torch.as_tensor(b0, device=device)
        eta = 0.
    else:
        k = n_components
        eta = positive_number(10. if eta is None else eta, 'eta', True)
        if eta and (links is None or set(links) != set(names)):
            raise ValueError('Free delta fitting requires matched features for every modality')
        if links is not None:
            if set(links) != set(names):
                raise ValueError('Feature links must cover every modality')
            for j, name in enumerate(names):
                raw = list(links[name])
                ix = modalities[name].columns.get_indexer(raw) if frames[j] else np.asarray(raw)
                if len(ix) < 1 or not np.issubdtype(ix.dtype, np.integer) or (ix < 0).any() or (ix >= ys[j].shape[1]).any() or len(np.unique(ix)) != len(ix):
                    raise ValueError('Links must contain unique valid feature IDs/indices')
                if indices and len(ix) != len(indices[0]):
                    raise ValueError('Each modality needs the same number of matched features')
                indices.append(ix+offsets[j])
            if any((y[:, ix].sum(0) <= 0).any() for ix in indices):
                raise ValueError('Matched features must have positive observed signal')
        p0, b0 = initial_factors(y, k, initial_proportions, initial_profiles, random_state, device)
    if edges is None and spatial_weight:
        if coordinates is None:
            raise ValueError('Supply edges or coordinates for spatial fitting')
        from .spatial import spatial_graph
        i, j, _, _ = spatial_graph(coordinates, np.repeat('0', n) if sections is None else sections)
        edges = np.column_stack((i, j))
    edges = validate_edges(edges, n, sections)
    graph = SpatialLoss(edges, None, n, device, spatial_weight)
    model = MultiOmicsModel(y, graph, k, indices, eta, device, fixed)
    state, history, diag = optimize(model, [torch.as_tensor(p0, device=device), torch.as_tensor(b0, device=device)], **optimizer)
    extra = dict(assays=np.asarray(names, str), feature_offsets=offsets, edges=edges,
                 assay_scales=scales, assay_weights=weights)
    if indices:
        z = model.comparable(state[1])/len(indices[0])
        extra.update(shared_comparable_profiles=z.mean(0).cpu().numpy(),
                     delta_modality=(z-z.mean(0)).cpu().numpy(), matched_indices=np.stack(indices))
    diag.update(mode='multiomics_fixed' if references is not None else 'multiomics_free',
                eta=eta, spatial_weight=spatial_weight, paired=True, reference_profiles_fixed=references is not None)
    return JointFit(state[0].cpu().numpy(), state[1].cpu().numpy(), history, diag, extra)
