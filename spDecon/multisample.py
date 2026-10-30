"""Joint sample profiles, optionally nested within subjects/conditions."""
import numpy as np
import pandas as pd
import torch

from ._numerics import simplex, em_simplex, quadratic_log_simplex
from ._joint import (JointFit, SpatialLoss, initial_factors, validate_counts,
                     validate_edges, device_name, optimize, positive_number,
                     spatial_pair_curvature)


def profile_mstep(counts, profiles, q, curvature, inner_cycles=5):
    """Five auxiliary MM cycles with the manuscript's safeguarded extrapolation."""
    width = profiles.shape[-1]
    c = profiles.new_full((profiles.shape[0] * profiles.shape[1],), curvature)

    def step(b):
        gradient = (q @ b.reshape(len(b), -1)).reshape_as(b)
        linear = (gradient - curvature * b).reshape(-1, width)
        return quadratic_log_simplex(counts.reshape(-1, width), c, linear).reshape_as(b)

    def value(b):
        flat = b.reshape(len(b), -1)
        return float(-(counts * b.log()).sum() + .5 * (flat * (q @ flat)).sum())

    old = value(profiles)
    for _ in range(inner_cycles):
        first = step(profiles)
        second = step(first)
        r, v = first - profiles, second - 2 * first + profiles
        length = float(np.clip(np.sqrt(float(r.square().sum()) /
                                      max(float(v.square().sum()), 1e-30)), 1., 64.))
        chosen, loss = second, value(second)
        if length > 1.05:
            for _ in range(6):
                trial = simplex((profiles + 2 * length * r + length**2 * v)
                                .reshape(-1, width)).reshape_as(profiles)
                candidate = value(trial)
                if np.isfinite(candidate) and candidate <= loss + 1e-13:
                    chosen, loss = trial, candidate
                    break
                length = 1 + (length - 1) / 2
        if loss > old + 1e-10:
            raise RuntimeError('Profile auxiliary increased.')
        profiles = chosen
        if abs(old - loss) <= 1e-12:
            break
        old = loss
    return profiles


def design_matrix(metadata, group_by, subject_column=None):
    """Map spots to profiles; center equally over groups within each subject."""
    if not isinstance(metadata, pd.DataFrame) or not metadata.index.is_unique:
        raise ValueError("metadata must be a DataFrame with unique spot IDs")
    columns = [group_by] if isinstance(group_by, str) else list(group_by)
    required = list(dict.fromkeys(['section_id'] + columns + ([subject_column] if subject_column else [])))
    if not columns or not set(required).issubset(metadata.columns):
        raise ValueError("metadata needs section_id and the requested grouping columns")
    if metadata[required].isna().any().any() or metadata[required].astype(str).eq('').any().any():
        raise ValueError("Design labels cannot be missing or empty")
    frame = metadata[required].astype(str)
    keys = pd.MultiIndex.from_frame(frame[columns])
    group, names = pd.factorize(keys, sort=False)
    groups = frame.loc[~keys.duplicated(), columns].reset_index(drop=True)
    for section in pd.unique(frame.section_id):
        if len(np.unique(group[frame.section_id.eq(section)])) != 1:
            raise ValueError("Each physical section must belong to one profile group")
    if subject_column:
        parents = []
        for j in range(len(groups)):
            labels = frame.loc[group == j, subject_column].unique()
            if len(labels) != 1:
                raise ValueError("Each profile group must belong to exactly one subject; include subject in group_by")
            parents.append(labels[0])
        parent, subjects = pd.factorize(np.asarray(parents), sort=False)
    else:
        parent, subjects = np.arange(len(groups)), np.arange(len(groups)).astype(str)
    sizes = np.bincount(parent)
    a = (np.arange(len(subjects))[:, None] == parent[None, :]) / sizes[:, None]
    return group, parent, groups, np.asarray(subjects), a


class MultiSampleModel:
    def __init__(self, y, group, parent, a, k, eta_subject, eta_condition, graph, device):
        self.y = torch.as_tensor(y, dtype=torch.float64, device=device)
        self.mass = self.y.sum()
        self.s = self.y.sum(1, keepdim=True)
        self.const = (self.y*(self.y/self.s).clamp_min(1e-300).log()).sum()/self.mass
        self.graph, self.k = graph, k
        self.indices = [torch.as_tensor(np.flatnonzero(group == j), device=device) for j in range(len(parent))]
        self.parent = torch.as_tensor(parent, device=device)
        self.a = torch.as_tensor(a, dtype=torch.float64, device=device)
        d, r, g = len(a), len(parent), y.shape[1]
        center = np.eye(d) - np.ones((d, d))/d
        residual = np.eye(r) - a[parent]
        self.weights = torch.as_tensor(1/np.bincount(parent)[parent], device=device)
        q = g/(d*k) * (eta_subject*a.T@center@a + eta_condition*residual.T@np.diag(1/np.bincount(parent)[parent])@residual)
        q = (q + q.T)/2
        self.q = torch.as_tensor(q, dtype=torch.float64, device=device)
        self.curvature = max(0., float(np.linalg.eigvalsh(q).max()))*(1+1e-12)
        self.eta_subject, self.eta_condition = eta_subject, eta_condition
        self.scales = [len(y), torch.as_tensor(d*np.bincount(parent)[parent], device=device)[:, None, None]]

    def decompose(self, b):
        means = torch.einsum('dr,rkg->dkg', self.a, b)
        shared = means.mean(0)
        return shared, means, means-shared, b-means[self.parent]

    def value(self, state):
        p, b = state
        kl = self.const
        for ix, profile in zip(self.indices, b):
            rate = (p[ix]@profile).clamp_min(1e-300)
            kl = kl + ((self.s[ix]*rate).sum() - (self.y[ix]*rate.log()).sum() - self.s[ix].sum())/self.mass
        _, _, delta, within = self.decompose(b)
        scale = b.shape[-1]/(2*len(self.a)*self.k)
        first = self.eta_subject*scale*delta.square().sum()
        second = self.eta_condition*scale*(self.weights[:, None, None]*within.square()).sum()
        tv = self.graph.value(p)
        return kl+first+second+self.graph.strength*tv, dict(KL=kl, subject_penalty=first, condition_penalty=second, TV=tv)

    def gradients(self, state):
        p, b = state
        gp, gb = torch.empty_like(p), []
        for ix, profile in zip(self.indices, b):
            residual = (self.s[ix]-self.y[ix]/(p[ix]@profile).clamp_min(1e-300))/self.mass
            gp[ix] = residual@profile.T
            gb.append(p[ix].T@residual)
        gb = torch.stack(gb)+(self.q@b.reshape(len(b), -1)).reshape_as(b)
        return [gp+self.graph.gradient_curvature(p)[0], gb]

    def project(self, state):
        p, b = state
        return [simplex(p), simplex(b.reshape(-1, b.shape[-1])).reshape_as(b)]

    def step(self, state):
        p, b = state
        counts = torch.stack([profile*(p[ix].T@(self.y[ix]/(p[ix]@profile).clamp_min(1e-300)))/self.mass
                              for ix, profile in zip(self.indices, b)])
        if self.curvature > 0:
            b = profile_mstep(counts, b, self.q, self.curvature)
        else:
            b = em_simplex(counts.reshape(-1, b.shape[-1])).reshape_as(b)
        a = torch.empty_like(p)
        for ix, profile in zip(self.indices, b):
            a[ix] = p[ix]*((self.y[ix]/(p[ix]@profile).clamp_min(1e-300))@profile.T)*len(p)/self.mass
        return [self.graph.step(p, a), b]

    def p_curvature(self, state, enter, leave):
        p, b = state
        h = p.new_zeros(len(p))
        for ix, profile in zip(self.indices, b):
            db = profile[enter[ix]]-profile[leave[ix]]
            h[ix] = (self.y[ix]*(db/(p[ix]@profile).clamp_min(1e-300)).square()).sum(1)*len(p)/self.mass
        return h+spatial_pair_curvature(self.graph, p)


def fit_multisample(counts, metadata, n_components, *, group_by='sample_id',
                    subject_column=None, edges=None, coordinates=None,
                    eta_subject=10., eta_condition=20., spatial_weight=.05,
                    initial_proportions=None, initial_profiles=None, random_state=1,
                    device='auto', **optimizer):
    """Fit flat group deviations or subject + within-subject group deviations.

    For condition nesting, use group_by=['subject_id', 'condition'] and
    subject_column='subject_id'. Physical sections always retain separate graphs.
    """
    y = validate_counts(counts)
    if len(metadata) != len(y):
        raise ValueError("One metadata row is required per spot")
    if isinstance(counts, pd.DataFrame) and not counts.index.equals(metadata.index):
        raise ValueError("Counts and metadata spot IDs/order must match")
    group, parent, groups, subjects, a = design_matrix(metadata, group_by, subject_column)
    device = device_name(device)
    eta_subject = positive_number(eta_subject, 'eta_subject', True)
    eta_condition = positive_number(eta_condition, 'eta_condition', True) if subject_column else 0.
    if edges is None and spatial_weight:
        from .spatial import spatial_graph
        xy = metadata[['x', 'y']].to_numpy() if coordinates is None else coordinates
        i, j, _, _ = spatial_graph(xy, metadata.section_id.to_numpy())
        edges = np.column_stack((i, j))
    edges = validate_edges(edges, len(y), metadata.section_id.to_numpy())
    graph = SpatialLoss(edges, None, len(y), device, spatial_weight)
    p0, b0 = initial_factors(y, n_components, initial_proportions, initial_profiles, random_state, device)
    model = MultiSampleModel(y, group, parent, a, n_components, eta_subject, eta_condition, graph, device)
    tensors = [torch.as_tensor(p0, device=device), torch.as_tensor(np.repeat(b0[None], len(parent), axis=0), device=device)]
    state, history, diag = optimize(model, tensors, **optimizer)
    shared, means, delta, within = [v.cpu().numpy() for v in model.decompose(state[1])]
    diag.update(mode='multisample', group_by=[group_by] if isinstance(group_by, str) else list(group_by),
                subject_column=subject_column, eta_subject=eta_subject, eta_condition=eta_condition,
                spatial_weight=spatial_weight, n_groups=len(groups), n_subjects=len(subjects))
    extra = dict(shared_profiles=shared, subject_profiles=means, delta_subject=delta,
                 edges=edges,
                 delta_condition=within, group_index=group, group_parent=parent,
                 subjects=subjects.astype(str), group_labels=groups.astype(str).to_numpy(),
                 group_columns=np.asarray(groups.columns, str))
    return JointFit(state[0].cpu().numpy(), state[1].cpu().numpy(), history, diag, extra)
