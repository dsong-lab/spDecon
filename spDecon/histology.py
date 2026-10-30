"""Joint molecular and HIPT/UNI morphology reconstruction with gated spatial TV."""
import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

from ._numerics import FLOOR, simplex, simplex_mass, em_simplex, em_mass
from ._joint import (JointFit, SpatialLoss, MolecularLoss, initial_factors, validate_counts,
                     device_name, positive_number, optimize, spatial_pair_curvature)


def morphology_project(h):
    return torch.cat([simplex_mass(h[:, :16], .5), simplex_mass(h[:, 16:], .5)], dim=1)


def morphology_em(counts):
    return torch.cat([em_mass(counts[:, :16], .5), em_mass(counts[:, 16:], .5)], dim=1)


@torch.no_grad()
def initialize_morphology(p, q, exposure):
    weights = np.asarray(exposure, float)/np.mean(exposure)
    scale = np.sqrt(weights[:, None])
    pa = p.cpu().numpy()
    h = np.column_stack([nnls(pa*scale, q[:, j]*scale[:, 0])[0] for j in range(32)])+1e-4
    h = torch.as_tensor(h, device=p.device)
    h = morphology_project(torch.cat([.5*h[:, :16]/h[:, :16].sum(1, keepdim=True),
                                      .5*h[:, 16:]/h[:, 16:].sum(1, keepdim=True)], dim=1))
    qt, v = torch.as_tensor(q, device=p.device), torch.as_tensor(weights[:, None], device=p.device)
    for iteration in range(3000):
        h = morphology_em(h*(p.T@(v*qt/(p@h).clamp_min(FLOOR))))
        if iteration % 20 == 0:
            gradient = p.T@(v*(1-qt/(p@h).clamp_min(FLOOR)))/len(p)
            if float((h-morphology_project(h-gradient)).abs().max()) <= 1e-7:
                break
    return h


class HistologyModel:
    def __init__(self, y, q, graph, image_weight, device):
        self.loss, self.graph = MolecularLoss(y, device), graph
        self.q = torch.as_tensor(q, device=device)
        self.v = self.loss.s/self.loss.s.mean()
        self.q_em = self.v*self.q
        self.qconst = (self.q_em*self.q.clamp_min(1e-300).log()).sum()/len(y)
        self.image_weight = image_weight
        self.scales = [len(y), 1., 1.]

    def value(self, state):
        p, b, h = state
        kl = self.loss.value(p, b)
        rate = (p@h).clamp_min(FLOOR)
        image = self.qconst-(self.q_em*rate.log()).sum()/len(p)+(self.v*(rate-self.q)).sum()/len(p)
        tv = self.graph.value(p)
        return kl+self.image_weight*image+self.graph.strength*tv, dict(KL=kl, image_KL=image, TV=tv)

    def gradients(self, state):
        p, b, h = state
        gp, gb = self.loss.gradients(p, b)
        residual = self.image_weight*self.v*(1-self.q/(p@h).clamp_min(FLOOR))/len(p)
        return [gp+residual@h.T+self.graph.gradient_curvature(p)[0], gb, p.T@residual]

    def project(self, state):
        return [simplex(state[0]), simplex(state[1]), morphology_project(state[2])]

    def step(self, state):
        p, b, h = state
        b = em_simplex(b*(p.T@(self.loss.y/(p@b).clamp_min(FLOOR))))
        h = morphology_em(h*(p.T@(self.q_em/(p@h).clamp_min(FLOOR))))
        counts = p*((self.loss.y/(p@b).clamp_min(FLOOR))@b.T)*len(p)/self.loss.mass
        counts += self.image_weight*p*((self.q_em/(p@h).clamp_min(FLOOR))@h.T)
        return [self.graph.step(p, counts), b, h]

    def p_curvature(self, state, enter, leave):
        p, b, h = state
        db, dh = b[enter]-b[leave], h[enter]-h[leave]
        curvature = (self.loss.y*(db/(p@b).clamp_min(FLOOR)).square()).sum(1)*len(p)/self.loss.mass
        curvature += self.image_weight*(self.q_em*(dh/(p@h).clamp_min(FLOOR)).square()).sum(1)
        return curvature+spatial_pair_curvature(self.graph, p)


def fit_histology(counts, morphology, coordinates, n_components, *, sections=None,
                  image_weight=None, spatial_weight=None,
                  image_gate=True,
                  initial_proportions=None, initial_profiles=None,
                  random_state=1, device='auto', verbose=True, **optimizer):
    """Fit from the unrefined RNA consensus and fixed 32-column HIPT/UNI Q.

    The image command produces Q from the registered H&E image. For new data,
    omitted coefficients use weighted gradient calibration (alpha .30/.05).
    Set image_gate=False for the image-reconstruction, coordinate-TV ablation.
    Explicit coefficients reproduce dataset-specific historical calibrations.
    """
    y = validate_counts(counts)
    if isinstance(counts, pd.DataFrame) and isinstance(morphology, pd.DataFrame):
        if not counts.index.equals(morphology.index):
            raise ValueError('Counts and morphology spot IDs/order differ')
    # The paired RNA-only control keeps exposure-weighted coordinate TV.
    q = np.full((len(y), 32), 1/32) if morphology is None else np.asarray(morphology, float)
    if morphology is None:
        image_weight = 0.
    if q.shape != (len(y), 32) or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError('Morphology must be a finite nonnegative spots-by-32 HIPT/UNI matrix')
    if not np.allclose(q.reshape(len(y), 2, 16).sum(2), .5, rtol=0, atol=1e-8):
        raise ValueError('Each 16-column morphology block must sum to 0.5')
    device = device_name(device)
    if verbose:
        arm = 'RNA only' if morphology is None else 'RNA + H&E'
        print(f"[spDecon:histology] preparing {arm} fit | seed={random_state}", flush=True)
    p0, b0 = initial_factors(y, n_components, initial_proportions, initial_profiles, random_state, device, verbose)
    from .spatial import spatial_graph
    sections = np.repeat('0', len(y)) if sections is None else sections
    i, j, coordinate_weights, info = spatial_graph(coordinates, sections)
    edges = np.column_stack((i, j))
    differences = ((np.sqrt(q[i])-np.sqrt(q[j]))**2).sum(1)
    positive = differences[differences > 0]
    gate = np.exp(-differences/np.median(positive)) if len(positive) else np.ones(len(i))
    if not image_gate:
        gate = np.ones(len(i))
    v = y.sum(1)/y.sum(1).mean()
    exposure_weights = 2*v[i]*v[j]/(v[i]+v[j])
    normalizer = coordinate_weights.sum()
    calibration = {}
    if image_weight is None or spatial_weight is None:
        pc, bc = [torch.as_tensor(value, device=device) for value in (.9*p0+.1/n_components, b0)]
        hc = initialize_morphology(pc, q, y.sum(1))
        coordinate_graph = SpatialLoss(edges, coordinate_weights*exposure_weights, len(y), device, 1., normalizer=normalizer)
        unit = HistologyModel(y, q, coordinate_graph, 1., device)
        rna_gradient = unit.loss.gradients(pc, bc)[0]
        image_gradient = (unit.v*(1-unit.q/(pc@hc).clamp_min(FLOOR))@hc.T)/len(y)
        spatial_gradient = coordinate_graph.gradient_curvature(pc)[0]
        norm = lambda g: float(torch.linalg.norm(g-g.mean(1, keepdim=True)))
        rn, im, sp = map(norm, (rna_gradient, image_gradient, spatial_gradient))
        image_weight = .30*rn/max(im, 1e-12) if image_weight is None else image_weight
        spatial_weight = .05*rn/max(sp, 1e-12) if spatial_weight is None else spatial_weight
        calibration = dict(RNA_gradient=rn, image_gradient=im, coordinate_TV_gradient=sp,
                           alpha_image=.30, alpha_spatial=.05, reference='0.9 P0 + 0.1/K')
    image_weight = positive_number(image_weight, 'image_weight', True)
    spatial_weight = positive_number(spatial_weight, 'spatial_weight', True)
    graph = SpatialLoss(edges, coordinate_weights*gate*exposure_weights, len(y), device, spatial_weight, normalizer=normalizer)
    model = HistologyModel(y, q, graph, image_weight, device)
    p, b = [torch.as_tensor(a, device=device) for a in (p0, b0)]
    h = initialize_morphology(p, q, y.sum(1))
    optimizer.setdefault('check_every', 10)
    optimizer.setdefault('minimum_iterations', 20)
    optimizer.setdefault('relative_tolerance', 1e-6)
    state, history, diag = optimize(model, [p, b, h], verbose=verbose, **optimizer)
    diag.update(mode='histology', image_weight=image_weight, spatial_weight=spatial_weight,
                image_gate=bool(image_gate),
                calibration=calibration, graph=info,
                start='supplied' if initial_proportions is not None else 'unrefined_consensus',
                image_exposure_weighting=True, tv_exposure_weighting=True)
    return JointFit(state[0].cpu().numpy(), state[1].cpu().numpy(), history, diag,
                    dict(morphology_profiles=state[2].cpu().numpy(), morphology=q,
                         edges=edges, coordinate_weights=coordinate_weights, morphology_gate=gate))
