"""Section-specific coordinate graph and seed-1 spatial-weight calibration."""
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from .config import SolverConfig
from .model import _tv_energy_gradient, fit_spdecon

ALPHA_S = .05
EPSILON = .02
CALIBRATION_SEED = 1


def spatial_graph(xy, blocks):
    """The selected symmetric 4NN graph, built separately within each section."""
    xy, blocks = np.asarray(xy, dtype=float), np.asarray(blocks, dtype=str)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Coordinates must be a finite N x 2 matrix')
    if blocks.shape != (len(xy),) or np.any(blocks == ''):
        raise ValueError('Each spot must have one nonempty section/block ID')
    all_edges, sections = {}, []
    for block in pd.unique(blocks):
        positions = np.flatnonzero(blocks == block)
        if len(positions) < 2:
            raise ValueError('Section too small for a graph')
        d, ix = NearestNeighbors(n_neighbors=min(5, len(positions))).fit(xy[positions]).kneighbors(xy[positions])
        h = float(np.median(d[:, 1]))
        if h <= 0:
            raise ValueError('Nonpositive nearest-neighbour spacing')
        before = len(all_edges)
        for local, source in enumerate(positions):
            for distance, target in zip(d[local, 1:], positions[ix[local, 1:]]):
                if distance <= 1.5*h:
                    edge = tuple(sorted((int(source), int(target))))
                    weight = float(np.exp(-.5*(distance/h)**2))
                    all_edges[edge] = max(all_edges.get(edge, 0.), weight)
        sections.append(dict(block=str(block), n_spots=len(positions), spacing=h,
                             n_undirected_edges=len(all_edges)-before))
    pairs = sorted(all_edges)
    if not pairs:
        raise ValueError('No spatial edges')
    i, j = np.array(pairs, dtype=np.int64).T
    weights = np.array([all_edges[e] for e in pairs])
    assert np.all(blocks[i] == blocks[j])
    return i, j, weights, dict(sections=sections, n_edges=len(i),
        graph='symmetric 4NN, distance <=1.5 times section median nearest spacing',
        weights='exp(-distance^2/(2*spacing^2))', normalizer=float(weights.sum()),
        cross_section_edges=0)


def calibrate(counts, p0, b0, i, j, weights):
    """Calibrate once on seed-1 consensus; no truth or fitted metric is used."""
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device='cuda')
    y, p, b = tensor(counts), tensor(.9*p0+.1/p0.shape[1]), tensor(b0)
    rna = ((y.sum(1, keepdim=True)-y/(p@b).clamp_min(1e-8))/y.sum())@b.T
    left, right = [torch.as_tensor(v, device='cuda', dtype=torch.long) for v in (i, j)]
    _, spatial = _tv_energy_gradient(p,
        (left, right, tensor(weights)/float(weights.sum()), 1., EPSILON))
    tangent = lambda g: float(torch.linalg.norm(g-g.mean(1, keepdim=True)))
    rn, sn = tangent(rna), tangent(spatial)
    strength = ALPHA_S*rn/max(sn, 1e-12)
    return dict(spatial_weight=strength, alpha_s=ALPHA_S, epsilon=EPSILON,
                calibration_seed=CALIBRATION_SEED, RNA_gradient=rn,
                spatial_unit_gradient=sn, reference='0.9*P0+0.1/K; B0',
                reused_across_seeds=True, truth_used=False)


def refine(counts, p0, b0, i, j, weights, strength, seed):
    """Jointly refine P/B from the unrefined consensus under KL + TV."""
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('Spatial weight must be finite and nonnegative')
    return fit_spdecon(counts, p0.shape[1],
        p0.copy(), b0.copy(), SolverConfig(random_state=seed), tv_edges=np.column_stack([i, j]),
        tv_weights=weights, tv_lambda=strength, tv_epsilon=EPSILON)
