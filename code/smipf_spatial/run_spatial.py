"""Section 1 reference-free pipeline: frozen consensus -> joint KL + TV fit."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'SMIPF_TORCH_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
CORE = ROOT.parent / 'smipf_core'
sys.path.insert(0, str(CORE / 'src'))
from consensus_raw_poisson import fit_consensus_raw_poisson
from rare_init_poisson.model import _raw_config
import louvain_init_poisson.model as louvain

# Resolve the archived utility at its current location without editing the core.
louvain.SPICEMIX_UTIL = CORE / 'benchmarking_reference_free_official/external/SpiceMix/SpiceMix/util.py'
spec = importlib.util.spec_from_file_location('poisson_hmrf.section1_tv', ROOT / 'native_model.py')
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)

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
    _, spatial = native._tv_energy_gradient(p,
        (left, right, tensor(weights)/float(weights.sum()), 1., EPSILON))
    tangent = lambda g: float(torch.linalg.norm(g-g.mean(1, keepdim=True)))
    rn, sn = tangent(rna), tangent(spatial)
    strength = ALPHA_S*rn/max(sn, 1e-12)
    return dict(spatial_weight=strength, alpha_s=ALPHA_S, epsilon=EPSILON,
                calibration_seed=CALIBRATION_SEED, RNA_gradient=rn,
                spatial_unit_gradient=sn, reference='0.9*P0+0.1/K; B0',
                reused_across_seeds=True, truth_used=False)


def refine(counts, p0, b0, i, j, weights, strength, seed):
    """Jointly refine C/P from unrefined consensus using the original solver."""
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('Spatial weight must be finite and nonnegative')
    return native.fit_poisson_hmrf(counts, np.zeros((len(counts), 2)), p0.shape[1],
        p0.copy(), b0.copy(), _raw_config(seed), tv_edges=np.column_stack([i, j]),
        tv_weights=weights, tv_lambda=strength, tv_epsilon=EPSILON)


def write_json(path, value):
    def serial(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, indent=2, default=serial)+'\n')


def read_frame(path):
    frame = pd.read_csv(path, index_col=0, keep_default_na=False)
    frame.index, frame.columns = frame.index.astype(str), frame.columns.astype(str)
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f'Duplicate identifiers: {path}')
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--counts', type=Path, required=True, help='Raw counts CSV: spots x genes, first column spot ID')
    ap.add_argument('--coordinates', type=Path, required=True, help='CSV with spot ID, x, y; optionally block')
    ap.add_argument('--blocks', type=Path, help='Optional CSV with spot ID and block; required for multiple sections unless coordinates contain block')
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--counts-order', choices=['F', 'C'], default='F',
                    help='Numeric memory layout: F for the seven CSV datasets; C for archived STARmap sparse input')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 3, 5])
    ap.add_argument('--spatial-weight', type=float, help='Explicit lambda override; 0 is the RNA-only control. Default: seed-1 gradient calibration')
    ap.add_argument('--output', type=Path, required=True, help='New output directory')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Output already exists; choose a new directory')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        ap.error('Seeds must be distinct nonnegative integers')
    if args.spatial_weight is not None and (not np.isfinite(args.spatial_weight) or args.spatial_weight < 0):
        ap.error('Spatial weight must be finite and nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('The selected float32 native solver requires CUDA')
    torch.set_num_threads(1)
    x, xy = read_frame(args.counts), read_frame(args.coordinates)
    y = np.array(x.to_numpy(), dtype=float, order=args.counts_order)
    if not np.isfinite(y).all() or (y < 0).any() or (y.sum(1) <= 0).any():
        raise ValueError('Counts must be finite and nonnegative, without empty spots')
    if not 1 <= args.k <= min(y.shape):
        raise ValueError('K is incompatible with counts')
    if set(x.index) != set(xy.index):
        raise ValueError('Coordinate and counts spot IDs must match')
    xy = xy.loc[x.index]
    block_frame = read_frame(args.blocks) if args.blocks else xy
    if args.blocks and set(block_frame.index) != set(x.index):
        raise ValueError('Block and counts spot IDs must match')
    blocks = (block_frame.loc[x.index, 'block'].to_numpy(dtype=str)
              if args.blocks or 'block' in block_frame else np.repeat('0', len(x)))
    i, j, w, graph_info = spatial_graph(xy[['x', 'y']].to_numpy(), blocks)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'graph.npz', edge_i=i, edge_j=j, weight=w,
                        spot_ids=x.index.to_numpy(str), blocks=blocks)
    write_json(args.output/'graph_info.json', graph_info)
    source_files = [Path(__file__), ROOT/'native_model.py']+list((CORE/'src').rglob('*.py'))
    input_files = [args.counts, args.coordinates]+([args.blocks] if args.blocks else [])
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.output/'config.json', dict(pipeline='section1_consensus_native_TV',
        K=args.k, seeds=args.seeds, counts_order=args.counts_order, alpha_s=ALPHA_S, epsilon=EPSILON,
        calibration_seed=CALIBRATION_SEED, lambda_override=args.spatial_weight,
        graph=graph_info, final_optimizer=_raw_config(1).to_dict(),
        start_stage='unrefined_consensus', truth_used=False,
        torch_version=torch.__version__,
        source_sha256={str(p.resolve()): sha(p) for p in source_files},
        input_sha256={str(p.resolve()): sha(p) for p in input_files}))
    started = time.perf_counter()
    # Reuse the frozen API, including its RNA-only fit. TV starts from P0/B0;
    # the extra RNA-only fit is retained as a control, never used as a warm start.
    calibration_fit = None
    if args.spatial_weight is None:
        print('Calibrating lambda from seed 1 consensus', flush=True)
        calibration_fit = fit_consensus_raw_poisson(y, args.k, random_state=CALIBRATION_SEED)
        calibration = calibrate(y, calibration_fit.consensus_proportions,
            calibration_fit.consensus_profiles, i, j, w)
        np.savez_compressed(args.output/'calibration_consensus.npz',
            P=calibration_fit.consensus_proportions, B=calibration_fit.consensus_profiles)
    else:
        calibration = dict(spatial_weight=args.spatial_weight, selection='explicit override',
                           epsilon=EPSILON, truth_used=False, reused_across_seeds=True)
    strength = calibration['spatial_weight']
    write_json(args.output/'calibration.json', calibration)
    records = []
    for seed in args.seeds:
        print(f'Seed {seed}, lambda={strength:.17g}', flush=True)
        base = (calibration_fit if seed == CALIBRATION_SEED and calibration_fit is not None
                else fit_consensus_raw_poisson(y, args.k, random_state=seed))
        fit = refine(y, base.consensus_proportions, base.consensus_profiles, i, j, w, strength, seed)
        dest = args.output/f'seed_{seed}'
        dest.mkdir()
        np.savez_compressed(dest/'fit.npz', P=fit.proportions, B=fit.profiles,
            consensus_P=base.consensus_proportions, consensus_B=base.consensus_profiles,
            RNA_only_P=base.proportions, RNA_only_B=base.profiles,
            spots=x.index.to_numpy(str), genes=x.columns.to_numpy(str), exposure=y.sum(1))
        branches = {f'P_{k}': v for k, v in base.branch_proportions.items()}
        branches.update({f'B_{k}': v for k, v in base.branch_profiles.items()})
        branches.update({f'order_{k}': v for k, v in base.alignments_to_all_gene.items()})
        np.savez_compressed(dest/'branches.npz', **branches)
        names = [f'C{k+1}' for k in range(args.k)]
        pd.DataFrame(fit.proportions, index=x.index, columns=names).to_csv(dest/'P.csv')
        pd.DataFrame(fit.profiles, index=names, columns=x.columns).to_csv(dest/'B.csv')
        pd.DataFrame(fit.history).to_csv(dest/'history.csv', index=False)
        info = dict(fit.diagnostics, seed=seed, start_stage='unrefined_consensus',
            branch_diagnostics=base.diagnostics['branch_diagnostics'],
            RNA_only_diagnostics=base.diagnostics, calibration=calibration)
        info.update(method='consensus_poisson_TV' if strength else 'consensus_poisson',
            backend_method=fit.diagnostics['method'],
            model='Y_ig ~ Poisson(exposure_i * (P B)_ig); normalized smoothed graph TV on P',
            warm_start='unrefined equal-weight three-branch consensus',
            coordinates_used=bool(strength), spatial_graph_used=bool(strength),
            graph=graph_info, n_edges=len(i) if strength else 0,
            gradient_calibration=calibration, final_spatial_lambda=strength)
        # This native field describes the inactive HMRF affinity penalty.
        info.pop('final_effective_gradient_ratio', None)
        write_json(dest/'diagnostics.json', info)
        records.append(dict(seed=seed, converged=bool(fit.diagnostics['converged']),
            branches_converged=all(d['converged'] for d in base.diagnostics['branch_diagnostics'].values())))
    passed = all(r['converged'] and r['branches_converged'] for r in records)
    write_json(args.output/'COMPLETE.json', dict(status='passed' if passed else 'nonconverged',
        fits=records, spatial_weight=strength, total_runtime_sec=time.perf_counter()-started))
    if not passed:
        raise RuntimeError('A fit did not converge; see saved diagnostics')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
