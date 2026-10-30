"""spDecon Section 1 reference-free spot-level deconvolution with KL + TV."""
import argparse
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch

from .config import SolverConfig
from ._io import read_frame, write_json
from .consensus import build_consensus
from .spatial import ALPHA_S, EPSILON, CALIBRATION_SEED, spatial_graph, calibrate, refine

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(prog='run_spdecon.py', description=__doc__,
        epilog='Other models: reference, histology, multisample, multiomics. Use run_spdecon.py MODEL --help.')
    ap.add_argument('--counts', type=Path, required=True, help='Raw counts CSV: spots x genes, first column spot ID')
    ap.add_argument('--coordinates', type=Path, required=True, help='CSV with spot ID, x, y; optionally block')
    ap.add_argument('--blocks', type=Path, help='Optional CSV with spot ID and block; required for multiple sections unless coordinates contain block')
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--counts-order', choices=['F', 'C'], default='F',
                    help='Dense count memory layout (default: F)')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 3, 5])
    ap.add_argument('--deterministic', action='store_true', help='Use deterministic CUDA kernels and fixed RNG states; may differ from historical float32 fits')
    ap.add_argument('--spatial-weight', type=float, help='Explicit lambda override; 0 is the RNA-only control. Default: seed-1 gradient calibration')
    ap.add_argument('--output', type=Path, required=True, help='New output directory')
    args = ap.parse_args()
    from .reproducibility import set_seed
    set_seed(args.seeds[0], deterministic=args.deterministic)
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
    write_json(args.output/'config.json', dict(pipeline='spDecon_consensus_spatial_TV',
        K=args.k, seeds=args.seeds, counts_order=args.counts_order, alpha_s=ALPHA_S, epsilon=EPSILON,
        calibration_seed=CALIBRATION_SEED, lambda_override=args.spatial_weight,
        graph=graph_info, final_optimizer=SolverConfig(random_state=1).to_dict(),
        start_stage='unrefined_consensus', truth_used=False,
        torch_version=torch.__version__, deterministic_algorithms=args.deterministic))
    started = time.perf_counter()
    calibration_fit = None
    if args.spatial_weight is None:
        print('Calibrating lambda from seed 1 consensus', flush=True)
        calibration_fit = build_consensus(y, args.k, random_state=CALIBRATION_SEED)
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
                else build_consensus(y, args.k, random_state=seed))
        fit = refine(y, base.consensus_proportions, base.consensus_profiles, i, j, w, strength, seed)
        dest = args.output/f'seed_{seed}'
        dest.mkdir()
        np.savez_compressed(dest/'fit.npz', P=fit.proportions, B=fit.profiles,
            consensus_P=base.consensus_proportions, consensus_B=base.consensus_profiles,
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
            consensus_diagnostics=base.diagnostics, calibration=calibration)
        info.update(method='spDecon_KL_TV' if strength else 'spDecon_KL',
            backend_method=fit.diagnostics['method'],
            model='generalized_KL(Y || diag(exposure) P B)/sum(Y) + lambda TV(P)',
            warm_start='unrefined equal-weight three-branch consensus',
            coordinates_used=bool(strength), spatial_graph_used=bool(strength),
            graph=graph_info, n_edges=len(i) if strength else 0,
            gradient_calibration=calibration, final_spatial_lambda=strength)
        write_json(dest/'diagnostics.json', info)
        records.append(dict(seed=seed, converged=bool(fit.diagnostics['converged']),
            branches_converged=all(d['converged'] for d in base.diagnostics['branch_diagnostics'].values())))
    passed = all(r['converged'] and r['branches_converged'] for r in records)
    write_json(args.output/'COMPLETE.json', dict(status='passed' if passed else 'nonconverged',
        fits=records, spatial_weight=strength, total_runtime_sec=time.perf_counter()-started))
    if not passed:
        raise RuntimeError('A fit did not converge; see saved diagnostics')
    print(f'Saved {args.output}', flush=True)
