"""Command-line entry points for reference, histology, multi-sample and multi-omics."""
import argparse
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
import torch

from ._io import read_frame, write_json


def aligned(frame, index, label):
    if set(frame.index) != set(index):
        raise ValueError(f'{label} spot IDs must match the measurements')
    return frame.loc[index]


def read_edges(path, spots):
    edges = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(edges.columns) != ['spot_i', 'spot_j']:
        raise ValueError('Edges CSV needs spot_i,spot_j columns containing spot IDs')
    pairs = pd.Index(spots).get_indexer(edges.to_numpy().ravel()).reshape(-1, 2)
    if (pairs < 0).any():
        raise ValueError('An edge contains an unknown spot ID')
    return pairs


def check_initial_ids(path, spots, genes):
    if path is None:
        return
    with np.load(path, allow_pickle=False) as z:
        for keys, expected in ((('spots', 'spot_ids'), spots), (('genes', 'gene_ids'), genes)):
            key = next((key for key in keys if key in z), None)
            if key is not None and not np.array_equal(z[key].astype(str), np.asarray(expected, str)):
                raise ValueError(f'Initialization {key} differ from input IDs/order')


def write_result(result, output, spots, genes, types=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = list(types) if types is not None else [f'C{k+1}' for k in range(result.proportions.shape[1])]
    pd.DataFrame(result.proportions, index=spots, columns=labels).to_csv(output/'P.csv')
    if result.profiles.ndim == 2:
        pd.DataFrame(result.profiles, index=labels, columns=genes).to_csv(output/'B.csv')
    else:
        for j, b in enumerate(result.profiles):
            pd.DataFrame(b, index=labels, columns=genes).to_csv(output/f'B_group_{j}.csv')
        extra = result.extra
        pd.DataFrame(extra['group_labels'], columns=extra['group_columns']).rename_axis('group_index').to_csv(output/'profile_groups.csv')
    arrays = dict(P=result.proportions, B=result.profiles, spots=np.asarray(spots, str),
                  genes=np.asarray(genes, str), components=np.asarray(labels, str), **result.extra)
    np.savez_compressed(output/'fit.npz', **arrays)
    pd.DataFrame(result.history).to_csv(output/'history.csv', index=False)
    write_json(output/'diagnostics.json', result.diagnostics)
    write_json(output/'COMPLETE.json', dict(status='passed' if result.diagnostics['converged'] else 'nonconverged'))
    if not result.diagnostics['converged']:
        raise RuntimeError('Fit did not converge; outputs and diagnostics have been retained')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('reference', 'histology', 'multisample', 'multiomics'):
        ap = sub.add_parser(mode)
        ap.add_argument('--output', type=Path, required=True, help='New output directory')
        ap.add_argument('--device', default='auto')
        if mode == 'multiomics':
            ap.add_argument('--manifest', type=Path, required=True, help='JSON describing paired assays, references/links and coordinates')
        else:
            ap.add_argument('--counts', type=Path, required=True)
        if mode == 'reference':
            ap.add_argument('--reference', type=Path, required=True, help='Types x features CSV on the same feature panel')
            ap.add_argument('--max-steps', type=int, default=3000)
            ap.add_argument('--tolerance', type=float, default=1e-6)
        else:
            ap.add_argument('--k', type=int, required=mode != 'multiomics')
            ap.add_argument('--seed', type=int, default=1)
            ap.add_argument('--initialization', type=Path, help='Optional NPZ containing consensus_P/consensus_B or P/B')
            ap.add_argument('--max-iter', type=int, default=20000)
            ap.add_argument('--spatial-weight', type=float, default=None if mode == 'histology' else .05)
        if mode == 'histology':
            ap.add_argument('--coordinates', type=Path, required=True)
            ap.add_argument('--image', type=Path, required=True)
            ap.add_argument('--image-spots', type=Path, required=True, help='Spot ID, x, y in raw image pixels')
            ap.add_argument('--microns-per-pixel', type=float, required=True)
            ap.add_argument('--radius-pixels', type=float, required=True)
            ap.add_argument('--encoder-source', type=Path, required=True)
            ap.add_argument('--image-weight', type=float)
            ap.add_argument('--offline', action='store_true')
        if mode == 'multisample':
            ap.add_argument('--metadata', type=Path, required=True, help='Spot ID, section_id, x/y and design columns')
            ap.add_argument('--group-by', nargs='+', default=['sample_id'])
            ap.add_argument('--subject-column')
            ap.add_argument('--eta-subject', type=float, default=10.)
            ap.add_argument('--eta-condition', type=float, default=20.)
            ap.add_argument('--edges', type=Path, help='Optional spot_i,spot_j CSV; otherwise construct coordinate neighbors')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    args.output.mkdir(parents=True)
    write_json(args.output/'run_arguments.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    torch.set_num_threads(1)
    initial, options = {}, {}
    if args.mode != 'reference':
        if args.initialization:
            with np.load(args.initialization, allow_pickle=False) as z:
                keys = ('consensus_P', 'consensus_B') if 'consensus_P' in z else ('P', 'B')
                initial = dict(initial_proportions=z[keys[0]], initial_profiles=z[keys[1]])
        options = dict(random_state=args.seed, spatial_weight=args.spatial_weight,
                       max_iter=args.max_iter, device=args.device, **initial)
    if args.mode == 'multiomics':
        from .multiomics import fit_multiomics
        plan = json.loads(args.manifest.read_text())
        base = args.manifest.resolve().parent
        data = {name: read_frame(base/entry['counts']) for name, entry in plan['assays'].items()}
        first = next(iter(data.values()))
        data = {name: aligned(frame, first.index, name) for name, frame in data.items()}
        refs = None
        if any('reference' in entry for entry in plan['assays'].values()):
            if not all('reference' in entry for entry in plan['assays'].values()):
                raise ValueError('Provide references for every assay or none')
            refs = {name: read_frame(base/entry['reference']) for name, entry in plan['assays'].items()}
        xy = aligned(read_frame(base/plan['coordinates']), first.index, 'Coordinates') if 'coordinates' in plan else None
        genes = [f'{name}::{gene}' for name, frame in data.items() for gene in frame.columns]
        check_initial_ids(args.initialization, first.index, genes)
        write_json(args.output/'manifest.json', plan)
        result = fit_multiomics(data, args.k, references=refs, links=plan.get('links'),
                               coordinates=None if xy is None else xy[['x', 'y']].to_numpy(),
                               sections=xy['block'].to_numpy() if xy is not None and 'block' in xy else None,
                               edges=read_edges(base/plan['edges'], first.index) if 'edges' in plan else None,
                               eta=plan.get('eta'), assay_weights=plan.get('assay_weights'), **options)
        types = next(iter(refs.values())).index if refs is not None else None
        write_result(result, args.output, first.index, genes, types)
    else:
        data = read_frame(args.counts)
        if args.mode != 'reference':
            check_initial_ids(args.initialization, data.index, data.columns)
        if args.mode == 'reference':
            from .reference import fit_reference
            reference = read_frame(args.reference)
            result = fit_reference(data, reference, device=args.device, max_steps=args.max_steps, tolerance=args.tolerance)
            types = reference.index
        elif args.mode == 'multisample':
            from .multisample import fit_multisample
            metadata = aligned(read_frame(args.metadata, text=True), data.index, 'Metadata')
            result = fit_multisample(data, metadata, args.k, group_by=args.group_by,
                                     subject_column=args.subject_column, eta_subject=args.eta_subject,
                                     eta_condition=args.eta_condition,
                                     edges=read_edges(args.edges, data.index) if args.edges else None, **options)
            types = None
        else:
            from .histology import fit_histology
            from .image_features import extract_histology
            xy = aligned(read_frame(args.coordinates), data.index, 'Coordinates')
            image_spots = aligned(read_frame(args.image_spots), data.index, 'Image coordinates')
            q = extract_histology(args.image, image_spots, microns_per_pixel=args.microns_per_pixel,
                                  radius_pixels=args.radius_pixels, encoder_source=args.encoder_source,
                                  output=args.output/'image_features', device='cuda', offline=args.offline)
            result = fit_histology(data, q, xy[['x', 'y']].to_numpy(), args.k,
                                   image_weight=args.image_weight,
                                   sections=xy['block'].to_numpy() if 'block' in xy else None, **options)
            types = None
        write_result(result, args.output, data.index, data.columns, types)
    print(f'Saved {args.output}')
