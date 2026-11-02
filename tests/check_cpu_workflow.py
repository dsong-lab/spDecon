"""Run the full pseudo-spot and non-spatial bulk workflows without CUDA.

Usage: python tests/check_cpu_workflow.py
Synthetic single-cell counts are pooled into 72 mixtures of three cell types.
The check covers all initializers, consensus, calibration and refinement.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    rng = np.random.default_rng(8)
    signatures = np.ones((3, 60))
    for k in range(3):
        signatures[k, 20*k:20*(k+1)] = 20
    signatures /= signatures.sum(1, keepdims=True)
    single_cells = [rng.multinomial(500, b, size=200) for b in signatures]
    mixtures = []
    for spot in range(72):
        probabilities = np.full(3, .1)
        probabilities[spot // 24] = .8
        types = rng.choice(3, size=20, p=probabilities)
        mixtures.append(sum(single_cells[k][rng.integers(200)] for k in types))
    counts = pd.DataFrame(mixtures, index=[f'sample_{i}' for i in range(72)],
                          columns=[f'gene_{i}' for i in range(60)])
    coordinates = pd.DataFrame({'x': np.tile(np.arange(8), 9),
                                'y': np.repeat(np.arange(9), 8)}, index=counts.index)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    with TemporaryDirectory(prefix='spdecon-cpu-') as directory:
        root = Path(directory)
        counts.to_csv(root/'counts.csv')
        coordinates.to_csv(root/'coordinates.csv')
        base = [sys.executable, str(ROOT/'run_spdecon.py'), 'core', '--counts',
                str(root/'counts.csv'), '--k', '3', '--seeds', '1']
        for name, options in (
            ('pseudo_spots', ['--coordinates', str(root/'coordinates.csv'), '--device', 'cpu']),
            ('bulk', ['--spatial-weight', '0', '--device', 'auto']),
        ):
            output = root/name
            run = subprocess.run(base + options + ['--output', str(output)], cwd=ROOT,
                                 env=env, text=True, capture_output=True, timeout=600)
            assert run.returncode == 0, run.stdout[-12000:] + run.stderr[-4000:]
            complete = json.loads((output/'COMPLETE.json').read_text())
            diag = json.loads((output/'seed_1/diagnostics.json').read_text())
            assert complete['status'] == 'passed', complete
            assert diag['device'] == 'cpu' and diag['converged'], diag
            assert len(diag['branch_diagnostics']) == 3
            assert all(d['device'] == 'cpu' and d['converged']
                       for d in diag['branch_diagnostics'].values())
            p = pd.read_csv(output/'seed_1/P.csv', index_col=0)
            b = pd.read_csv(output/'seed_1/B.csv', index_col=0)
            assert p.shape == (72, 3) and b.shape == (3, 60)
            assert p.index.equals(counts.index) and b.columns.equals(counts.columns)
            for a in (p.to_numpy(), b.to_numpy()):
                assert np.isfinite(a).all() and (a >= 0).all()
                np.testing.assert_allclose(a.sum(1), 1, atol=2e-6, rtol=0)
            if name == 'pseudo_spots':
                assert complete['spatial_weight'] > 0 and diag['n_edges'] > 0
            else:
                assert complete['spatial_weight'] == 0 and diag['graph']['n_edges'] == 0
            print(f"PASS: {name}, 3/3 branches + refinement converged on CPU "
                  f"({complete['total_runtime_sec']:.1f}s).", flush=True)
        reference = pd.DataFrame([cells.mean(0) for cells in single_cells],
                                 index=['type_A', 'type_B', 'type_C'], columns=counts.columns)
        reference.to_csv(root/'reference.csv')
        output = root/'bulk_reference'
        run = subprocess.run(
            [sys.executable, str(ROOT/'run_spdecon.py'), 'reference',
             '--counts', str(root/'counts.csv'), '--reference', str(root/'reference.csv'),
             '--device', 'cpu', '--output', str(output)],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
        assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
        diag = json.loads((output/'diagnostics.json').read_text())
        assert diag['device'] == 'cpu' and diag['converged'], diag
        p = pd.read_csv(output/'P.csv', index_col=0)
        assert p.index.equals(counts.index) and p.columns.equals(reference.index)
        assert np.isfinite(p).all().all() and (p >= 0).all().all()
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-12, rtol=0)
        print('PASS: fixed-reference bulk deconvolution converged on CPU.', flush=True)
        invalid = subprocess.run(base + ['--output', str(root/'invalid')],
                                 cwd=ROOT, env=env, text=True, capture_output=True)
        assert invalid.returncode != 0 and 'Without --coordinates' in invalid.stderr
        assert not (root/'invalid').exists()


if __name__ == '__main__':
    main()
