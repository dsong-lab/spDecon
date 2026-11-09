"""Run all six notebook workflows on temporary user-style AnnData inputs.

Usage: python tests/check_user_data_tutorials.py
Joint fits use 30 iterations for this interface check, not a convergence claim.
Image extraction requires encoder weights; this check supplies precomputed Q.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

os.environ['MPLBACKEND'] = 'Agg'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tutorials import common


def make_inputs(path, *, exact=False):
    rng = np.random.default_rng(8)
    b = np.ones((3, 60))
    for k in range(3):
        b[k, 20*k:20*(k+1)] = 20
    b /= b.sum(1, keepdims=True)
    p = np.full((72, 3), .1)
    p[np.arange(72), np.arange(72)//24] = .8
    # The paired joint example uses exact low-rank counts to check the interface
    # without making convergence of noisy concatenated fits part of this test.
    y = np.rint(44000 * (p @ b)).astype(np.int64) if exact else rng.poisson(10000*(p @ b))
    obs = pd.DataFrame({'section_id': np.repeat(['s1', 's2', 's3'], 24),
                        'sample_id': np.repeat(['A', 'B', 'C'], 24),
                        'region': np.repeat(['type_A', 'type_B', 'type_C'], 24)},
                       index=[f'spot_{i}' for i in range(72)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(60)])
    data = ad.AnnData(y, obs=obs, var=var)
    data.obsm['spatial'] = np.c_[np.tile(np.arange(8), 9), np.repeat(np.arange(9), 8)]
    data.obsm['morphology'] = np.c_[.5*rng.dirichlet(np.ones(16), 72),
                                   .5*rng.dirichlet(np.ones(16), 72)]
    for name in ('counts', 'rna', 'samples'):
        data.write_h5ad(path / f'{name}.h5ad')
    selected = np.r_[0:6, 20:26, 40:46]
    protein_profile = b[:, selected] / b[:, selected].sum(1, keepdims=True)
    protein_counts = (np.rint(6600*(p @ protein_profile)).astype(np.int64) if exact
                      else rng.poisson(5000*(p @ protein_profile)))
    protein = ad.AnnData(protein_counts,
                         obs=obs.copy(), var=var.iloc[selected].copy())
    protein.obsm['spatial'] = data.obsm['spatial'].copy()
    protein = protein[::-1].copy()  # Exercise spot-ID alignment.
    protein.var_names = [f'protein_{i}' for i in range(len(selected))]
    protein.write_h5ad(path / 'protein.h5ad')
    ref = ad.AnnData(500*b, obs=pd.DataFrame(index=['type_A', 'type_B', 'type_C']), var=var)
    ref.write_h5ad(path / 'reference.h5ad')
    cell_labels = np.repeat(ref.obs_names, 12)
    cell_counts = rng.poisson(np.repeat(500*b, 12, axis=0) * np.tile(np.arange(1, 13), 3)[:, None])
    scrna = ad.AnnData(np.log1p(cell_counts),
        obs=pd.DataFrame({'cell_type': cell_labels}, index=[f'cell_{i}' for i in range(36)]),
        var=var.copy())
    # X is transformed deliberately: reference preparation must use the raw layer.
    scrna.layers['counts'] = cell_counts
    scrna[:, ::-1].copy().write_h5ad(path / 'scrna.h5ad')
    pd.DataFrame(p, index=obs.index, columns=ref.obs_names).to_csv(path / 'fractions.csv')
    pd.DataFrame({'RNA': data.var_names[selected], 'protein': protein.var_names}).to_csv(
        path / 'feature_links.csv', index=False)


def main():
    torch.set_num_threads(1)
    os.chdir(ROOT)
    with TemporaryDirectory(prefix='spdecon-user-input-') as directory:
        path = Path(directory)
        make_inputs(path)
        paired = path / 'paired'
        paired.mkdir()
        make_inputs(paired, exact=True)
        common.ROOT = path
        for notebook in sorted((ROOT / 'tutorials').glob('*.ipynb')):
            cells = json.loads(notebook.read_text())['cells']
            codes = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
            ns = {}
            log = io.StringIO()
            print(f'Running {notebook.name}...', flush=True)
            try:
                with contextlib.redirect_stdout(log):
                    exec(compile(codes[0], notebook.name, 'exec'), ns)
                    inputs = paired if notebook.name.startswith('06_') else path
                    ns.update(ROOT=path, DATA_DIR=inputs, K=3, DEVICE='cpu', MAX_ITER=30,
                              VERBOSE=False, TRUTH_CSV=path/'fractions.csv', REGION_COLUMN='region')
                    for code in codes[1:]:
                        exec(compile(code, notebook.name, 'exec'), ns)
                    # Optional scoring must also work when no truth is supplied.
                    if notebook.name.startswith(('01_', '02_', '03_')):
                        ns.update(TRUTH_CSV=None, REGION_COLUMN=None)
                        exec(compile(codes[-1], notebook.name, 'exec'), ns)
                if notebook.name.startswith('04_'):
                    weights = list(ns['fits'].values())
                elif notebook.name.startswith('03_'):
                    weights = [ns['he'].proportions, ns['rna'].proportions]
                else:
                    weights = [ns['p']]
                for p in weights:
                    assert p.shape == (72, 3) and np.isfinite(p).all() and (p >= 0).all()
                    np.testing.assert_allclose(p.sum(1), 1, atol=2e-6, rtol=0)
                if notebook.name.startswith(('04_', '06_')):
                    assert ns['protein_adata'].obs_names.equals(ns['rna_adata'].obs_names)
                if notebook.name.startswith('02_'):
                    saved = ad.read_h5ad(path / 'reference.h5ad')
                    raw = ns['raw_counts']
                    totals = raw.sum(axis=1)
                    expected = (raw.div(totals, axis=0).groupby(ns['cell_types'], sort=False).mean()
                                .mul(totals.groupby(ns['cell_types'], sort=False).mean(), axis=0))
                    np.testing.assert_allclose(saved.to_df(), expected.loc[saved.obs_names, saved.var_names])
                    assert ns['K'] == saved.n_obs == 3
                    assert saved.var_names.equals(ns['adata'].var_names)
                    markers = pd.read_csv(path / 'reference_markers.csv')
                    assert markers.count().between(1, 50).all()
                    for label in markers:
                        chosen = markers[label].dropna()
                        detected = (raw.loc[ns['cell_types'].eq(label), chosen] > 0).sum()
                        assert (detected >= max(3, .1 * ns['cell_types'].eq(label).sum())).all()
                    # Exercise arithmetic means through the same notebook preparation cells.
                    preparation = [c for c in cells if c.get('id') in (
                        'reference-build-profiles', 'reference-select-markers', 'reference-save')]
                    ns['BALANCE_REFERENCE'] = False
                    with contextlib.redirect_stdout(log):
                        for cell in preparation:
                            exec(compile(''.join(cell['source']), notebook.name, 'exec'), ns)
                    raw_mean = raw.groupby(ns['cell_types'], sort=False).mean()
                    saved = ad.read_h5ad(path / 'reference.h5ad')
                    np.testing.assert_allclose(saved.to_df(), raw_mean.loc[saved.obs_names, saved.var_names])
                    # Reusing reference.h5ad must work without the single-cell input.
                    (path / 'scrna.h5ad').unlink()
                    ns['BUILD_REFERENCE'] = False
                    with contextlib.redirect_stdout(log):
                        for cell in cells:
                            if cell['cell_type'] == 'code' and (
                                    cell.get('id', '').startswith('reference-') or cell.get('id') == 'cell-04'):
                                exec(compile(''.join(cell['source']), notebook.name, 'exec'), ns)
                    np.testing.assert_allclose(ns['reference'], saved.to_df())
                    print('PASS: raw-layer reference preparation, top50 markers, both profile modes and reuse.')
                print(f'PASS: {notebook.name}: input, fit, plot and export.', flush=True)
            except Exception:
                print(log.getvalue()[-6000:])
                raise
            finally:
                plt.close('all')
        required = ['reference_free/deconvolved.h5ad', 'reference_based/cell_fractions.csv',
                    'histology/deconvolved.h5ad', 'cross_modality/RNA_weights.csv',
                    'cross_modality/protein_weights.csv', 'multisample/effects.npz',
                    'multiomics/shared_weights.csv', 'multiomics/RNA_profiles.csv',
                    'multiomics/protein_profiles.csv']
        assert all((path/'outputs'/name).is_file() for name in required)
        assert len(list((path/'outputs/figures').glob('*.png'))) == 6
        print('PASS: all six notebooks run without tutorial data or saved fits.')


if __name__ == '__main__':
    main()
