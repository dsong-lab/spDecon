"""Check named flat/nested effects against the legacy model and CLI exports."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spDecon.multisample import fit_multisample


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(15)
    labels = [('D1', 'early'), ('D1', 'late'), ('D2', 'early')]
    metadata = pd.DataFrame(
        [dict(donor=donor, timepoint=time, section_id=f'section_{j}', x=i, y=0)
         for j, (donor, time) in enumerate(labels) for i in range(8)],
        index=[f'spot_{i}' for i in range(24)])
    b0 = np.array([[8]*6+[1]*6, [1]*6+[8]*6], dtype=float)
    b0 /= b0.sum(1, keepdims=True)
    p0 = np.tile(np.array([[.8, .2], [.2, .8]]), (12, 1))
    counts = pd.DataFrame([rng.multinomial(1000, p@b0) for p in p0],
                          index=metadata.index, columns=[f'g{i}' for i in range(12)])
    options = dict(initial_proportions=p0, initial_profiles=b0, spatial_weight=.01,
                   device='cpu', random_state=1, verbose=False, max_iter=3000)
    effects = {
        # Child first deliberately tests that parent links determine ordering.
        'visit': dict(column='timepoint', parent='donor', penalty=20.),
        'donor': dict(column='donor', penalty=10.),
    }
    old = fit_multisample(counts, metadata, 2, group_by=['donor', 'timepoint'],
                          subject_column='donor', eta_subject=10., eta_condition=20., **options)
    fit = fit_multisample(counts, metadata, 2, effects=effects, **options)
    assert fit.diagnostics['converged'], fit.diagnostics
    for a, b in ((old.proportions, fit.proportions), (old.profiles, fit.profiles)):
        np.testing.assert_array_equal(a, b)
    assert old.diagnostics['objective'] == fit.diagnostics['objective']
    extra = fit.extra
    reconstructed = np.broadcast_to(extra['shared_profiles'], (len(counts), 2, 12)).copy()
    for name in extra['effect_names']:
        reconstructed += extra[f'effect_delta_{name}'][extra[f'effect_index_{name}']]
    np.testing.assert_allclose(reconstructed, fit.profiles[extra['group_index']], atol=1e-14)
    np.testing.assert_allclose(extra['effect_delta_donor'].sum(0), 0, atol=1e-14)
    for parent in np.unique(extra['group_parent']):
        np.testing.assert_allclose(extra['effect_delta_visit'][extra['group_parent'] == parent].sum(0), 0, atol=1e-14)
    np.testing.assert_allclose(extra['effect_delta_visit'][2], 0, atol=1e-14)
    np.testing.assert_array_equal(extra['effect_labels_visit'], labels)

    flat = fit_multisample(counts, metadata, 2,
                          effects={'section': dict(column='section_id', penalty=10.)}, **options)
    flat_old = fit_multisample(counts, metadata, 2, group_by='section_id', **options)
    np.testing.assert_array_equal(flat.proportions, flat_old.proportions)
    np.testing.assert_array_equal(flat.profiles, flat_old.profiles)
    np.testing.assert_allclose(flat.extra['effect_delta_section'].sum(0), 0, atol=1e-14)
    assert flat.diagnostics['converged'], flat.diagnostics

    invalid = [
        {}, {'bad': dict(column='donor', penalty=-1)},
        {'bad': dict(column='donor', penalty=np.nan)},
        {'bad': dict(column='missing', penalty=1)},
        {'a': dict(column='donor', penalty=1), 'b': dict(column='timepoint', penalty=1)},
        {'a': dict(column='donor', penalty=1, parent='b'), 'b': dict(column='timepoint', penalty=1, parent='a')},
        {'a': dict(column='donor', penalty=1, parent='missing')},
        {'a': dict(column='donor', penalty=1, typo=True)},
        {'a': dict(column='donor', penalty=1), 'b': dict(column='donor', penalty=1, parent='a')},
        {str(i): dict(column='donor', penalty=1) for i in range(3)},
    ]
    for spec in invalid:
        try:
            fit_multisample(counts, metadata, 2, effects=spec, **options)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid effects accepted: {spec}')
    try:
        fit_multisample(counts, metadata, 2, effects=effects, group_by='donor', **options)
    except ValueError as error:
        assert 'legacy' in str(error)
    else:
        raise AssertionError('Mixed interfaces must be rejected')

    with TemporaryDirectory(prefix='spdecon-effects-') as temporary:
        root = Path(temporary)
        counts.to_csv(root/'counts.csv')
        metadata.to_csv(root/'metadata.csv')
        (root/'effects.json').write_text(json.dumps(effects))
        np.savez(root/'initial.npz', P=p0, B=b0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1')
        env.pop('PYTHONPATH', None)
        command = [sys.executable, str(ROOT/'run_spdecon.py'), 'multisample',
                   '--counts', str(root/'counts.csv'), '--metadata', str(root/'metadata.csv'),
                   '--effects', str(root/'effects.json'), '--initialization', str(root/'initial.npz'),
                   '--k', '2', '--device', 'cpu', '--spatial-weight', '.01', '--max-iter', '3000',
                   '--output', str(root/'fit')]
        run = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
        assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]
        assert json.loads((root/'fit/COMPLETE.json').read_text())['status'] == 'passed'
        with np.load(root/'fit/fit.npz', allow_pickle=False) as saved:
            for key in saved.files:
                saved[key]  # Every exported array must load without pickle.
            np.testing.assert_allclose(saved['P'], fit.proportions, atol=1e-12)
            np.testing.assert_array_equal(saved['effect_labels_visit'], labels)
        exported = pd.read_csv(root/'fit/effect_visit_levels.csv', index_col=0)
        np.testing.assert_array_equal(exported.to_numpy(), labels)
        diagnostics = json.loads((root/'fit/diagnostics.json').read_text())
        assert diagnostics['effects']['visit']['parent'] == 'donor'
        assert set(diagnostics['effect_penalties']) == {'donor', 'visit'}
    print('PASS: converged CPU fits; exact legacy equivalence; flat/nested centering; '
          'singleton child; invalid designs; JSON CLI and named output labels.')


if __name__ == '__main__':
    main()
