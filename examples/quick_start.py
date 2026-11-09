"""Generate a small synthetic dataset and fit it on CPU using the public CLI."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/quick_start'),
                        help='New directory for inputs and fitted results')
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Three synthetic types, 60 genes and 72 mixtures of 20 cells each.
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
    counts = pd.DataFrame(mixtures, index=[f'spot_{i}' for i in range(72)],
                          columns=[f'gene_{i}' for i in range(60)])
    counts.index.name = 'spot'
    coordinates = pd.DataFrame({'x': np.tile(np.arange(8), 9),
                                'y': np.repeat(np.arange(9), 8)}, index=counts.index)
    counts.to_csv(output / 'counts.csv')
    coordinates.to_csv(output / 'coordinates.csv')

    subprocess.run([sys.executable, str(ROOT / 'run_spdecon.py'), 'core',
        '--counts', str(output / 'counts.csv'),
        '--coordinates', str(output / 'coordinates.csv'),
        '--k', '3', '--device', 'cpu', '--seeds', '1', '--deterministic',
        '--output', str(output / 'fit')], check=True)

    weights = pd.read_csv(output / 'fit/seed_1/P.csv', index_col=0)
    profiles = pd.read_csv(output / 'fit/seed_1/B.csv', index_col=0)
    status = json.loads((output / 'fit/COMPLETE.json').read_text())
    assert status['status'] == 'passed', status
    assert weights.shape == (72, 3) and profiles.shape == (3, 60)
    for values in (weights.to_numpy(), profiles.to_numpy()):
        assert np.isfinite(values).all() and (values >= 0).all()
        np.testing.assert_allclose(values.sum(1), 1, atol=2e-6, rtol=0)
    print(f'PASS: 72 spots, 60 genes, 3 learned components; results in {output}')
    print(weights.head().to_string())


if __name__ == '__main__':
    main()
