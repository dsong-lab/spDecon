"""Check that all six included tutorial datasets load with valid spot alignment."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutorials import common


def main():
    for name in ('cerebellum', 'xenium', 'g2', 'rcc', 'crc', 'tonsil'):
        path, data = common.dataset(name)
        with data:
            n = len(data['spots'])
            assert data['coordinates'].shape == (n, 2), name
            for key in data.files:
                if key == 'Y' or key.startswith('Y_'):
                    values = data[key]
                    assert values.shape[0] == n, (name, key)
                    assert np.isfinite(values).all() and (values >= 0).all(), (name, key)
            assert any(p.name != 'data.npz' for p in path.glob('*.npz')), name
        print(f'PASS: {name}, {n:,} spots; included inputs and saved results found.')


if __name__ == '__main__':
    main()
