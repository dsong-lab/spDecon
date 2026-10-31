"""Check cached loading and explicit refresh without downloading real data."""
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tutorials import common


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        cache = root / 'tutorial_data/g2'
        cache.mkdir(parents=True)
        np.savez(cache / 'data.npz', Y=[0])
        payload = root / 'payload/tutorial_data'
        (payload / 'g2').mkdir(parents=True)
        np.savez(payload / 'g2/data.npz', Y=[1])
        archive = root / 'fixture.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(payload, arcname='tutorial_data')
        with patch.object(common, 'ROOT', root), patch.object(common, 'urlretrieve') as download:
            download.side_effect = lambda url, path: shutil.copyfile(archive, path)
            _, data = common.dataset('g2')
            np.testing.assert_array_equal(data['Y'], [0])
            data.close()
            download.assert_not_called()
            _, data = common.dataset('g2', refresh=True)
            np.testing.assert_array_equal(data['Y'], [1])
            data.close()
            download.assert_called_once_with(common.DATA_URL, root / 'tutorial_data/spdecon-tutorial-data.tar.gz')
            _, data = common.dataset('g2')
            data.close()
            download.assert_called_once()
    print('PASS: cached inputs are reused and explicit refresh retrieves the current bundle.')


if __name__ == '__main__':
    main()
