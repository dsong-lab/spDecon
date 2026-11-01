"""Check bundled preprocessing and extraction dispatch without pretrained weights.

Only the two encoder subprocesses are replaced with synthetic embedding maps.
Image preparation, spot pooling, PCA, dictionaries and output files run normally.
"""
import json
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spDecon.image_features import extract_histology


def main():
    real_run = subprocess.run
    workers = []
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoints = root / 'weights'
        checkpoints.mkdir()
        for name in ('vit256_small_dino.pth', 'vit4k_xs_dino.pth'):
            (checkpoints / name).touch()
        image = root / 'he.png'
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 256, (672, 672, 3), dtype=np.uint8)).save(image)
        spots = pd.DataFrame({'x': [300, 340], 'y': [300, 340]}, index=['spot02', 'spot01'])
        output = root / 'features'

        def run(command, **options):
            source = Path(options['cwd'])
            assert source == Path(__file__).resolve().parents[1] / 'spDecon/_image_encoders'
            assert options['env']['SPDECON_HIPT_CHECKPOINT_DIR'] == str(checkpoints)
            assert options['env']['HF_HUB_OFFLINE'] == '1'
            if '--hipt-worker' in command or '--uni-worker' in command:
                worker = command[3]
                workers.append(worker)
                assert Path(command[4]) == source
                assert Path(command[5]) == output
                with Image.open(output / 'he.jpg') as im:
                    width, height = im.size
                groups = {'sub': 4, 'rgb': 3, 'cls': 2} if worker == '--hipt-worker' else {'his': 4, 'rgb': 3, 'pos': 2}
                maps = {key: rng.normal(size=(size, height // 16, width // 16)).astype(np.float32)
                        for key, size in groups.items()}
                suffix = 'vit' if worker == '--hipt-worker' else 'uni'
                with (output / f'embeddings-hist-{suffix}.pickle').open('wb') as stream:
                    pickle.dump(maps, stream)
                return subprocess.CompletedProcess(command, 0)
            return real_run(command, **options)

        with patch('spDecon.image_features.subprocess.run', side_effect=run):
            q = extract_histology(image, spots, microns_per_pixel=.5, radius_pixels=48,
                                  checkpoint_dir=checkpoints, output=output, device='cpu', offline=True)
        assert workers == ['--hipt-worker', '--uni-worker']
        assert q.shape == (2, 32) and np.isfinite(q).all().all() and (q >= 0).all().all()
        np.testing.assert_allclose(q.to_numpy().reshape(2, 2, 16).sum(axis=2), .5)
        with np.load(output / 'morphology.npz') as saved:
            np.testing.assert_array_equal(saved['spots'], spots.index)
            np.testing.assert_array_equal(saved['Q'], q)
        metadata = json.loads((output / 'provenance.json').read_text())
        assert metadata['hipt_checkpoint_dir'] == str(checkpoints)
        assert not metadata['molecular_data_used']
        try:
            extract_histology(image, spots, microns_per_pixel=.5, radius_pixels=48,
                              checkpoint_dir=root / 'missing', output=root / 'unused')
        except FileNotFoundError as error:
            assert 'Missing HIPT checkpoint' in str(error)
        else:
            raise AssertionError('Missing checkpoints must fail before extraction')
    print('PASS: bundled preprocessing, encoder dispatch and saved morphology; encoder outputs simulated.')


if __name__ == '__main__':
    main()
