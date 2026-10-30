"""Registered H&E -> HIPT + UNI -> PCA99 -> two 16-word spot histograms.

The frozen encoders and their extraction arithmetic are supplied by the
spEnhance source dependency. This adapter performs the spDecon PCA/dictionary
steps and records the source and geometry used. Pretrained weights are external.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


def capture_indices(xy, radius, shape):
    """Grid centers are 16 pixels apart, with first center at pixel 7.5."""
    height, width = shape
    indices, sizes = [], []
    for x, y in np.asarray(xy, float):
        xs = np.arange(max(0, int(np.floor((x-radius)/16))), min(width, int(np.ceil((x+radius)/16))+1))
        ys = np.arange(max(0, int(np.floor((y-radius)/16))), min(height, int(np.ceil((y+radius)/16))+1))
        xx, yy = np.meshgrid(xs, ys)
        keep = (xx*16+7.5-x)**2+(yy*16+7.5-y)**2 <= radius**2
        ix = (yy[keep]*width+xx[keep]).ravel()
        if not len(ix):
            raise ValueError('A capture footprint contains no feature-grid centers')
        indices.append(ix)
        sizes.append(len(ix))
    return np.concatenate(indices), np.r_[0, np.cumsum(sizes)]


def affine_capture_indices(physical_xy, image_xy, radius, image_to_physical, shape):
    """Pool circular physical footprints after an affine image registration."""
    transform = np.asarray(image_to_physical, float)
    extent = radius*np.linalg.norm(np.linalg.inv(transform)[:2, :2], axis=1)
    height, width = shape
    indices, sizes = [], []
    for center, (x, y) in zip(np.asarray(physical_xy, float), np.asarray(image_xy, float)):
        xx, yy = np.meshgrid(np.arange(max(0, int(np.floor((x-extent[0])/16))), min(width, int(np.ceil((x+extent[0])/16))+1)),
                             np.arange(max(0, int(np.floor((y-extent[1])/16))), min(height, int(np.ceil((y+extent[1])/16))+1)))
        points = np.c_[xx.ravel()*16+7.5, yy.ravel()*16+7.5, np.ones(xx.size)]@transform.T
        keep = ((points[:, :2]-center)**2).sum(1) <= radius**2
        ix = (yy.ravel()[keep]*width+xx.ravel()[keep])
        indices.append(ix)
        sizes.append(len(ix))
    return np.concatenate(indices), np.r_[0, np.cumsum(sizes)]


def covariance_pca(raw, output):
    """Exact whole-image covariance PCA; no feature standardization."""
    height, width, dim = raw.shape
    flat = raw.reshape(-1, dim)
    total, cross = np.zeros(dim), np.zeros((dim, dim))
    for start in range(0, len(flat), 8192):
        block = flat[start:start+8192].astype(np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Encoder features contain nonfinite values')
        total += block.sum(0)
        cross += block.T@block
    mean = total/len(flat)
    values, vectors = np.linalg.eigh(cross/len(flat)-np.outer(mean, mean))
    order = np.argsort(values)[::-1]
    values, vectors = np.maximum(values[order], 0), vectors[:, order]
    if values.sum() <= 0:
        raise ValueError('Image features have zero variance')
    cumulative = np.cumsum(values)/values.sum()
    n = int(np.searchsorted(cumulative, .99)+1)
    basis = vectors[:, :n]
    scores = np.lib.format.open_memmap(output, mode='w+', dtype=np.float32, shape=(height, width, n))
    dest = scores.reshape(-1, n)
    for start in range(0, len(flat), 8192):
        dest[start:start+8192] = (flat[start:start+8192].astype(float)-mean)@basis
    scores.flush()
    return scores, dict(mean=mean, components=basis.T, eigenvalues=values,
                        retained_dimensions=n, explained_variance=float(cumulative[n-1]))


def dictionary_from_scores(scores, indices, offsets):
    """Fit the frozen 16-word dictionary and pool the specified capture footprints."""
    values = np.asarray(scores.reshape(-1, scores.shape[-1])[indices], np.float32)
    selected = np.random.default_rng(0).choice(len(values), min(30000, len(values)), replace=False)
    dictionary = MiniBatchKMeans(n_clusters=16, random_state=0, n_init=5,
                                batch_size=2048, max_iter=300).fit(values[selected])
    labels = dictionary.predict(values)
    q = np.stack([np.bincount(labels[a:b], minlength=16)/(b-a)
                  for a, b in zip(offsets[:-1], offsets[1:])])
    return q, dictionary, len(selected)


def build_morphology(directory, spots, radius):
    directory = Path(directory)
    histograms, records, first_shape = [], {}, None
    for branch, suffix, keys in [('hipt', 'vit', ['sub', 'rgb', 'cls']), ('uni', 'uni', ['his', 'rgb', 'pos'])]:
        # These files are created by this run's encoder subprocesses.
        with (directory/f'embeddings-hist-{suffix}.pickle').open('rb') as stream:
            features = pickle.load(stream)
        raw = np.concatenate([features[key] for key in keys]).transpose(1, 2, 0).copy()
        del features
        if first_shape is not None and raw.shape[:2] != first_shape:
            raise ValueError('HIPT and UNI feature grids differ')
        first_shape = raw.shape[:2]
        indices, offsets = capture_indices(spots[['x', 'y']], radius, raw.shape[:2])
        scores, transform = covariance_pca(raw, directory/f'{branch}_pca99.npy')
        del raw
        np.savez_compressed(directory/f'{branch}_pca_transform.npz', **transform)
        q, dictionary, n_selected = dictionary_from_scores(scores, indices, offsets)
        histograms.append(q)
        np.savez_compressed(directory/f'{branch}_dictionary.npz', centers=dictionary.cluster_centers_)
        records[branch] = dict(pca_dimensions=int(transform['retained_dimensions']),
                               explained_variance=transform['explained_variance'],
                               dictionary_size=16, training_positions=n_selected)
    q = np.concatenate(histograms, axis=1)/2
    ids = spots.index.to_numpy(str)
    np.savez_compressed(directory/'morphology.npz', Q=q, spots=ids)
    frame = pd.DataFrame(q, index=spots.index, columns=[f'{name}_{i+1}' for name in ('HIPT', 'UNI') for i in range(16)])
    frame.to_csv(directory/'morphology.csv')
    (directory/'dictionary.json').write_text(json.dumps(records, indent=2)+'\n')
    return frame


def extract_histology(image, spots, *, microns_per_pixel, radius_pixels,
                      encoder_source, output, device='cuda', offline=False):
    """Extract the paper's two-encoder features from an image and pixel centers.

    `encoder_source` is a spEnhance checkout with pretrained HIPT checkpoints;
    UNI weights are resolved through the user's Hugging Face cache/access.
    `spots` is a spot-indexed DataFrame with x/y in original image pixels.
    """
    from PIL import Image
    source, image, output = Path(encoder_source).resolve(), Path(image).resolve(), Path(output).resolve()
    needed = ['rescale.py', 'preprocess.py', 'extract_features_vit.py', 'histological_feature_map.py',
              'RGB_feature_map.py', 'pos_feature_map.py', 'hipt_4k.py', 'hipt_model_utils.py',
              'vision_transformer.py', 'vision_transformer4k.py', 'utils.py', 'image.py',
              'reduce_dim.py', 'connected_components.py', 'checkpoints/vit256_small_dino.pth',
              'checkpoints/vit4k_xs_dino.pth']
    for name in needed:
        if not (source/name).is_file():
            raise FileNotFoundError(f'Missing encoder dependency: {source/name}')
    if not image.is_file() or image.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
        raise ValueError('Provide an existing RGB H&E JPEG, PNG or TIFF image')
    if not isinstance(spots, pd.DataFrame) or not spots.index.is_unique or not {'x', 'y'}.issubset(spots.columns):
        raise ValueError('spots must have unique spot IDs and registered image-pixel x/y columns')
    xy = spots[['x', 'y']].to_numpy(float)
    if len(xy) < 1 or not np.isfinite(xy).all() or not np.isfinite([microns_per_pixel, radius_pixels]).all() or min(microns_per_pixel, radius_pixels) <= 0:
        raise ValueError('Pixel centers, image resolution and capture radius must be valid')
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image) as im:
        shape = np.asarray(im.size)
        if im.mode != 'RGB':
            raise ValueError('Convert the registered H&E image to RGB before extraction')
    if (xy-radius_pixels < 0).any() or (xy+radius_pixels >= shape).any():
        raise ValueError('A capture footprint extends beyond the image')
    if output.exists():
        raise FileExistsError('Choose a new image-feature output directory')
    output.mkdir(parents=True)
    suffix = {'.jpeg': '.jpg', '.tif': '.tiff'}.get(image.suffix.lower(), image.suffix.lower())
    (output/f'he-raw{suffix}').symlink_to(image)
    (output/'pixel-size-raw.txt').write_text(str(microns_per_pixel))
    (output/'pixel-size.txt').write_text('0.5')
    (output/'radius-raw.txt').write_text(str(radius_pixels))
    spots[['x', 'y']].to_csv(output/'locs-raw.csv')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if offline:
        env['HF_HUB_OFFLINE'] = '1'
    def run(script, *args):
        subprocess.run([sys.executable, '-B', str(source/script), str(output)+'/', *args], cwd=source, env=env, check=True)
    run('rescale.py', '--image', '--locs', '--radius')
    run('preprocess.py', '--image')
    scaled = pd.read_csv(output/'locs.csv', index_col=0, keep_default_na=False)
    scaled.index = spots.index.astype(str)
    scaled.to_csv(output/'locs.csv')
    radius = float((output/'radius.txt').read_text())
    with Image.open(output/'he.jpg') as im:
        size = np.asarray(im.size)
    margin = np.c_[scaled.to_numpy()-radius, size-scaled.to_numpy()-radius]
    if (margin[:, :2] < 192).any() or (margin[:, 2:] < 256).any():
        raise ValueError('Capture footprints enter incomplete HIPT shift margins (192 pixels top/left, 256 bottom/right at 0.5 um/pixel)')
    for worker in ('--hipt-worker', '--uni-worker'):
        subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), worker,
                        str(source), str(output), device], cwd=source, env=env, check=True)
    frame = build_morphology(output, scaled, radius)
    metadata = dict(image=str(image),
                    original_microns_per_pixel=microns_per_pixel, target_microns_per_pixel=.5,
                    original_radius_pixels=radius_pixels, scaled_radius_pixels=radius,
                    device=device, n_spots=len(spots), encoders=['HIPT', 'UNI'],
                    uni_model='hf-hub:MahmoodLab/uni',
                    workflow='16-shift HIPT + UNI; covariance PCA99 per encoder; 16-word dictionaries; capture frequencies',
                    molecular_data_used=False)
    (output/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return frame


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--hipt-worker':
        import runpy
        import torch
        source, output, device = sys.argv[2:]
        sys.path.insert(0, source)
        load = torch.load
        checkpoints = {(Path(source)/'checkpoints'/name).resolve() for name in
                       ('vit256_small_dino.pth', 'vit4k_xs_dino.pth')}
        def load_hipt(path, *args, **kwargs):
            # Official HIPT checkpoint dictionaries predate weights_only loading.
            if isinstance(path, (str, Path)) and Path(path).resolve() in checkpoints:
                kwargs.setdefault('weights_only', False)
            return load(path, *args, **kwargs)
        torch.load = load_hipt
        torch.manual_seed(0)
        sys.argv = [str(Path(source)/'extract_features_vit.py'), output+'/', '--device', device]
        runpy.run_path(sys.argv[0], run_name='__main__')
        return
    if len(sys.argv) > 1 and sys.argv[1] == '--uni-worker':
        source, output, device = sys.argv[2:]
        sys.path.insert(0, source)
        import torch
        import histological_feature_map as official
        from RGB_feature_map import main as rgb
        from pos_feature_map import main as pos
        torch.set_num_threads(1)
        torch.manual_seed(0)
        np.random.seed(0)
        # Use existing Hugging Face credentials/cache; never pass or print tokens.
        official.login = lambda *_args, **_kwargs: None
        if device != 'cuda' and torch.cuda.is_available():
            raise ValueError('The original UNI adapter selects CUDA when available; use device=cuda')
        hist = official.main(output+'/', None)
        with (Path(output)/'embeddings-hist-uni.pickle').open('wb') as stream:
            pickle.dump(dict(his=hist, rgb=rgb(output+'/'), pos=pos(output+'/', hist)), stream)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--spots', required=True, help='CSV: spot ID, x, y in original image pixels')
    parser.add_argument('--microns-per-pixel', required=True, type=float)
    parser.add_argument('--radius-pixels', required=True, type=float)
    parser.add_argument('--encoder-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    values = vars(args)
    from ._io import read_frame
    values['spots'] = read_frame(values['spots'])
    extract_histology(**values)


if __name__ == '__main__':
    main()
