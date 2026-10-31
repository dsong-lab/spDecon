"""Check same-Q, exposure-weighted calibration and the paired coordinate-TV control."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spDecon.histology import fit_histology, initialize_morphology
from spDecon.spatial import spatial_graph


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    y = (rng.poisson(3, (12, 18)) + 1) * np.arange(1, 13)[:, None]
    q = np.concatenate([.5 * rng.dirichlet(np.ones(16), 12) for _ in range(2)], axis=1)
    xy = np.array([(x, z) for x in range(4) for z in range(3)])
    p0, b0 = rng.dirichlet(np.ones(3), 12), rng.dirichlet(np.ones(18), 3)
    options = dict(initial_proportions=p0, initial_profiles=b0, device='cpu',
                   max_iter=1, verbose=False)
    fit = fit_histology(y, q, xy, 3, **options)
    p = torch.tensor(.9 * p0 + .1 / 3, requires_grad=True)
    b, counts, qt = map(lambda x: torch.tensor(x, dtype=torch.float64), (b0, y, q))
    h = initialize_morphology(p.detach(), q, y.sum(1))
    exposure = counts.sum(1)
    v = exposure / exposure.mean()
    mean = exposure[:, None] * (p @ b)
    rna = (counts * torch.log(counts / mean) + mean - counts).sum() / counts.sum()
    image_mean = p @ h
    image = (v[:, None] * (qt * torch.log(qt / image_mean) + image_mean - qt)).sum() / len(y)
    i, j, weights, _ = spatial_graph(xy, np.repeat('section', len(y)))
    edge_exposure = 2 * v[i] * v[j] / (v[i] + v[j])
    delta = p[i] - p[j]
    tv = (torch.tensor(weights) * edge_exposure *
          (torch.sqrt(delta.square().sum(1) + .02**2) - .02)).sum() / weights.sum()
    gradients = [torch.autograd.grad(loss, p, retain_graph=True)[0] for loss in (rna, image, tv)]
    norms = [float(torch.linalg.norm(g - g.mean(1, keepdim=True))) for g in gradients]
    np.testing.assert_allclose(
        [fit.diagnostics['image_weight'], fit.diagnostics['spatial_weight']],
        [.30 * norms[0] / norms[1], .05 * norms[0] / norms[2]], atol=1e-12, rtol=1e-10)
    differences = ((np.sqrt(q[i]) - np.sqrt(q[j]))**2).sum(1)
    np.testing.assert_allclose(fit.extra['morphology_gate'],
                               np.exp(-differences / np.median(differences[differences > 0])))
    control = fit_histology(y, None, xy, 3, spatial_weight=fit.diagnostics['spatial_weight'], **options)
    assert control.diagnostics['image_weight'] == 0
    assert control.diagnostics['spatial_weight'] == fit.diagnostics['spatial_weight']
    np.testing.assert_array_equal(control.extra['morphology_gate'], 1)
    changed = fit_histology(y, q[::-1].copy(), xy, 3, **options)
    np.testing.assert_allclose(changed.diagnostics['spatial_weight'], fit.diagnostics['spatial_weight'])
    assert not np.isclose(changed.diagnostics['image_weight'], fit.diagnostics['image_weight'])
    print('PASS: weighted calibration agrees with autograd; changed Q recalibrates image loss; paired TV is shared.')


if __name__ == '__main__':
    main()
