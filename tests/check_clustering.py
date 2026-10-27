"""CPU check for standalone fixed-K clustering and its input validation."""
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/spDecon_core/src"))
from louvain_init_poisson import louvain_labels


def main():
    rng = np.random.default_rng(23)
    x = np.vstack([rng.normal(loc=j * 5, scale=0.6, size=(24, 6)) for j in range(3)])
    labels, neighbors = louvain_labels(x, 3)
    assert labels.shape == (72,) and np.unique(labels).size == 3
    assert adjusted_rand_score(np.repeat(np.arange(3), 24), labels) == 1
    assert 2 <= neighbors < len(x)
    single, used = louvain_labels(x, 1)
    assert np.array_equal(single, np.zeros(len(x))) and used == 0
    singleton, used = louvain_labels(x, len(x))
    assert np.unique(singleton).size == len(x) and used == 0
    for values, k, nn in (
        (np.empty((0, 6)), 1, 20),
        (np.full((5, 2), np.nan), 2, 20),
        (x, 0, 20), (x, 73, 20), (x, 2.5, 20), (x, 3, 1),
    ):
        try:
            louvain_labels(values, k, initial_neighbors=nn)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid clustering input was accepted")
    print("PASS: standalone clustering recovers three groups and validates edge cases.")


if __name__ == "__main__":
    main()
