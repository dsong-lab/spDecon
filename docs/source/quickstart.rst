Quick start
===========

Run your first reference-free fit with a small, generated dataset. The example
uses **72 synthetic spots, 60 genes and three components**, runs on CPU, and
requires no data download. It pools simulated single-cell counts into mixtures
on an artificial grid. This is a software example, not a study result.

Run the example
---------------

Complete :doc:`installation`, activate that environment, and run this command
from the cloned repository root:

.. code-block:: bash

   python examples/quick_start.py

The script generates count and coordinate CSVs, then calls the public CLI with
``--k 3 --device cpu --seeds 1 --deterministic``. It runs all three initialization
branches, consensus, spatial-weight calibration and joint refinement. Fitting
prints progress and convergence diagnostics.

At completion, the script checks convergence, matrix shapes and normalization,
prints ``PASS: 72 spots, 60 genes, 3 learned components``, and displays the first
five rows of the learned weights. If fitting does not converge, the command
fails and leaves diagnostics for inspection.

Use a new directory for another run:

.. code-block:: bash

   python examples/quick_start.py --output outputs/quick_start_2

Inspect the result
------------------

The default output directory is ``outputs/quick_start/``.

.. list-table:: Files to open first
   :header-rows: 1
   :widths: 42 58

   * - File
     - Contents
   * - ``counts.csv``, ``coordinates.csv``
     - Generated inputs, with matching spot IDs.
   * - ``fit/seed_1/P.csv``
     - 72 × 3 spot-by-component molecular weights; each row sums to one.
   * - ``fit/seed_1/B.csv``
     - 3 × 60 component-by-gene profiles; each row sums to one.
   * - ``fit/seed_1/diagnostics.json``
     - Convergence status, objective, KKT residuals and branch diagnostics.
   * - ``fit/COMPLETE.json``
     - Overall run status; ``passed`` requires converged branches and refinement.
   * - ``fit/config.json``, ``fit/calibration.json``
     - Recorded settings and the calibrated spatial coefficient.

For example, read the fitted weights in Python:

.. code-block:: python

   import pandas as pd

   weights = pd.read_csv("outputs/quick_start/fit/seed_1/P.csv", index_col=0)
   print(weights.head())
   print(weights.sum(axis=1).head())

The components are learned and may appear in a different order across runs.
Their weights describe molecular contributions; naming them or interpreting
them as cell fractions requires additional evidence. See :doc:`model`.

Continue with your data
-----------------------

Use :doc:`prepare_data` to prepare real inputs, then choose one of the six
workflows in :doc:`tutorial`. See :doc:`api` for CLI options and Python outputs.

Inspect the complete source in
`examples/quick_start.py <https://github.com/dsong-lab/spDecon/blob/main/examples/quick_start.py>`_.
