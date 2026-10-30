Python and command-line entry points
====================================

Inputs
------

Count tables have spots in rows and features in columns. Coordinates have matching
spot IDs and columns ``x,y``. Reference tables have types in rows and the same
ordered features as the counts. Use raw or explicitly prepared nonnegative
measurements, with positive total signal per spot. Do not silently reorder or
log-transform a fitting matrix.

Core reference-free command
---------------------------

.. code-block:: bash

   python run_spdecon.py core \
     --counts X.csv --coordinates coords.csv \
     --k 11 --seeds 1 3 5 --deterministic --output runs/example

Use a new output directory. This performs full three-branch initialization,
consensus alignment and KL + TV refinement. The CLI writes factors, settings,
optimization histories and convergence diagnostics. A supplied ``block`` coordinate
column keeps graph construction within sections. ``--spatial-weight 0`` runs the
complete count-only model; omitting it uses the recorded seed-1 calibration rule.

Model entry points
------------------

.. list-table:: Public functions
   :header-rows: 1
   :widths: 42 58

   * - Function
     - Purpose
   * - ``spDecon.consensus.build_consensus``
     - Three branch fits, profile matching, equal-weight consensus
   * - ``spDecon.spatial.spatial_graph`` / ``calibrate`` / ``refine``
     - Coordinate graph, truth-free coefficient calibration and joint refinement
   * - ``spDecon.reference_spatial.fit``
     - Fixed-reference RNA fitting
   * - ``spDecon.reference.fit_reference`` / ``cell_fractions``
     - General fixed-profile fit and yield-based fraction conversion
   * - ``spDecon.reference.fit_reference_consensus``
     - Separate fixed-reference ATAC three-initializer protocol
   * - ``spDecon.image_features.extract_histology``
     - Registered H&E → HIPT/UNI embeddings → 32-bin morphology features
   * - ``spDecon.histology.fit_histology``
     - RNA/morphology reconstruction with gated spatial TV
   * - ``spDecon.multisample.fit_multisample``
     - Shared plus subject and within-subject condition profiles
   * - ``spDecon.multiomics.fit_multiomics``
     - Paired assays with shared spot weights and free or fixed profiles
   * - ``spDecon.reproducibility.set_seed``
     - Explicit RNG and optional deterministic execution settings

The :doc:`tutorial` notebooks provide complete calls with actual data, coefficients,
input ordering and output interpretation. Histology, multi-sample and multi-omics
functions return ``JointFit`` objects with ``proportions``, ``profiles``,
``history``, ``diagnostics`` and mode-specific ``extra`` arrays.

Progress output
---------------

The main fitting functions print progress by default. Initialization branches
are named before they run; optimization reports its objective, KKT residuals and
elapsed time at intervals, then explicitly states whether it converged or reached
its iteration limit. The separate ATAC reference wrapper reports its start and
final status.

Use ``verbose=False`` to silence a fitting call. For the low-level core solver,
set ``SolverConfig(verbose=False)``. The ``log_every`` parameter controls output
frequency: the core defaults to 10 outer iterations, the strict reference solver
to 1,000 steps, and joint optimization to 500 iterations. Joint messages are
printed at the first convergence-check point at or after each output interval.
Changing output frequency does not change convergence-check frequency.

``objective`` is the complete normalized loss, including active penalties. KKT
residuals measure constrained stationarity. Histories retain their existing
evaluation frequency even when fewer rows are printed. Models with different
penalties or data should not be ranked by their total loss values alone.

The notebooks distinguish loading a saved full fit from running a new fit.
Saved mode displays recorded training diagnostics; ``RUN_FIT=True`` enables live
optimization output. Both paths retain the full dataset.
