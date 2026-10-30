Functions and parameters
========================

Inputs
------

The tutorials organize inputs in AnnData; see :ref:`anndata-input` for fields,
``.h5ad`` loading and CSV conversion. Pass ``adata.to_df()`` or its NumPy array
to the numerical functions, with ``adata.obsm["spatial"]`` for coordinates and
``adata.obs`` for sample metadata. Use ``to_df(layer="counts")`` when raw counts
are stored in a layer instead of ``X``.

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

Reference-free fitting
----------------------

.. code-block:: python

   initial = build_consensus(y, n_components=K, random_state=1, verbose=True)
   i, j, weights, info = spatial_graph(xy, sections)
   calibration = calibrate(
       y, initial.consensus_proportions, initial.consensus_profiles,
       i, j, weights)
   result = refine(
       y, initial.consensus_proportions, initial.consensus_profiles,
       i, j, weights, strength=calibration["spatial_weight"], seed=1)

Import these functions from ``spDecon.consensus`` and ``spDecon.spatial``.
The :doc:`tutorial` page gives a complete AnnData example.

``y``, ``n_components``
   ``y`` is a dense ``N × G`` raw measurement matrix. ``n_components`` is the
   number ``K`` of learned components.
``random_state``, ``solver_config``
   ``random_state`` sets initialization randomness. ``solver_config`` optionally
   supplies ``spDecon.config.SolverConfig`` for the three branch fits; its defaults
   include 500 outer iterations, relative tolerance ``1e-6`` and P/B KKT
   tolerances ``1e-4``. It does not change the separate refinement configuration.
``xy``, ``sections``
   ``N × 2`` coordinates and one physical section ID per spot. ``spatial_graph``
   returns two endpoint arrays, edge weights and graph information.
``strength``, ``seed``
   Spatial penalty coefficient and refinement seed. Zero strength disables TV.
   ``calibrate`` computes a coefficient from the seed-1 unrefined consensus,
   without labels; retain that coefficient when comparing other fitting seeds.

``build_consensus`` returns ``consensus_proportions`` (``N × K``) and
``consensus_profiles`` (``K × G``), together with branch results and diagnostics.
``refine`` returns ``proportions``, ``profiles``, ``history`` and ``diagnostics``.
For the independent RCC comparison, run this workflow separately on the RNA and
protein AnnData objects; component matching happens after fitting.

Fixed-reference RNA
-------------------

.. code-block:: python

   from spDecon.reference_spatial import fit
   from spDecon.reference import cell_fractions

   x = adata.to_df()
   reference = reference_adata[:, adata.var_names].to_df()
   p, b, history, diagnostics = fit(
       x, reference, i, j, weights, strength=0.,
       max_steps=100000, tolerance=1e-10, verbose=True)
   fractions = cell_fractions(p, reference.sum(axis=1).to_numpy())

``reference``
   ``K × G`` arithmetic mean raw counts per reference cell type. Gene columns
   must match the target in identity and order. The solver row-normalizes this
   table internally; retain the original row sums for cell-fraction conversion.
``i``, ``j``, ``weights``, ``strength``
   Graph endpoints and weights, as returned by ``spatial_graph``. The main RNA
   reference analysis uses ``strength=0``; the graph then contributes no penalty.
``max_steps``, ``tolerance``
   Iteration cap and conditional KKT tolerance. The Xenium tutorial explicitly
   uses ``100000`` and ``1e-10``. The function's general defaults are ``3000`` and
   ``1e-6``; use the explicit tutorial settings to reproduce that analysis.

This CPU function returns ``(p, b, history, diagnostics)``. ``p`` contains
molecular contributions; ``cell_fractions`` divides them by the reference
per-cell yields and renormalizes each spot. Store the two outputs separately:
``adata.obsm["spDecon"] = p`` and
``adata.obsm["spDecon_cell_fractions"] = fractions``.

Histology-guided fitting
------------------------

.. code-block:: python

   from spDecon.histology import fit_histology

   result = fit_histology(
       adata.to_df(), adata.obsm["morphology"], adata.obsm["spatial"], K,
       sections=adata.obs["section_id"].to_numpy(),
       image_weight=None, spatial_weight=None,
       random_state=0, device="cuda", max_iter=20000)

``morphology``
   The ``N × 32`` HIPT/UNI frequency matrix, with two blocks each summing to 0.5.
   ``None`` selects the RNA-only control and sets image reconstruction weight
   to zero.
``image_weight``, ``spatial_weight``
   Coefficients for morphology reconstruction and spatial TV. ``None`` calibrates
   the coefficient from the initial fit. The G2 notebook supplies its recorded
   coefficients for the paired comparison.
``image_gate``
   Defaults to ``True``: image differences reduce smoothing across edges.
   ``False`` keeps coordinate-based smoothing without the image gate.
``initial_proportions``, ``initial_profiles``
   Optional ``N × K`` and ``K × G`` unrefined consensus factors. Supply the same
   pair when comparing RNA-only and RNA + H&E fits. If omitted, a consensus is
   fitted internally.

The result contains ``N × K`` proportions, ``K × G`` RNA profiles and
``result.extra["morphology_profiles"]`` of shape ``K × 32``.

To obtain morphology features, call
``spDecon.image_features.extract_histology(image, spots, ...)``:

* ``image``: path to the registered RGB H&E image.
* ``spots``: spot-indexed DataFrame with ``x,y`` centers in original image pixels.
* ``microns_per_pixel``, ``radius_pixels``: image resolution and capture radius.
* ``encoder_source``: spEnhance checkout containing the HIPT code and checkpoints;
  UNI access and weights are configured separately in the G2 notebook.
* ``output``: new directory for extracted features; ``device="cuda"`` selects GPU
  extraction. The return value is a spot-indexed 32-column DataFrame, which can
  be aligned and assigned to ``adata.obsm["morphology"]``.

Joint multi-sample fitting
--------------------------

.. code-block:: python

   from spDecon.multisample import fit_multisample

   result = fit_multisample(
       adata.to_df(), adata.obs, K,
       group_by=["subject_id", "condition"], subject_column="subject_id",
       coordinates=adata.obsm["spatial"],
       eta_subject=10., eta_condition=20., spatial_weight=.05,
       random_state=0, device="cuda", max_iter=40000)

``metadata``
   One row per spot, in count-matrix order. Include ``section_id`` for physical
   sections and the columns referenced by ``group_by`` and ``subject_column``.
``group_by``, ``subject_column``
   ``group_by`` defines expression-profile groups. A list of subject and condition
   columns gives groups nested within subjects; ``subject_column`` identifies
   the parent subject. Omitting ``subject_column`` fits flat group deviations.
``eta_subject``, ``eta_condition``
   Nonnegative penalties on subject and within-subject condition deviations.
   Larger values shrink the corresponding deviations more strongly.
``edges``, ``coordinates``, ``spatial_weight``
   Supply an ``E × 2`` integer edge array, or coordinates for graph construction.
   Endpoint indices refer to the current spot order. Edges must stay within
   ``section_id``. With coordinates, the function constructs section-specific
   edges internally; ``spatial_weight`` controls their smoothing penalty.

``result.proportions`` is ``N × K``. ``result.profiles`` is
``n_groups × K × G``. The ``extra`` dictionary contains ``shared_profiles``
(``K × G``), ``delta_subject`` (``n_subjects × K × G``), ``delta_condition``
(``n_groups × K × G``), ``subjects`` and ``group_labels``. The shared profiles can
be stored as ``adata.varm["spDecon_profiles"] = result.extra["shared_profiles"].T``.

Joint multi-omics fitting
-------------------------

.. code-block:: python

   import pandas as pd
   from spDecon.multiomics import fit_multiomics

   modalities = {name: assay.to_df() for name, assay in assays.items()}
   # One row per corresponding feature pair; columns match the assay names.
   pairs = pd.read_csv("feature_links.csv")
   links = {name: pairs[name].tolist() for name in assays}
   result = fit_multiomics(
       modalities, n_components=K, links=links,
       coordinates=assays["RNA"].obsm["spatial"],
       sections=assays["RNA"].obs["section_id"].to_numpy(),
       eta=10., spatial_weight=.05, random_state=0, device="cuda")

``modalities``
   Dictionary of ``N × G_assay`` matrices. All assays must have identical spot
   IDs and order. DataFrames preserve those IDs and feature names.
``links``
   Dictionary of equally sized lists of corresponding features. Use column
   names with DataFrames, or zero-based column indices with NumPy arrays.
   For example, each row of ``feature_links.csv`` pairs one RNA feature with one
   protein feature. Features may occur in only one pair per assay.
``eta``
   Coupling penalty for matched-feature profiles, default ``10`` in free-profile
   mode. Larger values encourage closer cross-assay agreement. With ``eta=0``,
   links may be omitted, but the model still shares spot weights.
``assay_weights``
   Optional positive weights in dictionary insertion order. The default gives
   each assay equal total contribution after rescaling measurements.
``references``
   Optional dictionary of fixed reference tables, one per assay. Cell-type
   identities and order must match across references, and features must align
   with each assay. Fixed-reference mode requires ``eta=0`` or ``None`` and
   disables the learned profile-deviation penalty.

The result shares ``N × K`` proportions across assays. ``profiles`` concatenates
all assay features; ``extra["assays"]`` and ``extra["feature_offsets"]`` identify
the blocks. To retain each block in its AnnData object:

.. code-block:: python

   offsets = result.extra["feature_offsets"]
   for m, name in enumerate(result.extra["assays"]):
       assays[name].obsm["spDecon"] = result.proportions
       assays[name].varm["spDecon_profiles"] = result.profiles[:, offsets[m]:offsets[m+1]].T

These profile blocks retain their assay mass in the joint model; their rows do
not individually sum to one. Spatial edges, if supplied, use the common spot
order. Otherwise, ``coordinates`` and ``sections`` define the graph.

Joint optimization controls
---------------------------

``fit_histology``, ``fit_multisample`` and ``fit_multiomics`` accept these keyword
arguments through their shared optimizer:

.. list-table:: Optimization parameters
   :header-rows: 1
   :widths: 28 18 54

   * - Parameter
     - Default
     - Meaning
   * - ``max_iter``
     - ``20000``
     - Maximum optimization iterations; CRC explicitly uses ``40000``.
   * - ``tolerance``
     - ``1e-4``
     - KKT stationarity tolerance.
   * - ``relative_tolerance``
     - ``1e-7``
     - Relative objective-change tolerance; histology defaults to ``1e-6``.
   * - ``check_every``
     - ``25``
     - Convergence-check interval; histology defaults to ``10``.
   * - ``minimum_iterations``
     - ``25``
     - Earliest stopping iteration; histology defaults to ``20``.
   * - ``random_state``
     - ``1``
     - Initialization seed for these three functions.
   * - ``device``
     - ``"auto"``
     - Select GPU when available, otherwise CPU, for joint optimization.
       The reference-free spatial example above uses CUDA for calibration
       and refinement.

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
