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
     --k 11 --device auto --seeds 1 3 5 --deterministic --output runs/example

Use a new output directory. This performs full three-branch initialization,
consensus alignment and KL + TV refinement. The CLI writes factors, settings,
optimization histories and convergence diagnostics. A supplied ``block`` coordinate
column keeps graph construction within sections. ``--spatial-weight 0`` runs the
complete count-only model; omitting it uses the recorded seed-1 calibration rule.

``--device auto`` selects CUDA if available and CPU otherwise; ``--device cpu``
forces CPU throughout initialization, calibration and refinement.

Bulk deconvolution without coordinates
--------------------------------------

Use a raw-count CSV with samples in rows and genes in columns. Coordinates can
be omitted when ``--spatial-weight 0``; no coordinate graph is constructed.

.. code-block:: bash

   python run_spdecon.py core \
     --counts bulk_counts.csv --k 3 --spatial-weight 0 \
     --device cpu --seeds 1 --output runs/bulk_cpu

This runs the same three-branch reference-free workflow and estimates molecular
component contributions. Its components are not automatically named cell types.

For a fixed reference, provide a types × genes CSV with the same ordered gene
panel as the bulk counts:

.. code-block:: bash

   python run_spdecon.py reference \
     --counts bulk_counts.csv --reference reference.csv \
     --device cpu --output runs/bulk_reference_cpu

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
     - Shared profiles plus named flat or nested design effects
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

   from spDecon.config import SolverConfig

   device = "auto"  # or "cpu" / "cuda"
   initial = build_consensus(y, n_components=K, random_state=1,
                             solver_config=SolverConfig(device=device, random_state=1))
   i, j, weights, info = spatial_graph(xy, sections)
   calibration = calibrate(
       y, initial.consensus_proportions, initial.consensus_profiles,
       i, j, weights, device=device)
   result = refine(
       y, initial.consensus_proportions, initial.consensus_profiles,
       i, j, weights, strength=calibration["spatial_weight"], seed=1, device=device)

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
``device``
   ``SolverConfig.device`` selects the device for branch fits. Pass the same
   value to ``calibrate`` and ``refine``. All three default to ``"auto"``.
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

The workflow is registered H&E → HIPT and UNI embeddings → spot-level morphology
matrix ``Q`` → joint deconvolution. ``extract_histology`` runs both encoders and
builds ``Q``; ``fit_histology`` uses ``Q`` with raw RNA counts and coordinates.

Prepare the encoders
~~~~~~~~~~~~~~~~~~~~

The feature-extraction code is included in spDecon. From the repository root,
install the optional image dependencies and prepare a directory for HIPT weights:

.. code-block:: bash

   python -m pip install -r requirements-histology.lock.txt
   mkdir -p checkpoints/hipt
   hf auth login

For CPU-only image dependencies, use the commands in :doc:`installation`.

Download ``vit256_small_dino.pth`` and ``vit4k_xs_dino.pth`` from the
`official HIPT repository <https://github.com/mahmoodlab/HIPT>`_ into
``checkpoints/hipt/``. Request access to
`UNI <https://huggingface.co/MahmoodLab/UNI>`_ with the account used for
``hf auth login``. The adapter uses these checkpoints and the Hugging Face cache;
it does not require a token in the notebook.

Extract embeddings and build Q
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a single section, prepare an AnnData file with raw counts in ``X`` and spatial
graph coordinates in ``obsm["spatial"]``. A separate ``image_spots.csv`` has spot
IDs in its first column and ``x,y`` centers registered to the original RGB H&E
image. Image centers are in original pixels, with x to the right and y downward.
``microns_per_pixel.txt`` and ``radius_pixels.txt`` each contain one number: the
original image resolution and the capture radius in that image's pixels.

.. code-block:: python

   from pathlib import Path
   import anndata as ad
   import pandas as pd
   from spDecon.image_features import extract_histology

   adata = ad.read_h5ad("sample_counts.h5ad")
   image_spots = pd.read_csv("image_spots.csv", index_col=0).loc[adata.obs_names]
   q_frame = extract_histology(
       "he.jpg", image_spots,
       microns_per_pixel=float(Path("microns_per_pixel.txt").read_text()),
       radius_pixels=float(Path("radius_pixels.txt").read_text()),
       checkpoint_dir="checkpoints/hipt",
       output="outputs/sample_features", device="auto")
   adata.obsm["morphology"] = q_frame.loc[adata.obs_names].to_numpy()
   adata.obs["section_id"] = "sample"

Use a new output directory for extraction. The adapter rescales the image and
capture geometry to 0.5 µm/pixel, then writes the HIPT and UNI embedding maps.
For each encoder, it retains 99% of variance by PCA, fits a 16-word dictionary,
and pools word frequencies over each spot's capture footprint. Concatenating
the two frequency blocks with equal weight produces nonnegative ``Q`` with
shape ``N × 32`` and row sums of one. The deconvolution input is this frequency
matrix; the intermediate embeddings and PCA scores can contain signed values.
No RNA counts or pathology labels are used to construct ``Q``.

The output directory contains:

* ``embeddings-hist-vit.pickle`` and ``embeddings-hist-uni.pickle``: encoder maps;
* ``hipt_pca99.npy``, ``uni_pca99.npy`` and the corresponding
  ``*_pca_transform.npz`` and ``*_dictionary.npz`` files;
* ``morphology.npz`` (arrays ``Q`` and ``spots``), ``morphology.csv``, and geometry
  and extraction records in ``provenance.json``.

To reuse the extracted features in another session, load the saved matrix and
align its rows by spot ID before continuing:

.. code-block:: python

   import numpy as np

   with np.load("outputs/sample_features/morphology.npz") as features:
       q_frame = pd.DataFrame(features["Q"], index=features["spots"])
   adata.obsm["morphology"] = q_frame.loc[adata.obs_names].to_numpy()

Fit and save the paired models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continue with the same AnnData object. ``K=6`` is the G2 example's component
count; choose the component count for your own analysis.

.. code-block:: python

   from spDecon.histology import fit_histology
   from spDecon.consensus import build_consensus

   K = 6
   counts = adata.to_df()
   initial = build_consensus(counts.to_numpy(), K, random_state=0)
   shared = dict(
       initial_proportions=initial.consensus_proportions,
       initial_profiles=initial.consensus_profiles,
       sections=adata.obs["section_id"].to_numpy(), random_state=0,
       device="auto", max_iter=20000)
   result = fit_histology(
       counts, adata.obsm["morphology"], adata.obsm["spatial"], K,
       image_weight=None, spatial_weight=None,
       **shared)
   rna_only = fit_histology(
       counts, None, adata.obsm["spatial"], K,
       spatial_weight=result.diagnostics["spatial_weight"], **shared)
   adata.obsm["spDecon_HE"] = result.proportions
   adata.obsm["spDecon_RNA"] = rna_only.proportions
   adata.varm["spDecon_HE_profiles"] = result.profiles.T
   adata.write_h5ad("sample_spdecon.h5ad")

Both arms start from the same unrefined RNA consensus. The H&E arm calibrates
its coefficients with the supplied ``Q``; the RNA-only arm reuses its spatial
coefficient. Replacing the extracted features therefore also recalibrates the
image term. Inspect ``result.diagnostics`` and ``rna_only.diagnostics`` for the
fitting status and final objective.

``morphology``
   The ``N × 32`` HIPT/UNI frequency matrix, with two blocks each summing to 0.5.
   ``None`` selects the RNA-only control and sets image reconstruction weight
   to zero.
``image_weight``, ``spatial_weight``
   Coefficients for morphology reconstruction and spatial TV. ``None`` calibrates
   the coefficient near the unrefined RNA consensus, at ``0.9 P0 + 0.1/K``.
   Calibration uses the same morphology matrix as fitting, exposure-weighted
   image reconstruction and exposure-weighted coordinate TV, with multipliers
   0.30 and 0.05. The two arms share the calibrated spatial coefficient.
   The morphology gate is computed from that same matrix and applied only
   when fitting the H&E arm. Recalibrate when the input features change.
``image_gate``
   Defaults to ``True``: image differences reduce smoothing across edges.
   ``False`` keeps coordinate-based smoothing without the image gate.
``initial_proportions``, ``initial_profiles``
   Optional ``N × K`` and ``K × G`` unrefined consensus factors. Supply the same
   pair when comparing RNA-only and RNA + H&E fits. If omitted, a consensus is
   fitted internally.

The result contains ``N × K`` proportions, ``K × G`` RNA profiles and
``result.extra["morphology_profiles"]`` of shape ``K × 32``.

The `G2 notebook <https://github.com/dsong-lab/spDecon/blob/main/tutorials/03_histology.ipynb>`_
provides the registered image, inputs and executed plots for this workflow.

Joint multi-sample fitting
--------------------------

Combine samples measured on a common feature panel. Define deviations through
``effects``: each key names an effect, ``column`` selects its categorical labels
in ``adata.obs``, and ``penalty`` controls shrinkage toward its parent profile.
The names are chosen by the user; patient and condition are labels used in the
CRC example, not required field names.

**One grouping level**, for example sample-specific deviations:

.. code-block:: python

   effects = {
       "sample": {"column": "sample_id", "penalty": 10.},
   }

**Two nested levels**, for example timepoints within each donor:

.. code-block:: python

   from spDecon.multisample import fit_multisample

   effects = {
       "donor": {"column": "donor_id", "penalty": 10.},
       "timepoint": {"column": "visit", "parent": "donor", "penalty": 20.},
   }
   result = fit_multisample(
       adata.to_df(), adata.obs, K, effects=effects,
       coordinates=adata.obsm["spatial"], spatial_weight=.05,
       random_state=0, device="auto", max_iter=40000)

``metadata``
   One row per spot, in count-matrix order. Include ``section_id`` for physical
   sections and the columns referenced in ``effects``. Each section must belong
   to one profile group. Several sections may share a group while keeping their
   spatial graphs separate.
``effects``
   A dictionary defining one flat effect or a root effect and one nested child.
   Effect names start with a letter and contain letters, digits or underscores.
   Each entry requires ``column`` (a nonempty metadata column name) and
   ``penalty`` (a finite nonnegative number). The optional ``parent`` names
   another effect key, not a metadata column. Dictionary order does not determine
   the hierarchy. The two effects must use different columns.
``penalty``
   Larger values shrink the corresponding profile deviations more strongly;
   zero removes that penalty. The model retains its feature/component scaling
   and equal-parent weighting. Values 10 and 20 reproduce the CRC example's
   settings; they are not selected automatically for other datasets.
``parent``
   Child levels are identified by the combination of parent and child labels.
   Thus ``early`` in donor D1 and ``early`` in donor D2 have separate deviations.
   Child deviations average to zero within each parent. A parent with only one
   observed child has a zero child deviation. This interface supports one or two
   nested levels; crossed effects, continuous covariates and deeper hierarchies
   are not implemented.
``edges``, ``coordinates``, ``spatial_weight``
   Supply an ``E × 2`` integer edge array, or coordinates for graph construction.
   Endpoint indices refer to the current spot order. Edges must stay within
   ``section_id``. With coordinates, the function constructs section-specific
   edges internally; ``spatial_weight`` controls their smoothing penalty.

The same interface describes donor/visit, batch/sample or patient/condition
nesting when that hierarchy matches the experiment. For the CRC dataset:

.. code-block:: python

   effects = {
       "patient": {"column": "subject_id", "penalty": 10.},
       "condition": {"column": "condition", "parent": "patient", "penalty": 20.},
   }

See :doc:`notebooks/05_multisample` for the complete CRC example. For paired assays
with different feature panels, use ``fit_multiomics`` and feature ``links`` as
shown below; it couples profiles on matched features and shares spot weights.

Outputs and effect labels
~~~~~~~~~~~~~~~~~~~~~~~~~

``result.proportions`` is ``N × K`` and ``result.profiles`` is
``n_groups × K × G``. ``result.extra["shared_profiles"]`` is ``K × G``.
For each effect name, the ``extra`` dictionary includes:

* ``effect_delta_<name>``: deviations of shape ``n_levels × K × G``.
* ``effect_labels_<name>`` and ``effect_columns_<name>``: the metadata labels and
  their column names in deviation-row order. Nested levels include both labels.
* ``effect_index_<name>``: one deviation-row index per input spot.

``effect_names`` lists effects from root to child. Read all deviations without
using any dataset-specific names:

.. code-block:: python

   import pandas as pd

   deviations = {
       name: result.extra[f"effect_delta_{name}"]
       for name in result.extra["effect_names"]
   }
   for name, delta in deviations.items():
       levels = pd.DataFrame(
           result.extra[f"effect_labels_{name}"],
           columns=result.extra[f"effect_columns_{name}"])
       print(name, delta.shape)
       print(levels)

``result.diagnostics["effects"]`` records the resolved design, and
``result.diagnostics["effect_penalties"]`` reports each weighted penalty value.
Shared profiles can be stored as
``adata.varm["spDecon_profiles"] = result.extra["shared_profiles"].T``.

Command line
~~~~~~~~~~~~

Save the same specification as ``effects.json``. JSON numbers use ``10.0``
instead of Python's ``10.``:

.. code-block:: json

   {
     "donor": {"column": "donor_id", "penalty": 10.0},
     "timepoint": {"column": "visit", "parent": "donor", "penalty": 20.0}
   }

.. code-block:: bash

   python run_spdecon.py multisample \
     --counts counts.csv --metadata metadata.csv --effects effects.json \
     --k 10 --device auto --output runs/joint

The metadata CSV contains spot IDs, ``section_id``, ``x``, ``y`` and the design
columns. Coordinates can be omitted with explicit ``--edges`` or
``--spatial-weight 0``. In addition to ``P.csv`` and group profile tables, the
CLI saves named deviations in ``fit.npz`` and the corresponding labels in
``effect_<name>_levels.csv``. Arrays can be loaded with ``allow_pickle=False``.

Compatibility
~~~~~~~~~~~~~

Existing calls with ``group_by``, ``subject_column``, ``eta_subject`` and
``eta_condition`` retain their behavior. Use either that interface or ``effects``
in a call. The CRC effects specification above gives the same objective and
updates as ``group_by=["subject_id", "condition"]``,
``subject_column="subject_id"``, ``eta_subject=10.`` and ``eta_condition=20.``.
The existing ``delta_subject`` and ``delta_condition`` output aliases remain
available; new code can use the named outputs described above.

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
       eta=10., spatial_weight=.05, random_state=0, device="auto")

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
       The reference-free workflow also supports both devices throughout
       initialization, calibration and refinement.

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

The notebooks fit user-provided inputs and display the resulting diagnostics.
Set ``VERBOSE=True`` for live optimization output.
