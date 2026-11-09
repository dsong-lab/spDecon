Prepare your data
=================

Prepare counts, identifiers and metadata before fitting. For a first run without
your own files, use :doc:`quickstart`. The six :doc:`tutorial` notebooks show the
complete workflow-specific loading and fitting code.

.. _anndata-input:

AnnData fields
--------------

Use one ``AnnData`` object per assay, with **spatial locations in rows and
features in columns**. A location may be a measured capture spot or a constructed
pseudo-spot. Keep unique spot and feature IDs. The conventions below apply to all
six notebooks; ``N`` denotes locations and ``G`` denotes features.

.. list-table:: Input fields
   :header-rows: 1
   :widths: 32 18 50

   * - Field
     - Shape
     - Contents
   * - ``adata.X``
     - ``N × G``
     - Raw counts, or the explicitly prepared nonnegative assay measurements
       described in the notebook. Each location needs positive total signal.
   * - ``adata.obs_names`` / ``adata.var_names``
     - ``N`` / ``G``
     - Spot IDs / gene or protein names, in matrix order.
   * - ``adata.obsm["spatial"]``
     - ``N × 2``
     - Spatial coordinates used to construct the graph, in a common coordinate
       system within each section.
   * - ``adata.obs["section_id"]``
     - ``N``
     - Physical section IDs; graph edges stay within sections. A single section
       can use the same ID for every spot.
   * - Design columns in ``adata.obs``
     - ``N`` each
     - Categorical labels selected by ``effects`` for joint samples,
       such as sample, donor or visit. The CRC study example uses subject_id
       and condition.
   * - ``adata.obsm["morphology"]``
     - ``N × 32``
     - Histology only: HIPT and UNI visual-word frequencies. Each 16-column
       block sums to 0.5. These are derived features, not the raw embeddings.

**Read an existing file:**

.. code-block:: python

   import anndata as ad
   import numpy as np

   adata = ad.read_h5ad("sample.h5ad")
   counts = adata.to_df()  # labeled spots × features, from adata.X
   y = np.array(counts, dtype=float, order="F")
   xy = adata.obsm["spatial"]

If ``X`` contains normalized or log-transformed RNA, use a stored raw-count
layer instead: ``counts = adata.to_df(layer="counts")``. Do not pass log expression
to the fitting functions. ``to_df`` also converts a sparse matrix to a dense
DataFrame, preserving its spot and feature order; fitting requires memory for
the dense matrix. See the `AnnData conversion documentation
<https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.to_df.html>`_.

**Create an object from CSV files:** ``counts.csv`` has spot IDs in its first
column and genes in the remaining columns; ``coordinates.csv`` has spot IDs
and ``x,y`` columns.

.. code-block:: python

   import pandas as pd

   counts = pd.read_csv("counts.csv", index_col=0)
   coordinates = pd.read_csv("coordinates.csv", index_col=0).loc[counts.index]
   adata = ad.AnnData(counts)
   adata.obsm["spatial"] = coordinates[["x", "y"]].to_numpy()
   adata.obs["section_id"] = "section_1"
   adata.write_h5ad("sample.h5ad")

Reference profiles
------------------

Start from a cells × genes single-cell AnnData with raw counts and cell-type
labels in ``obs``. The :doc:`reference-based tutorial
<notebooks/02_reference_based>` builds per-type profiles, selects the COSG
top50 marker union, aligns genes to the target and saves ``reference.h5ad``.
It uses the study's cell-balanced profiles by default; set
``BALANCE_REFERENCE=False`` for arithmetic raw-count means. Both retain
unnormalized per-cell signals for cell-fraction conversion. An existing
types × genes reference can be loaded with ``BUILD_REFERENCE=False``.

Registered images and morphology
--------------------------------

Provide a registered H&E image and spot centers in original image
pixels to ``extract_histology``. Store its aligned output in
``adata.obsm["morphology"]``. Image-pixel coordinates can be stored separately
in ``adata.obsm["spatial_image"]``; do not substitute array-grid coordinates
when extracting image patches. Also supply the image resolution
(``microns_per_pixel``) and capture radius in original pixels (``radius_pixels``).
The histology notebook includes both encoder steps.

Multiple sections
-----------------

Concatenate sections along the observation axis, using a common gene panel and
unique spot IDs. Store ``section_id`` and the chosen
design labels in ``adata.obs``; pass that table as ``metadata`` and specify
``effects``. One grouping level gives flat deviations; a parent and child
give two nested levels. Patient/condition is the CRC study example's configuration.
See :ref:`joint-design-tutorial`.

Paired assays and feature links
-------------------------------

Keep a dictionary such as
``assays = {"RNA": rna_adata, "protein": protein_adata}``. Objects share
identical spot IDs in identical order but may have different feature panels.
Align protein rows with ``protein_adata = protein_adata[rna_adata.obs_names].copy()``
before constructing the dictionary. ``links`` identifies corresponding
RNA/protein features; unmatched features remain in the model.

.. _joint-design-tutorial:

Choose a multi-sample design
----------------------------

Use the metadata to define the profile variation you want to model. For
sample-specific deviations, one effect is sufficient:

.. code-block:: python

   effects = {"sample": {"column": "sample_id", "penalty": 10.}}

For visits nested within donors:

.. code-block:: python

   effects = {
       "donor": {"column": "donor_id", "penalty": 10.},
       "visit": {"column": "timepoint", "parent": "donor", "penalty": 20.},
   }

Pass this dictionary to ``fit_multisample(..., effects=effects)``. Change the
names and columns to match your metadata. ``parent`` refers to an effect key;
the same visit label in different donors has a separate within-donor deviation.
The current interface supports one flat effect or two nested effects. The
:doc:`multi-sample notebook <notebooks/05_multisample>` shows how to fit your own
sections with these design labels. See :doc:`api` for output labels,
centering constraints and the JSON command-line interface.

Next, open the corresponding workflow in :doc:`tutorial`. Function parameters
and output formats are documented in :doc:`api`.
