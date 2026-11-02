spDecon
========

Spatial molecular deconvolution across RNA, histology, samples and modalities.

.. container:: hero

   **Learn tissue components, estimate reference-defined mixtures, and compare
   spatial structure across measurements.**

   spDecon uses a common Poisson/KL factorization with simplex-constrained weights
   and profiles. Spatial regularization and task-specific terms extend the same
   model to histology, multiple samples and paired assays.

.. figure:: _static/figures/Figure_1_decon.png
   :alt: Overview of spDecon: reference-free, reference-based, histology-guided, multi-sample and multi-omics deconvolution.
   :width: 100%
   :target: _static/figures/Figure_1_decon.png

   Overview of spDecon. Click the figure to view it at full resolution.

Start with :doc:`installation`, then choose a workflow in :doc:`tutorial`.
Each tutorial specifies the input format and provides code to fit, inspect and
export results from your own data.

.. list-table:: Choose an analysis
   :header-rows: 1
   :widths: 25 43 32

   * - Analysis
     - What it estimates
     - Tutorial
   * - Reference-free
     - Learned component weights and molecular profiles
     - :doc:`Learn components <notebooks/01_reference_free>`
   * - Reference-based
     - Weights for fixed, named reference signatures
     - :doc:`Use a fixed reference <notebooks/02_reference_based>`
   * - Histology-guided
     - Shared RNA/morphology mixtures with image-aware smoothing
     - :doc:`Extract and integrate morphology <notebooks/03_histology>`
   * - Cross-modality comparison
     - Independently inferred RNA and protein structures
     - :doc:`Compare paired assays <notebooks/04_cross_modality>`
   * - Joint multi-sample
     - Shared profiles and user-defined flat or nested effects
     - :doc:`Choose named design effects <notebooks/05_multisample>`
   * - Joint multi-omics
     - Shared spot weights across paired assays
     - :doc:`Fit paired assays jointly <notebooks/06_joint_multiomics>`

Source code and notebooks:
`dsong-lab/spDecon <https://github.com/dsong-lab/spDecon>`_.
spDecon is released under the
`MIT License <https://github.com/dsong-lab/spDecon/blob/main/LICENSE>`_.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   installation
   model
   tutorial
   api

.. toctree::
   :maxdepth: 1
   :caption: Workflow notebooks

   notebooks/01_reference_free
   notebooks/02_reference_based
   notebooks/03_histology
   notebooks/04_cross_modality
   notebooks/05_multisample
   notebooks/06_joint_multiomics
