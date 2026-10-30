Tutorials
=========

Each notebook explains the model, loads complete example data, provides the full
fitting call, and computes figures from the supplied fitted arrays. Read the
rendered guide here or download the notebook to run locally.

.. list-table:: Worked examples
   :header-rows: 1
   :widths: 24 43 33

   * - Workflow
     - Read guide and outputs
     - Run notebook
   * - Reference-free
     - :doc:`Cerebellum: correlations and spatial pies <notebooks/01_reference_free>`
     - `01_reference_free.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/01_reference_free.ipynb>`_
   * - Reference-based
     - :doc:`Xenium lymph node: fixed signatures and cell fractions <notebooks/02_reference_based>`
     - `02_reference_based.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/02_reference_based.ipynb>`_
   * - Histology
     - :doc:`G2: HIPT/UNI extraction and paired improvement <notebooks/03_histology>`
     - `03_histology.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/03_histology.ipynb>`_
   * - Cross-modality
     - :doc:`RCC: independent RNA/protein structures <notebooks/04_cross_modality>`
     - `04_cross_modality.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/04_cross_modality.ipynb>`_
   * - Joint multi-sample
     - :doc:`CRC: 480 genes, 12 sections and nested profiles <notebooks/05_multisample>`
     - `05_multisample.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/05_multisample.ipynb>`_
   * - Joint multi-omics
     - :doc:`Tonsil: paired RNA/protein spatial structure <notebooks/06_joint_multiomics>`
     - `06_joint_multiomics.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/06_joint_multiomics.ipynb>`_

How to use the examples
-----------------------

1. Follow :doc:`installation` and open a notebook with that Python environment.
2. Run the setup and loading cells. The versioned data bundle is downloaded once.
3. Keep ``RUN_FIT=False`` to inspect the complete saved fit, or set it to ``True``
   to retrain using the full dataset and recorded optimization limits.
4. Run the metric and plot cells. Tables and fitted-weight exports go to
   ``outputs/``; PNG figures are also written to ``docs/source/_static/figures``.

The notebooks never use evaluation labels for model fitting. Matching components
to known labels is explicitly performed after fitting. The CRC notebook uses the
archived full 12-section result; it does not report a new full retraining.

.. image:: _static/figures/cerebellum_pies.png
   :alt: Known and inferred Cerebellum pseudo-spot compositions, shown as spatial pies.

.. image:: _static/figures/rcc_structures.png
   :alt: Nine independently fitted RNA components matched to nine protein components in RCC.
