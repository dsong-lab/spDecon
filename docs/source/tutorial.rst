Tutorials
=========

Choose a workflow below to run with your own AnnData files. Each notebook covers
loading, fitting, plotting and export, with direct data links and inline example
figures. Start with :doc:`quickstart` for a small generated example, or use
:doc:`index` to choose an analysis.

.. raw:: html

   <span id="anndata-input"></span>
   <span id="prepare-input-data-with-anndata"></span>
   <span id="joint-design-tutorial"></span>
   <span id="choose-a-multi-sample-design"></span>

Input formats and design preparation are now in :doc:`prepare_data`, including
:ref:`AnnData fields <anndata-input>` and :ref:`multi-sample designs
<joint-design-tutorial>`. Use :doc:`api` for parameters, defaults and return values.

.. list-table:: Workflows
   :header-rows: 1
   :widths: 24 43 33

   * - Workflow
     - Read guide
     - Run notebook
   * - Reference-free
     - :doc:`Learn profiles and spatial weights <notebooks/01_reference_free>`
     - `01_reference_free.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/01_reference_free.ipynb>`_
   * - Reference-based
     - :doc:`Fixed signatures and cell fractions <notebooks/02_reference_based>`
     - `02_reference_based.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/02_reference_based.ipynb>`_
   * - Histology
     - :doc:`HIPT/UNI extraction and paired deconvolution <notebooks/03_histology>`
     - `03_histology.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/03_histology.ipynb>`_
   * - Cross-modality
     - :doc:`Independent RNA/protein structures <notebooks/04_cross_modality>`
     - `04_cross_modality.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/04_cross_modality.ipynb>`_
   * - Joint multi-sample
     - :doc:`Named effects across sections <notebooks/05_multisample>`
     - `05_multisample.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/05_multisample.ipynb>`_
   * - Joint multi-omics
     - :doc:`Shared RNA/protein spatial weights <notebooks/06_joint_multiomics>`
     - `06_joint_multiomics.ipynb <https://github.com/dsong-lab/spDecon/blob/main/tutorials/06_joint_multiomics.ipynb>`_

Run a notebook with your data
-----------------------------

1. Follow :doc:`installation` and open a notebook with that Python environment.
2. Set ``DATA_DIR`` and input filenames, then choose the relevant design columns
   or feature links. Choose ``K`` for workflows that learn components; in the
   reference-based tutorial (02), the number of reference cell types determines
   ``K``. Run the loading cells to inspect the input.
3. Run the fitting cells. Where ``DEVICE`` is available, choose ``"cpu"`` or
   ``"cuda"``, or keep ``"auto"``. Tutorial 02 uses its CPU solver, with
   ``MAX_STEPS`` controlling the optimization limit. Inspect the returned
   convergence diagnostics.
4. Run the plotting and export cells. All generated tables, figures and fitted
   weights are written under ``outputs/``.

Fraction and pathology evaluations are optional and run only when you provide
those annotations. These labels are not used for model fitting; component
matching for evaluation is performed after fitting. Each notebook displays archived
example figures and generates new results for the supplied inputs. Archived fit
diagnostics and evaluation tables are available in the repository's
`reproduction record <https://github.com/dsong-lab/spDecon/blob/main/reproduction/README.md>`_.

.. toctree::
   :hidden:
   :maxdepth: 1

   01 Reference-free <notebooks/01_reference_free>
   02 Reference-based <notebooks/02_reference_based>
   03 Histology-guided <notebooks/03_histology>
   04 Cross-modality <notebooks/04_cross_modality>
   05 Joint multi-sample <notebooks/05_multisample>
   06 Joint multi-omics <notebooks/06_joint_multiomics>
