Study results / Reproducibility
================================

Find study data sources, analysis settings and recorded results here. The
:doc:`tutorial` notebooks fit user-provided inputs and display archived example
figures. The :doc:`quickstart` dataset is synthetic. Neither is a download of
the complete study inputs and saved fits.

Study data and example figures
------------------------------

.. list-table:: Sources for the six illustrated workflows
   :header-rows: 1
   :widths: 20 52 28

   * - Study example
     - Data sources
     - Figures and workflow
   * - Cerebellum
     - `SCP354 / Slide-seq <https://singlecell.broadinstitute.org/single_cell/study/SCP354/slide-seq-study>`_;
       cell counts and labels from `RETROFIT <https://github.com/qunhualilab/retrofit-scripts>`_.
     - :doc:`Reference-free <notebooks/01_reference_free>`
   * - Xenium lymph node
     - `Xenium target <https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression>`_;
       `Flex reference counts <https://www.10xgenomics.com/datasets/320k_scFFPE_16-plex_GEM-X_FLEX>`_,
       sample ``LNReactive_BC9-10``. The study's cell-group labels were locally annotated.
     - :doc:`Reference-based <notebooks/02_reference_based>`
   * - HER2ST G2
     - Counts, registered H&E, coordinates and region annotations:
       `Zenodo 4751624 <https://doi.org/10.5281/zenodo.4751624>`_.
     - :doc:`Histology-guided <notebooks/03_histology>`
   * - RCC
     - Paired RNA/protein measurements:
       `10x Xenium renal carcinoma <https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma>`_.
     - :doc:`Independent assays <notebooks/04_cross_modality>`
   * - CRC liver sections
     - `GSE294385 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294385>`_;
       annotations in the ``Meta_data`` directory of
       `CRC_micromets_ST <https://github.com/yliuup/CRC_micromets_ST>`_.
     - :doc:`Joint multi-sample <notebooks/05_multisample>`
   * - Tonsil
     - `10x Visium CytAssist RNA/protein <https://www.10xgenomics.com/datasets/visium-cytassist-gene-and-protein-expression-library-of-human-tonsil-with-add-on-antibodies-h-e-6-5-mm-ffpe-2-standard>`_;
       optional pathology annotations from `DGAT <https://github.com/osmanbeyoglulab/DGAT>`_
       (``Tonsil_AddOns_ADT.h5ad``, ``obs["pathology"]``).
     - :doc:`Joint multi-omics <notebooks/06_joint_multiomics>`

Source downloads may require preprocessing and annotation before they match a
notebook's AnnData inputs. Follow :doc:`prepare_data` and the workflow-specific
loading cells. HIPT/UNI weights are obtained separately under their original
terms; see :doc:`installation`.

Study settings and tutorial settings
------------------------------------

Tutorial defaults are editable examples for new inputs. Match the study's
feature panel, reference construction, grouping, solver and evaluation support
when comparing against archived results.

.. list-table:: Settings and provenance to check
   :header-rows: 1
   :widths: 22 78

   * - Analysis
     - Study setting or interpretation
   * - Reference-free RNA
     - The cerebellum example uses 11 components, three-branch consensus and
       calibrated spatial TV. Its pseudo-spot display grid is artificial.
   * - Reference-based RNA
     - The updated workflow uses COSG top50 markers per type (``mu=1``),
       cell-balanced reference profiles, CPU fitting, spatial weight zero,
       ``max_steps=100000`` and KKT tolerance ``1e-10``. The restored Xenium
       tutorial figures and corresponding baseline records use the earlier
       top100 panel; those Xenium results are not top50 results.
   * - Histology
     - The two G2 arms share an unrefined RNA consensus. Calibration and fitting
       use the same HIPT/UNI features; RNA-only reuses the calibrated coordinate
       TV coefficient. The displayed paired comparison uses 402 annotated spots.
   * - Independent RNA/protein
     - RCC fits nine components separately in each assay, then matches spatial
       weights for display. Neither assay provides cell-fraction truth.
   * - Joint multi-sample
     - The archived CRC fit uses 12 sections, 480 genes, ten components and
       patient/within-patient condition effects. The public notebook starts
       from a generic flat sample design. The archived full CRC fit was not
       retrained in the reproduction campaign.
   * - Joint multi-omics
     - The tonsil example uses eleven shared components, 20 linked features and
       coupling coefficient 10. Its maps have labelled per-component
       99th-percentile display limits.

See :doc:`model` for the objectives and :doc:`api` for defaults and convergence
controls. Pinned environments and seeds support repeatable analyses;
cross-device or cross-version bitwise equality is not assumed.

Recorded results and checks
---------------------------

The repository's
`reproduction record <https://github.com/dsong-lab/spDecon/blob/main/reproduction/README.md>`_
describes the 30 October 2026 baseline campaign and the later histology update.
These files preserve their original scope and provenance; they do not certify
every subsequent manuscript revision or represent newly executed tutorials.

.. list-table:: Download recorded results
   :header-rows: 1
   :widths: 44 56

   * - Material
     - What it records
   * - :download:`training_summary.csv <../../reproduction/training_summary.csv>`
     - Status, iterations, objective and recorded runtime of archived tutorial fits.
   * - :download:`rna_metrics.csv <../../reproduction/rna_metrics.csv>`
     - Reference-free RNA scores by dataset and seed.
   * - :download:`reference_rna_metrics.csv <../../reproduction/reference_rna_metrics.csv>`
     - Baseline fixed-reference scores on frozen evaluation supports.
   * - :download:`histology_uniform_metrics.csv <../../reproduction/histology_uniform_metrics.csv>`
     - Updated MOB, G2 and H1 scores; supersedes the baseline histology values.
   * - :download:`rcc_metrics.csv <../../reproduction/rcc_metrics.csv>`
     - Archived, rerun and deterministic replay comparisons.
   * - :download:`tonsil_gc_metrics.csv <../../reproduction/tonsil_gc_metrics.csv>`
     - Profile-aligned pathology readouts for RNA, protein and joint fits.
   * - :download:`tutorial_metrics.csv <../../reproduction/tutorial_metrics.csv>`
     - Recorded example scores and display summaries, including G2 and tonsil.
   * - :download:`crc_full_data_model_check.json <../../reproduction/crc_full_data_model_check.json>`
     - Full-data objective and auxiliary-update check; ``full_retraining`` is false.
   * - :download:`manuscript_number_checks.csv <../../reproduction/manuscript_number_checks.csv>`
     - Dated comparison of printed values with archived sources and reruns.
   * - :download:`figure_contract.md <../../reproduction/figure_contract.md>`
     - Figure layouts, interpretation limits and source conventions.

Run local workflow checks
-------------------------

After installation, run these from the repository root:

.. code-block:: bash

   python tests/check_cpu_workflow.py
   python tests/check_user_data_tutorials.py
   python tests/check_histology.py

These checks exercise the implementation using synthetic or temporary inputs.
The six-notebook check supplies precomputed morphology and caps joint fits for
an interface check; it does not run image encoders or reproduce the full study.
Use the source links and recorded settings above to plan a study-data rerun.
