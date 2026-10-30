Data and provenance
===================

The `tutorial data release <https://github.com/dsong-lab/spDecon/releases/tag/tutorial-data-v0.1.0>`_
contains the complete processed matrices used by the six notebooks, fitted arrays,
spot/feature identifiers, evaluation annotations and recorded settings. The
notebooks download it automatically. Raw sequencing reads and pretrained encoder
weights are outside this bundle.

.. list-table:: Included examples
   :header-rows: 1
   :widths: 18 38 44

   * - Example
     - Fitting data
     - Source measurements
   * - Cerebellum
     - 1,000 pseudo-spots × 1,000 genes; 11 types
     - `Slide-seq SCP354 <https://singlecell.broadinstitute.org/single_cell/study/SCP354/slide-seq-study>`_; prepared cerebellum subset from the `RETROFIT benchmark <https://github.com/qunhualilab/retrofit-scripts>`_ (`archived data and scripts <https://doi.org/10.5281/zenodo.19241634>`_)
   * - Xenium lymph node
     - 256 pseudo-spots × 1,027 genes; 12 reference types
     - `Xenium Prime lymph node <https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression>`_; `Flex 320k scFFPE <https://www.10xgenomics.com/datasets/320k_scFFPE_16-plex_GEM-X_FLEX>`_ library LNReactive_BC9-10 for the reference
   * - HER2ST G2
     - 467 spots × 14,898 genes; 32 morphology bins
     - `Processed HER2ST deposit <https://zenodo.org/records/4751624>`_, including registered H&E and pathological annotations
   * - RCC
     - 1,595 matched spots; 405 RNA genes and 27 proteins
     - `Xenium Protein FFPE Human Renal Carcinoma <https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma>`_
   * - CRC
     - 102,544 spots in 12 sections; 480 genes
     - `GSE294385 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294385>`_; pathology from `CRC_micromets_ST <https://github.com/yliuup/CRC_micromets_ST>`_
   * - Tonsil
     - 4,908 matched spots; 2,000 RNA genes and 35 proteins
     - `Visium CytAssist Tonsil Gene and Protein Expression <https://www.10xgenomics.com/datasets/visium-cytassist-gene-and-protein-expression-library-of-human-tonsil-with-add-on-antibodies-h-e-6-5-mm-ffpe-2-standard>`_; Space Ranger 2.1.0 outputs

Arrays and interpretation
-------------------------

``data.npz`` stores counts or prepared nonnegative assay measurements, coordinates,
and explicit spot/feature names. Fitted files store ``P`` (spots × components)
and ``B`` (components × features). CRC stores effective condition profiles and
the shared/subject/condition decomposition separately. No example is reduced to
an artificial small subset for notebook execution.

Cerebellum coordinates are an artificial evaluation grid. Xenium evaluates seven
conservatively mapped groups on 255 common spots while retaining all 12 reference
types in the fraction denominator. G2 fits 467 spots and evaluates 402 labelled
spots. Tonsil pathology contains 391 GC and 4,515 other labelled spots. CRC retains
the complete 480-gene, 12-section analysis, including low-count regions.

Reference and image provenance
------------------------------

The Xenium reference uses raw mean counts on the frozen COSG top-100-per-type
union. Its local cell-group annotation and unresolved donor relationship limit
cross-donor interpretation. These details do not change the supplied fitted-panel
counts or reference yields.

The G2 morphology matrix was regenerated from its H&E image using
`spEnhance encoder utilities <https://github.com/dsong-lab/spEnhance>`_,
`HIPT <https://github.com/mahmoodlab/HIPT>`_ and
`UNI <https://huggingface.co/MahmoodLab/UNI>`_. Each user obtains encoder weights
under the respective model terms. Derived image features are for research use
under those applicable terms; the repository does not grant new rights to source
datasets or third-party models.


HER2ST original downloads
-------------------------

The public G2 deposit provides `count matrices <https://zenodo.org/records/4751624/files/count-matrices.zip?download=1>`_,
`H&E images <https://zenodo.org/records/4751624/files/images.zip?download=1>`_,
`metadata <https://zenodo.org/records/4751624/files/meta.zip?download=1>`_ and
`spot annotations <https://zenodo.org/records/4751624/files/spot-selections.zip?download=1>`_.
Follow the extraction instructions on the deposit page and select G2. The
tutorial release collects the registered G2 inputs in one folder and supplies
the derived HIPT/UNI morphology matrix; the notebook also shows how to regenerate
that matrix from the image.

Tonsil display coordinates
--------------------------

The Tonsil bundle records Visium array rows and columns. Its notebook converts
them to a regular hexagonal lattice before plotting, with equal axis aspect.
Only joint-model weights are displayed. Each component has a labelled colorbar
ending at its own 99th percentile; the complete weights remain unchanged in the
exported table.
