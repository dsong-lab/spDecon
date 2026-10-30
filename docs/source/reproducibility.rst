Reproducibility and manuscript consistency
==========================================

The release keeps archived manuscript results separate from newly fitted results.
The six notebook outputs were executed using the complete supplied arrays. The
notebook default loads saved fits, recomputes its metrics and redraws figures;
it does not imply that all training was repeated during website construction.

Completed reproduction campaign
--------------------------------

The preceding campaign completed **262 planned final fits**, all with converged
final solutions, and independently recomputed all 262 objectives. It includes
67 primary fits, 48 controls, 84 random-initialization candidates, 30 final-environment
deterministic fits and 33 RCC rank-selection fits. Counts exclude internal branch
and calibration work. One initialization branch in RCC protein K=2 reached its
cap; its final consensus refinement converged. Competitor retraining and the
12-section CRC/DLPFC joint runs were excluded.

The 89 checked printed numerical claims agree with their archived result sources
at the manuscript's stated precision. This does not mean every rerun equals every
printed value, or that every statement in subsequent manuscript revisions has
been validated. Machine-readable comparisons are provided in the repository's
``reproduction/`` directory.

.. list-table:: Selected complete-data comparisons
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - Archived
     - Full reproduction
   * - Cerebellum seed-1 mean type PCC
     - 0.949762
     - 0.949992
   * - Xenium lymph-node mean group PCC
     - 0.818086
     - 0.818086
   * - G2 RNA + H&E region PCC
     - 0.541292
     - 0.541291
   * - G2 RNA + H&E ARI
     - 0.381600
     - 0.381600
   * - RCC held-out mean PCC
     - 0.699004
     - 0.689139
   * - RCC deterministic solver replay PCC
     - 0.699004
     - 0.695808

Seeds and numerical environment
-------------------------------

``requirements.lock.txt`` pins the numerical environment, and
``requirements-histology.lock.txt`` additionally pins the encoders' dependencies.
The explicit PyNNDescent, Numba and llvmlite pins replace an implicit path override
in the earlier research environment. The release uses clean environment imports.

.. code-block:: python

   from spDecon.reproducibility import set_seed
   set_seed(0, deterministic=True)  # Call before fitting or CUDA initialization.

This sets the random generators, deterministic PyTorch execution and cuBLAS
workspace configuration. Repeated complete Cerebellum fits with this setting
returned identical arrays in the recorded environment. It does not promise
identical floating-point trajectories across hardware or software versions.
RCC already used seed 0; the observed historical difference was not caused by a
missing seed. The supplied historical-mode reproduction is retained without
choosing whichever result is closest to the paper.

Model-to-manuscript details
---------------------------

* Main fixed-reference RNA uses **100,000 steps / KKT 1e-10**. The reviewed
  manuscript draft still states 3,000 / 1e-6 in that paragraph; the independent
  STARmap sensitivity does use 3,000 / 1e-6. The website and tutorial show the
  actual main-analysis parameters.
* The code uses row-oriented ``B``; the paper uses the transpose ``C``. The
  multi-omics comparable-profile scaling is likewise an equivalent convention.
* G2 quantitative improvements use the same 402-spot support as the paper.
  The G2 score plot is an analogue of the Figure 3B presentation, whose original
  dataset is Normal1; it is not relabelled as that exact panel.
* CRC's tutorial uses the actual **480-gene** fit from the 500-gene candidate
  panel. Its full 12-section training remains archived. The released objective
  and five-cycle profile update were compared on all 102,544 spots against the
  saved analysis; a complete new optimizer trajectory has not been claimed.

Validation in this release
--------------------------

All six notebooks execute through their complete-data evaluation and plotting
cells. The documentation build treats warnings as errors. Figure review checks
labels, scales, sample counts and result tables. The website build renders saved
outputs and does not launch training jobs.
