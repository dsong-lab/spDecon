# Complete-data reproduction record

Prepared 30 October 2026.

The preceding reproduction campaign completed 262 planned final fits, all with
converged final solutions, and independently recalculated their objective values.
This count excludes internal initialization/calibration fits and diagnostic repeats.
It comprises 67 primary fits, 48 controls, 84 random-initialization candidates,
30 deterministic final-environment fits and 33 RCC rank-selection fits. One
internal RCC K=2 protein initialization branch reached its cap; its final consensus
refinement converged. Competitors and the 12-section CRC/DLPFC models were not
retrained in that campaign.

The tables below retain archived and rerun values separately. They are evidence
for the reported analyses, not a claim of bitwise equality or a review of every
sentence in every later manuscript revision.

- `rna_metrics.csv`: reference-free RNA scores by dataset and seed.
- `reference_rna_metrics.csv`: fixed-reference RNA scores on frozen supports.
- `histology_metrics.csv`: full-support and frozen-common-support histology scores.
- `rcc_metrics.csv`: ordinary rerun, archive and deterministic replay comparisons.
- `tonsil_gc_metrics.csv`: profile-aligned pathology readouts for three complete fits.
- `manuscript_number_checks.csv`: 89 printed claims compared with archived numerical
  sources; the `evidence` column names source tables in the earlier reproduction
  workspace. This is a campaign record, not a new complete manuscript audit.
- `crc_full_data_model_check.json`: the published hierarchical objective and
  five-cycle auxiliary update evaluated on the complete 102,544-spot, 480-gene
  CRC data. The full optimizer was not retrained.
- `tutorial_metrics.csv`: metrics recalculated directly by the executed notebooks,
  including G2 region-wise PCC and Tonsil joint-component display limits.
- `training_summary.csv`: convergence status, iteration count, final objective
  and recorded runtime from the diagnostics of the eight saved tutorial fits.
  These describe the corresponding saved fits, not a fresh notebook execution.

The main RNA fixed-reference recipe uses 100,000 steps and KKT 1e-10. The reviewed
manuscript draft's 3,000 / 1e-6 entry in that paragraph needs correction; those
latter settings remain valid for the independent STARmap sensitivity analysis.

The code now includes the original five-cycle profile auxiliary update for the
hierarchical model. Group-index order may differ from archived files and is
aligned by patient/condition identity before comparisons. Restoring this update
and checking the full-data objective do not establish an identical full training
trajectory; the tutorial therefore labels the CRC figures as archived results.

Pinned dependencies, explicit seeds and optional deterministic kernels support
repeatable new analyses. Cross-device or cross-version bitwise equality is not
assumed. No manuscript source files or raw encoder weights are distributed here.

The revised G2 figures retain the same 402-spot scores and the manuscript's fixed
illustrative zoom. The Tonsil maps use the Visium hexagonal coordinate transform,
verified against all 4,908 original spot records. The eleven displayed components
come only from the joint fit, with labelled per-component 99th-percentile color
limits. These display changes leave the complete fitted weights unchanged.
