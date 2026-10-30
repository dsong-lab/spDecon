# Tutorial figure specification

Author: Su Xu. Backend: Python / matplotlib. Output: PNG for the website,
editable-text SVG and PDF in `outputs/figures`, numerical tables in `outputs/tables`.
These are tutorial figures, not new manuscript panels.

| Example | Question and evidence | Panels | Interpretation limit |
|---|---|---|---|
| Cerebellum | Do inferred mixtures recover known pseudo-spot compositions? | Matched correlation matrix; truth/inferred spatial pies | Display coordinates are an artificial pseudo-spot grid |
| Xenium | How do fixed-reference cell fractions compare with scored groups? | Seven per-group correlations; B/T spatial fractions | 255 scored of 256 fitted; partial annotation coverage |
| G2 | Does the paired H&E model recover pathological structure more accurately? | Truth/RNA/H&E region maps; paired PCC/ARI/boundary F1 | One section; 402 labelled spots; no independent spot significance test |
| RCC | Which structures agree across independent assays? | Nine matched RNA/protein component pairs | Full-data matching for display, not held-out validation |
| CRC | What structure is captured by the 480-gene hierarchical fit? | All 12 section maps; shared/subject/condition decomposition | Saved full fit; no new 12-section training |
| Tonsil | What spatial structures occur in the shared paired-assay fit? | Eleven component maps; GC quantitative readout | Fitted component weights are molecular contributions |

Axes use common scales within comparable panels; categorical identity uses a
consistent palette. Numeric sources are recomputed from all supplied arrays.
All main example fits use one named seed; there are no confidence intervals or
p-values in these tutorial summaries. Morphology maps display derived assignments,
not altered H&E micrographs. Registered image and pixel calibration are supplied
for extraction. Figure review checks labels, legends, clipping, scalar ranges,
source counts, and agreement with the reproduction tables.
