# Tutorial figure specification

Backend: Python / matplotlib. Output: PNG for the website,
editable-text SVG and PDF in `outputs/figures`, numerical tables in `outputs/tables`.
These are tutorial figures, not new manuscript panels.

| Example | Question and evidence | Panels | Interpretation limit |
|---|---|---|---|
| Cerebellum | Do inferred mixtures recover known pseudo-spot compositions? | Matched correlation matrix; truth/inferred spatial pies | Display coordinates are an artificial pseudo-spot grid |
| Xenium | How do fixed-reference cell fractions compare with scored groups? | Seven per-group correlations; B/T spatial fractions | 255 scored of 256 fitted; partial annotation coverage |
| G2 | Does the paired H&E model recover pathological structure more accurately? | Text-free H&E/pathology/RNA/RNA+H&E plate with a fixed manuscript zoom; region-wise paired PCC; stem-and-point ARI/PCC/boundary F1 | One section; 402 labelled spots; the archived zoom was selected for illustration, not independent evaluation |
| RCC | Which structures agree across independent assays? | Nine matched RNA/protein component pairs | Full-data matching for display, not held-out validation |
| CRC | What structure is captured by the 480-gene hierarchical fit? | All 12 section maps; shared/subject/condition decomposition | Saved full fit; no new 12-section training |
| Tonsil | What spatial structures occur in the shared paired-assay fit? | Eleven joint component maps on the correctly spaced Visium hexagonal lattice | Per-component 99th-percentile display caps are labelled; colors cannot be compared across different scales |

Axes use common scales within comparable panels; categorical identity uses a
consistent palette. Numeric sources are recomputed from all supplied arrays.
All main example fits use one named seed; there are no confidence intervals or
p-values in these tutorial summaries. Morphology maps display derived assignments,
not altered H&E micrographs. Registered image and pixel calibration are supplied
for extraction. Figure review checks labels, legends, clipping, scalar ranges,
source counts, and agreement with the reproduction tables.

The G2 plate uses original registered image pixels, the published six-region
palette, mixture pies, and the same 402 labelled locations in both model arms.
Its fixed manuscript crop is [1030.5094, 2997.1506, 1905.3794, 3872.0206]
in original image pixels (x0, x1, y0, y1). Column meanings and region colors are
explained outside the image; there are no embedded titles or legends. Region PCC
uses paired markers and region names without a numerical annotation column.
The three overall metrics use vertical stems and colored points on a 0–1 axis;
they are single-section measurements, so no uncertainty bars are drawn.

Tonsil coordinates are array_row/array_col, converted to x=array_col/2 and
y=array_row*sqrt(3)/2 before plotting with equal aspect. Hexagons represent display
neighborhoods, not physical capture footprints. Each of the eleven joint maps
has its own labelled colorbar; the top 1% is saturated only for display. Full
weights and the display limits are exported without changing the fitted arrays.
