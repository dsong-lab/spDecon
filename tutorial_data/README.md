# Tutorial data

The six complete example inputs and saved fits are distributed as the
[versioned data release](https://github.com/dsong-lab/spDecon/releases/tag/tutorial-data-v0.1.0).
The notebooks download and extract this bundle on first use. No subsampling is
applied for the tutorial figures. Each folder contains `data.npz`, fitted arrays,
and the annotations or settings used in that tutorial.

| Folder | Spots | Features | Saved fit |
|---|---:|---|---|
| cerebellum | 1,000 | 1,000 RNA | Full seed-1 reproduction |
| xenium | 256 | 1,027 RNA; 12 reference types | Full strict fixed-reference reproduction |
| g2 | 467 | 14,898 RNA; 32 morphology bins | Paired RNA and RNA + H&E reproduction |
| rcc | 1,595 | 405 RNA; 27 protein | Two independent seed-0 reproductions |
| crc | 102,544 | 480 RNA | Archived 12-section patient/condition fit |
| tonsil | 4,908 | 2,000 RNA; 35 protein | Full joint and assay-only reproductions |

Source datasets retain their original attribution and terms. Encoder weights are
not redistributed. See the website's Data and reproducibility page for provenance
and the distinction between saved fit inspection and optional model retraining.
