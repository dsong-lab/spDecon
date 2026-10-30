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

## Original public sources

| Example | Original public sources |
|---|---|
| Cerebellum | [Slide-seq SCP354](https://singlecell.broadinstitute.org/single_cell/study/SCP354/slide-seq-study); [RETROFIT preparation scripts](https://github.com/qunhualilab/retrofit-scripts) and [archived benchmark](https://doi.org/10.5281/zenodo.19241634) |
| Xenium lymph node | [Xenium Prime target](https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression); [Flex 320k scFFPE reference](https://www.10xgenomics.com/datasets/320k_scFFPE_16-plex_GEM-X_FLEX), library LNReactive_BC9-10 |
| HER2ST G2 | [HER2ST Zenodo deposit](https://zenodo.org/records/4751624): counts, H&E images, coordinates and pathology annotations |
| RCC | [Xenium Protein FFPE Human Renal Carcinoma](https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma) |
| CRC | [GEO GSE294385](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294385); [CRC_micromets_ST pathology and code](https://github.com/yliuup/CRC_micromets_ST) |
| Tonsil | [Visium CytAssist Human Tonsil Gene and Protein Expression](https://www.10xgenomics.com/datasets/visium-cytassist-gene-and-protein-expression-library-of-human-tonsil-with-add-on-antibodies-h-e-6-5-mm-ffpe-2-standard) |

Source datasets retain their original attribution and terms. Encoder weights are
not redistributed. See the website's Data and reproducibility page for provenance
and the distinction between saved fit inspection and optional model retraining.
