# spDecon

**spDecon** is a Python framework for spatial molecular deconvolution. It learns
component profiles and spatial mixtures, supports fixed references and histology
guidance, and jointly analyzes multiple samples or paired RNA–protein assays.
Its core model, `Y ≈ diag(exposure) @ P @ B`, combines Poisson/KL reconstruction
with spatial regularization.

[Documentation](https://dsong-lab.github.io/spDecon/) ·
[Tutorials](https://dsong-lab.github.io/spDecon/tutorial.html) ·
[Example data](https://github.com/dsong-lab/spDecon/releases/tag/tutorial-data-v0.1.0)

## Installation

Use Python 3.10. Reference-free training requires an NVIDIA GPU; the supplied
results and tutorial plots can be explored on CPU.

```bash
git clone https://github.com/dsong-lab/spDecon.git
cd spDecon
python3.10 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install -r requirements-notebooks.txt
```

See the [installation guide](https://dsong-lab.github.io/spDecon/installation.html)
for Conda and optional HIPT/UNI dependencies.

## Quick start

Run reference-free deconvolution on a counts CSV (spots × genes) and a coordinate
CSV (matching spot IDs with `x,y` columns). Both files use spot IDs in the first
column.

```bash
python run_spdecon.py core \
  --counts X.csv --coordinates coords.csv \
  --k 11 --seeds 1 3 5 --deterministic --output runs/example
```

The output includes component weights, profiles and optimization history.
Fitting prints the current objective, KKT residuals and final convergence status.
See the [model](https://dsong-lab.github.io/spDecon/model.html) and
[Python API](https://dsong-lab.github.io/spDecon/api.html) for the other modes.

## Tutorials

Each notebook explains the model, loads example data and produces figures.
Open it from the cloned repository; the data bundle downloads automatically.

| Workflow | Example and output | Notebook |
|---|---|---|
| Reference-free | Cerebellum: correlations and composition pies | [01](tutorials/01_reference_free.ipynb) |
| Reference-based | Xenium lymph node: fixed signatures and cell fractions | [02](tutorials/02_reference_based.ipynb) |
| Histology-guided | HER2ST G2: HIPT/UNI extraction and spatial mixtures | [03](tutorials/03_histology.ipynb) |
| Cross-modality | RCC: independently fitted RNA and protein structures | [04](tutorials/04_cross_modality.ipynb) |
| Joint multi-sample | CRC: shared, patient and condition profiles across 12 sections | [05](tutorials/05_multisample.ipynb) |
| Joint multi-omics | Tonsil: shared RNA–protein spatial components | [06](tutorials/06_joint_multiomics.ipynb) |

`RUN_FIT=False` loads saved fits; set it to `True` to train the model.
`VERBOSE=True` shows fitting progress. The CRC example uses the archived
480-gene, 12-section fit by default.

## Public data sources

Original datasets are linked below. The [example-data release](https://github.com/dsong-lab/spDecon/releases/tag/tutorial-data-v0.1.0)
provides prepared inputs and saved spDecon fits for the tutorials.

| Example | Public sources |
|---|---|
| Cerebellum | [Slide-seq SCP354](https://singlecell.broadinstitute.org/single_cell/study/SCP354/slide-seq-study); [RETROFIT data and scripts](https://github.com/qunhualilab/retrofit-scripts), [v1.0.0 code archive](https://zenodo.org/records/19241634) |
| Xenium lymph node | [Xenium Prime](https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression); [Flex 320k scFFPE reference](https://www.10xgenomics.com/datasets/320k_scFFPE_16-plex_GEM-X_FLEX), library LNReactive_BC9-10 |
| HER2ST G2 | [HER2ST counts, H&E images and annotations](https://zenodo.org/records/4751624) |
| RCC | [Xenium Protein Human Renal Carcinoma](https://www.10xgenomics.com/datasets/xenium-protein-ffpe-human-renal-carcinoma) |
| CRC | [GEO GSE294385](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294385); [pathology annotations](https://github.com/yliuup/CRC_micromets_ST) |
| Tonsil | [Visium CytAssist Human Tonsil Gene and Protein Expression](https://www.10xgenomics.com/datasets/visium-cytassist-gene-and-protein-expression-library-of-human-tonsil-with-add-on-antibodies-h-e-6-5-mm-ffpe-2-standard) |

Source datasets and encoder weights retain their original attribution and terms.
