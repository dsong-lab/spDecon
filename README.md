# spDecon

**spDecon** is a Python framework for spatial molecular deconvolution. It learns
component profiles and spatial mixtures, supports fixed references and histology
guidance, and jointly analyzes multiple samples or paired RNA–protein assays.
Its core model, `Y ≈ diag(exposure) @ P @ B`, combines Poisson/KL reconstruction
with spatial regularization.

[Documentation](https://dsong-lab.github.io/spDecon/) ·
[Tutorials](https://dsong-lab.github.io/spDecon/tutorial.html)

## Installation

Use Python 3.10. Fitting supports CPU and NVIDIA GPU; the study's reference-free
and joint analyses used GPU acceleration.

```bash
git clone https://github.com/dsong-lab/spDecon.git
cd spDecon
python3.10 -m venv .venv
source .venv/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1
# Choose one: GPU (CUDA 12.8) or CPU-only.
python -m pip install -r requirements.lock.txt       # GPU
# python -m pip install -r requirements-cpu.txt      # CPU
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
  --k 11 --device auto --seeds 1 3 5 --deterministic --output runs/example
```

`--device auto` uses CUDA when available and CPU otherwise. Select `--device cpu`
or `--device cuda` explicitly to choose the fitting device.

The output includes component weights, profiles and optimization history.
Fitting prints the current objective, KKT residuals and final convergence status.
See the [model](https://dsong-lab.github.io/spDecon/model.html) and
[Python API](https://dsong-lab.github.io/spDecon/api.html) for the other modes.

## Tutorials

Each notebook explains the model, loads example data and produces figures.
Open it from the cloned repository; the data bundle downloads automatically.
Inputs use AnnData for measurements, coordinates and sample metadata; see the
[input guide](https://dsong-lab.github.io/spDecon/tutorial.html#anndata-input)
and [function parameters](https://dsong-lab.github.io/spDecon/api.html).

| Workflow | Example and output | Notebook |
|---|---|---|
| Reference-free | Cerebellum: correlations and composition pies | [01](tutorials/01_reference_free.ipynb) |
| Reference-based | Xenium lymph node: fixed signatures and cell fractions | [02](tutorials/02_reference_based.ipynb) |
| Histology-guided | HER2ST G2: HIPT/UNI extraction and spatial mixtures | [03](tutorials/03_histology.ipynb) |
| Cross-modality | RCC: independently fitted RNA and protein structures | [04](tutorials/04_cross_modality.ipynb) |
| Joint multi-sample | Named design effects; CRC as a 12-section worked example | [05](tutorials/05_multisample.ipynb) |
| Joint multi-omics | Tonsil: shared RNA–protein spatial components | [06](tutorials/06_joint_multiomics.ipynb) |

`RUN_FIT=False` loads saved fits; set it to `True` to train the model.
Device selection, design effects, bulk workflows and optional checks are
explained in the corresponding tutorials.

The [H&E tutorial](tutorials/03_histology.ipynb) also covers extracting HIPT/UNI
embeddings and using the resulting morphology features in spDecon.
