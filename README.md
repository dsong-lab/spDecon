# spDecon

**spDecon** is a Python framework for spatial molecular deconvolution. It learns
component profiles and spatial mixtures, supports fixed references and histology
guidance, and jointly analyzes multiple samples or paired RNA–protein assays.
Its core model, `Y ≈ diag(exposure) @ P @ B`, combines Poisson/KL reconstruction
with spatial regularization.

[Documentation](https://dsong-lab.github.io/spDecon/) ·
[Tutorials](https://dsong-lab.github.io/spDecon/tutorial.html)

## Installation

Use Python 3.10. spDecon supports CPU and NVIDIA GPU fitting, including the full
pseudo-spot workflow and non-spatial bulk deconvolution. The study's reference-free
and joint analyses were run on GPU; GPU acceleration is recommended for large datasets.

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

For **bulk samples without spatial coordinates**, use a samples × genes raw-count
CSV and disable the spatial penalty:

```bash
python run_spdecon.py core \
  --counts bulk_counts.csv --k 3 --spatial-weight 0 \
  --device cpu --seeds 1 --output runs/bulk_cpu
```

Both commands run all three initializations, component alignment, consensus and
refinement. The bulk fit estimates molecular component contributions; cell-type
names and cell fractions require suitable reference information.

With a fixed reference (types × genes, the same ordered genes as the counts):

```bash
python run_spdecon.py reference \
  --counts bulk_counts.csv --reference reference.csv \
  --device cpu --output runs/bulk_reference_cpu
```

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
`DEVICE="auto"` selects CPU or GPU for new fits; use `DEVICE="cpu"` to force CPU.
The strict fixed-reference RNA notebook uses its CPU solver.
`VERBOSE=True` shows fitting progress. The CRC example uses the archived
480-gene, 12-section fit by default.

The histology tutorial uses the same HIPT/UNI morphology matrix for coefficient
calibration, reconstruction and edge weighting. Its paired RNA-only control
shares the unrefined consensus and calibrated spatial coefficient. It includes
code for extracting HIPT/UNI embeddings, building spot-level morphology features,
and passing them to spDecon. See the
[H&E workflow](https://dsong-lab.github.io/spDecon/api.html#histology-guided-fitting)
for the same steps with your own image and AnnData object.

## Joint samples with named effects

Choose the grouping labels from your metadata. One effect gives flat deviations;
a parent and child give two nested levels. For example, visits within donors:

```python
from spDecon.multisample import fit_multisample

effects = {
    "donor": {"column": "donor_id", "penalty": 10.},
    "visit": {"column": "timepoint", "parent": "donor", "penalty": 20.},
}
result = fit_multisample(
    adata.to_df(), adata.obs, n_components=10, effects=effects,
    coordinates=adata.obsm["spatial"], device="auto",
)
deviations = {
    name: result.extra[f"effect_delta_{name}"]
    for name in result.extra["effect_names"]
}
```

`adata.obs` also needs `section_id`; spatial edges stay within each physical
section. `parent` refers to an effect name. Each donor has its own visit
deviations, centered within that donor. Larger penalties shrink deviations more
strongly; the values above are example settings. For one grouping level, use
`effects = {"sample": {"column": "sample_id", "penalty": 10.}}`.

The CRC tutorial uses the same interface with patient and condition labels.
The [API guide](https://dsong-lab.github.io/spDecon/api.html#joint-multi-sample-fitting)
describes the supported designs, named output labels, and `--effects effects.json`
command-line option. Existing `group_by` / `subject_column` calls remain supported.
Paired assays with different feature panels use the separate multi-omics interface.
Run `python tests/check_multisample_effects.py` to check flat/nested designs,
equivalence with the existing fitting interface and JSON/array exports on CPU.

## CPU workflow check

```bash
python tests/check_cpu_workflow.py
```

This pools synthetic single-cell counts into 72 mixtures and runs the complete
pseudo-spot (calibrated spatial TV) and bulk (no spatial coordinates) workflows
with CUDA disabled. It checks convergence of all three branches and refinement,
finite nonnegative factors, row sums and sample/feature IDs.
It also checks fixed-reference bulk fitting on CPU.
