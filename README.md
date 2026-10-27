# spDecon

spDecon performs reference-free, spot-level deconvolution of spatial gene
expression counts. Given a spot-by-gene count matrix, spatial coordinates,
and a chosen number of components, it estimates component weights for each
spot and a gene expression profile for each component.

This repository contains the basic RNA workflow: three complementary
initializations, profile-based alignment and consensus, followed by joint
generalized KL optimization with spatial total variation (TV) regularization.
Histology, reference-based, multi-sample, and multimodal extensions are outside
the scope of this release.

## Core code

Start with [`code/spDecon_spatial/model.py`](code/spDecon_spatial/model.py):
its `fit_spdecon()` function is the main optimizer that jointly updates the
spot weights `P` and expression profiles `B` with generalized KL loss and spatial TV.

| File | Role |
| --- | --- |
| [`run_spdecon.py`](run_spdecon.py) | Command-line entry point. |
| [`code/spDecon_spatial/run_spatial.py`](code/spDecon_spatial/run_spatial.py) | Complete workflow: read inputs, build the spatial graph, calibrate TV, initialize, fit, and save outputs. |
| [`code/spDecon_spatial/model.py`](code/spDecon_spatial/model.py) | Main joint Poisson/KL + TV optimizer (`fit_spdecon`). |
| [`code/spDecon_core/src/consensus_raw_poisson/model.py`](code/spDecon_core/src/consensus_raw_poisson/model.py) | Fit the three initialization branches, align components, and form their consensus. |
| [`code/spDecon_core/src/louvain_init_poisson/clustering.py`](code/spDecon_core/src/louvain_init_poisson/clustering.py) | Local fixed-K Louvain initialization using Scanpy and igraph. |
| [`code/spDecon_core/src/poisson_hmrf/model.py`](code/spDecon_core/src/poisson_hmrf/model.py) | Shared count-only optimizer used to fit the initialization branches. |

The other modules under `code/spDecon_core/src/` provide initialization and
numerical helpers. The `poisson_hmrf` package name describes the shared solver;
the basic workflow uses its `rna_only_poisson` mode, with spatial TV applied
by the main optimizer above.

## Installation

Use Python 3.10 and an NVIDIA GPU with CUDA-enabled PyTorch. The spatial
command-line workflow requires CUDA and loads the count matrix into memory.

```bash
git clone https://github.com/dsong-lab/spDecon.git
cd spDecon
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA-enabled PyTorch build appropriate for your NVIDIA driver.
# The environment used to check this release runs PyTorch 2.10.0 + CUDA 12.8.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available` must be `True`. For a different CUDA setup, select the
appropriate build from the [PyTorch installation page](https://pytorch.org/get-started/locally/).

## Input

Supply two CSV files. The first column of each file contains unique spot IDs.
The two files must contain the same spots; coordinates are reordered to match
the count matrix automatically.

**`counts.csv`**: rows are spots and columns are genes. Use raw, nonnegative
counts, with unique gene names and no empty spots. Do not log-transform or
library-normalize the input.

```csv
spot_id,GeneA,GeneB,GeneC
spot_1,12,3,0
spot_2,7,8,1
spot_3,0,4,15
```

**`coordinates.csv`**: provide numeric `x` and `y` coordinates. Use a consistent
coordinate scale within each tissue section.

```csv
spot_id,x,y
spot_1,0,0
spot_2,1,0
spot_3,2,0
```

These snippets illustrate the file format; they are not a fitting example.
For multiple sections, add a `block` column identifying each section, or
supply a separate spot-indexed CSV with a `block` column via `--blocks`.
Spatial edges are built within each block. Without block labels, all spots
are treated as a single section. Each section needs at least two spots and
positive median nearest-neighbor spacing.

## Run

Choose `K`, the number of latent components, for your dataset. For example,
to fit five components with one seed:

```bash
python run_spdecon.py \
  --counts counts.csv \
  --coordinates coordinates.csv \
  --k 5 \
  --seeds 1 \
  --output results/my_sample
```

The output directory must not already exist. Omit `--seeds` to run the default
seeds `1 3 5`. Each seed runs all three initialization branches. Components
are aligned between branches within a seed; component numbering across
different seeds is not guaranteed to match.

The spatial weight is calibrated once from the seed-1 consensus and reused
across requested seeds. Use `--spatial-weight 0` for a count-only fit, or
provide a nonnegative value to override calibration. Coordinates are still
required by this command when the spatial weight is zero.

```bash
python run_spdecon.py --help
```

Run a small synthetic end-to-end check, including output validation:

```bash
python tests/check_clustering.py  # CPU-only clustering check
python tests/smoke.py
```

## Output

Each `seed_<n>/` directory contains:

| File | Contents |
| --- | --- |
| `P.csv` | Spot-by-component weights; each row sums to one. |
| `B.csv` | Component-by-gene relative expression profiles; each row sums to one. |
| `fit.npz` | Final factors, unrefined consensus, count-only control, IDs, and library sizes. |
| `branches.npz` | Aligned factors from the three initialization branches. |
| `history.csv` | Optimization history. |
| `diagnostics.json` | Convergence information and model settings. |

The output root contains `graph.npz`, `graph_info.json`, `calibration.json`,
`config.json`, and `COMPLETE.json`. Automatic calibration also saves
`calibration_consensus.npz`. `COMPLETE.json` reports convergence of the final
spatial fit and all three initialization branches. A nonconverged run retains
its outputs and exits with an error.

## Model

For counts `Y`, spDecon minimizes generalized KL divergence plus spatial TV:

```text
M = diag(s) @ P @ B
s[i] = sum_g Y[i, g]
loss = D_KL(Y || M) / sum(Y) + lambda * TV(P)
D_KL(Y || M) = sum_{i,g} [Y[i,g] * log(Y[i,g] / M[i,g]) - Y[i,g] + M[i,g]]
```

The logarithmic term is zero where `Y[i,g] = 0`. This KL data term is also
equivalent, up to a parameter-independent constant and the stated
normalization, to the negative log-likelihood of the Poisson count model:

```text
Y[i, g] ~ Poisson(s[i] * (P @ B)[i, g])
s[i] = sum_g Y[i, g]
```

Both `P` and `B` are nonnegative and row-normalized. The initialization
branches use all genes, exposure-residual feature selection, and an
expression-neighborhood Louvain partition. Their profiles are aligned by
cosine similarity and Hungarian matching, then averaged with equal weights.
The final optimizer jointly updates `P` and `B` from this consensus.

Spatial regularization uses a symmetric four-nearest-neighbor graph, built
separately for each section and restricted to distances at most 1.5 times
that section's median nearest-neighbor spacing. The default gradient-ratio
calibration uses `alpha = 0.05`; the smoothed TV term uses `epsilon = 0.02`.

`P` represents component signal weights. Assigning cell-type names or
interpreting these weights as cell-number fractions requires biological
validation. The workflow does not use cell-type labels or a single-cell
reference during fitting.

The optimizer uses float32 on CUDA. CUDA accumulation in the TV gradient
can change optimization trajectories, so a fixed seed does not guarantee
identical fitted factors across repeated runs or environments.
Very small residual losses can also fail the relative-change convergence
check in float32 even when the projected-gradient checks pass; inspect the
saved diagnostics when a run reaches the iteration limit.

## Code and acknowledgments

The implementation is organized under `code/spDecon_core/` and
`code/spDecon_spatial/`, as described in the core-code map above.

The local Louvain initializer calls
[Scanpy](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.louvain.html)
for clustering and [igraph](https://python.igraph.org/) for modularity scoring.
It searches for the requested number of components using ten numerical
starts per resolution and expands the expression neighborhood when needed.
The required libraries are listed in `requirements.txt`.

Maintainer: [Su Xu](https://github.com/sxu509).
