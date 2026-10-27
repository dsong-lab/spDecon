# spDecon

spDecon performs reference-free, spot-level deconvolution of spatial gene
expression counts. It estimates component weights for each spot and a gene
expression profile for each component, without a single-cell reference.

This repository contains the model used in Section 1 of the manuscript:
three initialization branches, full-gene KL fitting, profile alignment and
consensus, followed by joint KL + spatial total variation (TV) optimization.

## Core code

The optimizer is **[`spDecon/model.py`](spDecon/model.py)**. Its
`fit_spdecon()` function jointly estimates spot weights `P` and component
profiles `B`. The same solver fits the initialization branches with TV disabled.

| File | Role |
| --- | --- |
| [`run_spdecon.py`](run_spdecon.py) | Command-line entry point. |
| [`spDecon/cli.py`](spDecon/cli.py) | Complete workflow and input/output handling. |
| [`spDecon/model.py`](spDecon/model.py) | Simplex-constrained KL + TV optimizer. |
| [`spDecon/initialization.py`](spDecon/initialization.py) | Three initializations and their full-gene KL fits. |
| [`spDecon/consensus.py`](spDecon/consensus.py) | Profile alignment and equal-weight consensus. |
| [`spDecon/spatial.py`](spDecon/spatial.py) | Coordinate graph and spatial-weight calibration. |
| [`spDecon/preprocessing.py`](spDecon/preprocessing.py) | Exposure balancing and residuals for initialization. |
| [`spDecon/clustering.py`](spDecon/clustering.py) | Fixed-K expression clustering using Scanpy and igraph. |
| [`spDecon/config.py`](spDecon/config.py) | Numerical solver settings. |

## Installation

Use Python 3.10 and an NVIDIA GPU with CUDA-enabled PyTorch. The complete
workflow requires CUDA and loads the count matrix into memory.

```bash
git clone https://github.com/dsong-lab/spDecon.git
cd spDecon
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Build used for validation; select a compatible build for your NVIDIA driver.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available` must be `True`. Other PyTorch builds are listed on the
[PyTorch installation page](https://pytorch.org/get-started/locally/).

## Input and run

Provide two CSV files with unique spot IDs in the first column:

- `counts.csv`: spots by genes, with unique gene names and raw, nonnegative
  counts. Do not normalize or log-transform the input; remove empty spots.
- `coordinates.csv`: numeric `x` and `y` columns for the same spots.
  Coordinates are reordered to match the count matrix automatically.

For example, the file formats are:

```csv
spot_id,GeneA,GeneB,GeneC
spot_1,12,3,0
spot_2,7,8,1
spot_3,0,4,15
```

```csv
spot_id,x,y
spot_1,0,0
spot_2,1,0
spot_3,2,0
```

These snippets illustrate the formats, not a fitting dataset. If coordinates
cover multiple tissue sections, add a `block` column identifying each section,
or supply a spot-indexed CSV with a `block` column via `--blocks`. Graph edges
stay within each section. Each section needs at least two spots and positive
median nearest-neighbor spacing.

Choose `K`, the number of latent components, for your dataset:

```bash
python run_spdecon.py \
  --counts counts.csv \
  --coordinates coordinates.csv \
  --k 5 \
  --seeds 1 \
  --output results/my_sample
```

The output directory must be new. Omit `--seeds` to run seeds `1 3 5`.
Components are aligned between branches within each seed; numbering across
seeds may differ. Spatial weight is calibrated once from the seed-1 consensus
and reused across seeds. `--spatial-weight VALUE` overrides calibration;
zero disables TV. Coordinates are still required by this command.
Run `python run_spdecon.py --help` for all options.

## Model

For counts `Y` (spots by genes), the objective is normalized generalized KL
divergence plus smoothed graph TV:

```text
s[i] = sum_g Y[i, g]
M = diag(s) @ P @ B
loss = D_KL(Y || M) / sum(Y) + lambda * TV(P)
D_KL(Y || M) = sum_{i,g} [Y[i,g] * log(Y[i,g] / M[i,g]) - Y[i,g] + M[i,g]]
TV(P) = sum_{(i,j)} w[i,j] * (sqrt(||P[i,:] - P[j,:]||_2^2 + epsilon^2) - epsilon)
        / sum_{(i,j)} w[i,j]
```

The logarithmic term is zero where `Y[i,g] = 0`. Both `P` (spots by components)
and `B` (components by genes) are nonnegative and row-normalized. Relative to
the manuscript notation, `Y = G.T`, `B = C.T`, and code `P` is the transpose
of manuscript `P`. The KL term also equals a Poisson negative log-likelihood
up to a parameter-independent constant and normalization.

1. Initialize using all genes, exposure-residual feature selection, and
   expression-neighborhood Louvain clustering. Refine each branch against
   all genes of the original counts using KL loss.
2. Align component profiles to the all-gene branch using cosine similarity
   and Hungarian matching, then average the three branches equally.
3. Starting directly from this consensus, jointly update `P` and `B` with
   KL + TV. `P` uses projected-gradient updates; `B` uses mixture EM updates.

The spatial graph uses symmetric four-nearest-neighbor edges, restricted to
distances at most 1.5 times each section's median nearest-neighbor spacing,
with Gaussian distance weights. TV uses `epsilon = 0.02`; the seed-1 gradient
calibration uses a spatial-to-KL gradient ratio of `0.05`.

`P` represents component signal weights. Cell-type names and interpretation
as cell-number fractions require biological validation.

## Output

Each `seed_<n>/` directory contains:

| File | Contents |
| --- | --- |
| `P.csv` | Spot-by-component weights; each row sums to one. |
| `B.csv` | Component-by-gene profiles; each row sums to one. |
| `fit.npz` | Final factors, initial consensus, spot/gene IDs, and library sizes. |
| `branches.npz` | Aligned factors and component permutations for the three branches. |
| `history.csv` | Objective, KL, TV, and convergence checks per iteration. |
| `diagnostics.json` | Final and branch convergence information and model settings. |

The output root contains the graph, calibration, configuration, input/source
hashes, and `COMPLETE.json`. Automatic calibration also saves
`calibration_consensus.npz`. Nonconverged fits retain their outputs and exit
with an error; inspect their diagnostics.

## Validation

```bash
python tests/check_model.py       # CPU: KL/TV gradients and constrained fitting
python tests/check_clustering.py  # CPU: fixed-K clustering
python tests/smoke.py             # CUDA: complete workflow and output validation
```

The workflow uses float32 on CUDA. TV-gradient accumulation can vary between
runs, so a fixed seed does not guarantee identical factors. Very small losses
can also fail the relative-change convergence check in float32 despite passing
the projected-gradient checks.

Maintainer: [Su Xu](https://github.com/sxu509).
