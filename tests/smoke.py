"""Run the complete CUDA workflow on synthetic data and validate its outputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(17)
    n_spots, n_genes, k = 72, 240, 3
    xy = np.column_stack([np.tile(np.arange(12), 6), np.repeat(np.arange(6), 12)])
    groups = xy[:, 0] // 4
    p = np.full((n_spots, k), 0.05)
    p[np.arange(n_spots), groups] = 0.9
    b = np.full((k, n_genes), 0.1)
    genes_per_component = n_genes // k
    for component in range(k):
        b[component, component * genes_per_component:(component + 1) * genes_per_component] = 5.0
    b /= b.sum(axis=1, keepdims=True)
    counts = rng.poisson(300 * (p @ b))
    spots = [f"spot_{i}" for i in range(n_spots)]
    genes = [f"gene_{i}" for i in range(n_genes)]

    with tempfile.TemporaryDirectory(prefix="spdecon-smoke-") as temp:
        work = Path(temp)
        pd.DataFrame(counts, index=spots, columns=genes).to_csv(work / "counts.csv")
        # Reverse the coordinate rows to exercise identifier-based alignment.
        pd.DataFrame(xy, index=spots, columns=["x", "y"]).iloc[::-1].to_csv(
            work / "coordinates.csv"
        )
        process = subprocess.run([
            sys.executable, "-B", str(root / "run_spdecon.py"),
            "--counts", str(work / "counts.csv"),
            "--coordinates", str(work / "coordinates.csv"),
            "--k", str(k), "--seeds", "1",
            "--output", str(work / "fit"),
        ], cwd=work)
        output = work / "fit"
        if process.returncode:
            diagnostics = output / "seed_1/diagnostics.json"
            if diagnostics.exists():
                info = json.loads(diagnostics.read_text())
                print({key: info[key] for key in (
                    "converged", "termination_reason", "outer_iterations",
                    "final_all_KKT_pass", "branch_diagnostics",
                )}, flush=True)
            process.check_returncode()
        complete = json.loads((output / "COMPLETE.json").read_text())
        assert complete["status"] == "passed", complete
        assert np.isfinite(complete["spatial_weight"])
        assert complete["spatial_weight"] > 0
        fitted_p = pd.read_csv(output / "seed_1/P.csv", index_col=0)
        fitted_b = pd.read_csv(output / "seed_1/B.csv", index_col=0)
        assert fitted_p.index.tolist() == spots
        assert fitted_b.columns.tolist() == genes
        assert fitted_p.shape == (n_spots, k)
        assert fitted_b.shape == (k, n_genes)
        for matrix in (fitted_p.to_numpy(), fitted_b.to_numpy()):
            assert np.isfinite(matrix).all() and (matrix >= 0).all()
            np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=2e-6, rtol=0)
        with np.load(output / "graph.npz") as graph:
            assert len(graph["edge_i"]) > 0
            assert graph["spot_ids"].tolist() == spots
        print("PASS: complete three-branch consensus + spatial TV fit; valid P/B outputs.")


if __name__ == "__main__":
    main()
