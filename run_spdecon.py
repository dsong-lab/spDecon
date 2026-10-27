"""Run spot-level spDecon from the repository root."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "code/smipf_spatial/run_spatial.py"),
        run_name="__main__",
    )
