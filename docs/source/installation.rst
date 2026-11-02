Installation
============

spDecon supports **CPU and NVIDIA GPU** fitting. The study's reference-free and
joint analyses used GPUs. CPU fitting runs the same models and is suitable for
smaller datasets, including pseudo-spots and non-spatial bulk mixtures; large
datasets and image feature extraction benefit from a GPU. CPU and GPU fits can
differ numerically.

The analysis environment uses **Python 3.10.14 on Linux x86_64**, PyTorch 2.10.0,
NumPy 1.26.4 and SciPy 1.13.1. The GPU environment uses CUDA 12.8; the CPU setup
installs a CPU-only PyTorch build and does not require CUDA.

Install the pinned analysis environment
---------------------------------------

.. code-block:: bash

   git clone https://github.com/dsong-lab/spDecon.git
   cd spDecon
   python3.10 -m venv .venv
   source .venv/bin/activate
   unset PYTHONPATH
   export PYTHONNOUSERSITE=1
   # Choose one dependency set:
   python -m pip install -r requirements.lock.txt       # NVIDIA GPU, CUDA 12.8
   # python -m pip install -r requirements-cpu.txt      # CPU only
   python -m pip install --no-deps --no-build-isolation -e .
   python -m pip install -r requirements-notebooks.txt

Alternatively, ``conda env create -f environment.yml`` installs the pinned GPU Python
and numerical dependencies; then install the package and notebook requirements
with the last two commands above.

Open ``tutorials/*.ipynb`` in your preferred Jupyter interface or VS Code and
select this environment's Python kernel. The notebooks locate the repository
root automatically and read the example inputs and saved fits included in
``tutorial_data/``. Use ``git pull`` to obtain updated examples.

**Saved outputs:** the default ``RUN_FIT=False`` loads complete previously fitted
results and recomputes the tutorial metrics and plots. Change it to ``True`` to
train the model. Set ``VERBOSE=True`` for fitting progress and convergence status.

**Device selection:** use ``--device auto`` in the command line or
``DEVICE="auto"`` in notebooks to select CUDA when available and CPU otherwise.
Use ``cpu`` to force CPU or ``cuda`` (optionally ``cuda:0``) for a specific GPU.
The strict fixed-reference RNA notebook uses its CPU solver.

To check the full CPU pipeline after installation:

.. code-block:: bash

   python tests/check_cpu_workflow.py

The check fits synthetic pseudo-spots with calibrated spatial regularization and
bulk mixtures without coordinates, including all three initializations,
consensus and refinement. See :doc:`api` for the bulk command.

Optional H&E feature extraction
-------------------------------

.. code-block:: bash

   python -m pip install -r requirements-histology.lock.txt

For a CPU-only environment, use these commands instead:

.. code-block:: bash

   python -m pip install torchvision==0.25.0+cpu --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements-histology.txt

HIPT and UNI feature-extraction code is included in spDecon. Follow the G2
notebook to place the two HIPT checkpoints in ``checkpoints/hipt/`` and request
UNI access, then call ``spDecon.image_features.extract_histology`` with
``checkpoint_dir="checkpoints/hipt"``. The repository includes
the registered G2 H&E image, capture centers and complete derived morphology
features. Encoder weights are downloaded separately under their original terms.

Build the website locally
-------------------------

Use a separate environment for documentation; it needs no GPU, numerical package
or model data because the website renders the saved notebook outputs.

.. code-block:: bash

   python3.10 -m venv .docs-venv
   .docs-venv/bin/python -m pip install -r docs/requirements.txt
   .docs-venv/bin/python tools/build_docs.py
   .docs-venv/bin/python -m http.server --directory docs/build/html 8000

Open ``http://localhost:8000``. GitHub Actions uses the same build command and
publishes ``docs/build/html`` to GitHub Pages.
