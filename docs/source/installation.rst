Installation
============

The verified numerical environment is **Python 3.10.14 on Linux x86_64**, with
PyTorch 2.10.0 + CUDA 12.8, NumPy 1.26.4 and SciPy 1.13.1. Reference-free model
training uses an NVIDIA GPU. The supplied fitted results and all tutorial plots
can be inspected on a CPU; the strict RNA fixed-reference example also fits on CPU.

Install the pinned analysis environment
---------------------------------------

.. code-block:: bash

   git clone https://github.com/dsong-lab/spDecon.git
   cd spDecon
   python3.10 -m venv .venv
   source .venv/bin/activate
   unset PYTHONPATH
   export PYTHONNOUSERSITE=1
   python -m pip install -r requirements.lock.txt
   python -m pip install --no-deps --no-build-isolation -e .
   python -m pip install -r requirements-notebooks.txt

Alternatively, ``conda env create -f environment.yml`` installs the pinned Python
and numerical dependencies; then install the package and notebook requirements
with the last two commands above.

Open ``tutorials/*.ipynb`` in your preferred Jupyter interface or VS Code and
select this environment's Python kernel. The notebooks locate the repository
root automatically. Their first data-loading cell downloads the complete example
bundle from the versioned GitHub release if it is absent.

**Saved outputs:** the default ``RUN_FIT=False`` loads complete previously fitted
results and recomputes the tutorial metrics and plots. Change it to ``True`` to
run the full model. This setting does not shorten iterations or reduce the data.
The CRC 12-section fit is deliberately archived rather than retrained here.

Optional image encoders
-----------------------

.. code-block:: bash

   python -m pip install -r requirements-histology.lock.txt
   git clone https://github.com/dsong-lab/spEnhance.git external/spEnhance

Follow the G2 notebook to obtain HIPT checkpoints and request UNI access, then
call ``spDecon.image_features.extract_histology``. The data bundle already includes
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
