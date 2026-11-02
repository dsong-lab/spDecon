# Tutorial data

Prepared inputs and saved fits for all six notebooks are included in this
directory and obtained with `git clone`. Use `git pull` to update the examples.
Each dataset folder contains prepared arrays and saved fits.

The G2 paired fits were calibrated and fitted with the same HIPT/UNI morphology
matrix.

G2 includes `calibration.json`, `protocol.json` and the saved fit diagnostics.
Its training cell recalibrates from the current morphology matrix rather than
loading fixed coefficients, including when image features are re-extracted.

Each [tutorial notebook](../tutorials/) begins with a description of its dataset and links to the original sources.
Encoder weights are obtained separately following the G2 tutorial.
