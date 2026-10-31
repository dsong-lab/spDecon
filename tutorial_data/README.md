# Tutorial data

The notebooks automatically download their prepared inputs and saved fits.
To extract it manually, unpack `spdecon-tutorial-data.tar.gz` at the repository root.
Each dataset folder contains prepared arrays and saved fits.

The current bundle is `tutorial-data-v0.1.1`. It updates the G2 paired fits
to calibrate and fit with the same HIPT/UNI morphology matrix. The other
five examples are unchanged. The notebooks check the bundle version and
refresh earlier cached inputs automatically.

G2 includes `calibration.json`, `protocol.json` and the saved fit diagnostics.
Its training cell recalibrates from the current morphology matrix rather than
loading fixed coefficients, including when image features are re-extracted.

Each [tutorial notebook](../tutorials/) begins with a description of its dataset and links to the original sources.
Encoder weights are obtained separately following the G2 tutorial.
