# Tutorial data

The notebooks automatically download their prepared inputs and saved fits.
To extract it manually, unpack `spdecon-tutorial-data.tar.gz` at the repository root.
Each dataset folder contains prepared arrays and saved fits.

The bundle includes G2 paired fits calibrated and fitted with the same HIPT/UNI
morphology matrix. If you previously downloaded the G2 example, set
`REFRESH_DATA=True` in its loading cell once to retrieve the updated saved fits
from the existing download address. The other five examples are unchanged.

G2 includes `calibration.json`, `protocol.json` and the saved fit diagnostics.
Its training cell recalibrates from the current morphology matrix rather than
loading fixed coefficients, including when image features are re-extracted.

Each [tutorial notebook](../tutorials/) begins with a description of its dataset and links to the original sources.
Encoder weights are obtained separately following the G2 tutorial.
