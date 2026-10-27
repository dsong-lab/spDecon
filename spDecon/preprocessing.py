"""Spot-total balancing and residuals used only for initialization."""
import numpy as np

EPS = 1e-10


def compute_exposure(counts: np.ndarray) -> np.ndarray:
    """Library size relative to the median positive library size."""

    libraries = np.maximum(np.asarray(counts, dtype=float), 0.0).sum(axis=1)
    positive = libraries[libraries > 0]
    reference = float(np.median(positive)) if positive.size else 1.0
    return np.maximum(libraries / max(reference, EPS), EPS)


def exposure_null_residuals(
    counts: np.ndarray, exposure: np.ndarray
) -> np.ndarray:
    """Clipped Pearson residuals from an exposure-only gene model."""

    counts = np.maximum(np.asarray(counts, dtype=float), 0.0)
    exposure = np.maximum(np.asarray(exposure, dtype=float), EPS)
    abundance = counts.sum(axis=0) / max(float(exposure.sum()), EPS)
    expected = np.maximum(exposure[:, None] * abundance[None, :], EPS)
    residuals = (counts - expected) / np.sqrt(expected)
    bound = np.sqrt(max(counts.shape[0], 1))
    return np.clip(residuals, -bound, bound)


def row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def exposure_balanced_input(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balance spot totals to their median for initialization."""

    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("counts must be a finite nonnegative matrix")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("counts contain an empty spot")
    exposure = compute_exposure(counts)
    return counts / exposure[:, None], exposure
