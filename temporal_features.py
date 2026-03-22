"""
Temporal enrichment for frame-wise features: adds short-context motion and smoothing.

Use with EMG/IMU pipelines where each row is one time step; helps separability of
activities that differ by dynamics rather than instantaneous amplitude alone.
"""

from __future__ import annotations

import numpy as np

try: 
    from scipy.ndimage import uniform_filter1d

    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


def augment_temporal_features(features: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Concatenate [instantaneous, time-smoothed, first-difference] feature blocks.

    Parameters
    ----------
    features : (N, F)
    window : odd-ish smoothing length for moving average (via uniform_filter1d)

    Returns
    -------
    (N, F * 3)
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be 2D (N, F)")

    n, f = x.shape
    w = max(3, int(window) | 1)

    if SCIPY_OK:
        smooth = np.zeros_like(x)
        for j in range(f):
            smooth[:, j] = uniform_filter1d(x[:, j], size=w, mode="nearest")
    else:
        kernel = np.ones(w) / w
        pad = w // 2
        smooth = np.zeros_like(x)
        for j in range(f):
            padded = np.pad(x[:, j], (pad, pad), mode="edge")
            smooth[:, j] = np.convolve(padded, kernel, mode="valid")[:n]

    vel = np.zeros_like(x)
    vel[1:] = x[1:] - x[:-1]

    return np.hstack([x, smooth, vel])


def temporal_feature_dim(base_dim: int) -> int:
    return base_dim * 3


def temporal_feature_names(base_names: list) -> list:
    return (
        [f"{n}_inst" for n in base_names]
        + [f"{n}_smooth" for n in base_names]
        + [f"{n}_d1" for n in base_names]
    )
