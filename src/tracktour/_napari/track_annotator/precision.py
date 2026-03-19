"""Precision estimation utilities for the TrackAnnotator widget."""

import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


def _g(t, n, c, p):
    """Total errors seen by step t under D-UCB model."""
    numerator = 1 - (1 / c) ** (t + 1)
    denominator = c - 1
    multiplier = (c - 1) * n * p
    return multiplier * (numerator / denominator)


def _h(t, n, c, p):
    """Cumulative precision up to step t under D-UCB model.

    Parameters
    ----------
    t : float or array
        Number of edges sampled so far (1-indexed).
    n : int
        Total edges in the dataset.
    c : float
        D-UCB boost parameter (>= 1).
    p : float
        True error rate (0..1).
    """
    return (t - _g(t, n, c, p)) / t


def rolling_error_rate(errors, window=100):
    """Compute rolling error rate with a sliding window.

    Parameters
    ----------
    errors : list[int]
        Per-annotation error indicator: 1 = FP, 0 = TP.
    window : int
        Window size (default 100).

    Returns
    -------
    np.ndarray of float, same length as errors.
    """
    if not errors:
        return np.array([])
    arr = np.array(errors, dtype=float)
    cs = np.cumsum(arr)
    result = np.empty(len(arr))
    full = min(window, len(arr))
    result[:full] = cs[:full] / np.arange(1, full + 1)
    if len(arr) > window:
        result[window:] = (cs[window:] - cs[: len(arr) - window]) / window
    return result


def estimate_precision_simple(errors):
    """Estimate precision from an unbiased (random) sample.

    Returns None if errors is empty.
    """
    if not errors:
        return None
    fp = sum(errors)
    tp = len(errors) - fp
    return tp / (tp + fp)


def estimate_precision_ducb(errors, n_total):
    """Fit h(t) to cumulative precision history to estimate true precision.

    Parameters
    ----------
    errors : list[int]
        Per-annotation error indicator: 1 = FP, 0 = TP.
    n_total : int
        Total edges in the dataset (fixed parameter n in the model).

    Returns
    -------
    (c, p) or (None, None) on failure.
        fitted parameters (n is constant for each dataset).
    """
    T = len(errors)
    if T < 2:
        return None, None

    t_arr = np.arange(1, T + 1, dtype=float)
    cumulative = np.cumsum(errors)
    h_obs = (t_arr - cumulative) / t_arr  # cumulative precision at each step

    def h_fit(t, c, p):
        return _h(t, n_total, c, p)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                h_fit,
                t_arr,
                h_obs,
                p0=[2.0, 0.1],
                bounds=([1.0, 0.0], [np.inf, 1.0]),
                maxfev=10000,
            )
        c, p = popt
        return float(c), float(p)
    except RuntimeError:
        return None, None


def get_precision_estimate_at_end(t, n, c, p):
    """Get precision estimate at the end of sampling."""
    return _h(t, n, c, p)
