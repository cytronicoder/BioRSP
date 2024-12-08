"""Cumulative distribution functions (CDFs) for histograms."""

from typing import Tuple
import numpy as np


def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative distribution function (CDF) of a histogram.

    Parameters:
    - histogram (np.ndarray): 1D array of histogram counts.

    Returns:
    - np.ndarray: 1D array representing the CDF, normalized between 0 and 1.

    Raises:
    - TypeError: If the input histogram is not a NumPy array.
    - ValueError: If the input histogram is not one-dimensional.
    """
    if not isinstance(histogram, np.ndarray):
        raise TypeError("histogram must be a NumPy array.")
    if histogram.ndim != 1:
        raise ValueError("histogram must be a one-dimensional array.")

    total = histogram.sum()
    if total > 0:
        cdf = np.cumsum(histogram).astype(np.float64) / total
    else:
        cdf = np.zeros_like(histogram, dtype=np.float64)
    return cdf


def compute_cdfs(
    fg_histogram: np.ndarray, bg_histogram: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the cumulative distribution functions (CDFs) for foreground and background histograms.

    Parameters:
    - fg_histogram (np.ndarray): 1D array of histogram counts for foreground.
    - bg_histogram (np.ndarray): 1D array of histogram counts for background.

    Returns:
    - Tuple[np.ndarray, np.ndarray]:
        - fg_cdf (np.ndarray): CDF for the foreground histogram.
        - bg_cdf (np.ndarray): CDF for the background histogram.

    Raises:
    - TypeError: If either histogram is not a NumPy array.
    - ValueError: If either histogram is not one-dimensional.
    """
    fg_cdf = compute_cdf(fg_histogram)
    bg_cdf = compute_cdf(bg_histogram)
    return fg_cdf, bg_cdf
