from typing import Tuple

import numpy as np


def compute_absolute_mode(
    foreground_histogram: np.ndarray,
    background_histogram: np.ndarray,
    n_f: int,
    n_b: int,
) -> Tuple[np.ndarray, float]:
    """


    Parameters
    ----------
    foreground_histogram : np.ndarray
        Histogram counts of foreground points within each angular bin of the scanning window.
    background_histogram : np.ndarray
        Histogram counts of background points within each angular bin of the scanning window.
    n_f : int
        Total number of foreground cells within the current scanning window.
    n_b : int
        Total number of background cells within the current scanning window.

    Returns
    -------
    rsp_abs_bin : np.ndarray
        Absolute difference between observed foreground counts and expected counts per bin.
        Calculated as:
            rsp_abs_bin[k] = foreground_histogram[k] - (background_histogram[k] * (n_f / n_b))
                             if n_b > 0
            rsp_abs_bin[k] = foreground_histogram[k] if n_b == 0
    d_max_abs : float
        Maximum absolute difference between the cumulative distributions of foreground
        and expected foreground.
        Equivalent to the Kolmogorov-Smirnov (KS) statistic.

    Notes
    -----
    - If n_b is zero, the expected foreground counts are undefined; in this case,
      rsp_abs_bin[k] equals foreground_histogram[k].
    - Ensure that background_histogram contains no negative counts and n_b > 0 for
      meaningful comparisons.

    Example
    -------
    >>> foreground_histogram = np.array([10, 20, 30, 40])
    >>> background_histogram = np.array([20, 20, 20, 20])
    >>> n_f = 100
    >>> n_b = 80
    >>> rsp_abs_bin, d_max_abs = compute_absolute_mode(foreground_histogram,
                                                       background_histogram,
                                                       n_f,
                                                       n_b)
    >>> rsp_abs_bin
    array([-15., -15., -15., -15.])
    >>> d_max_abs
    0.15
    """
    foreground_histogram = np.asarray(foreground_histogram)
    background_histogram = np.asarray(background_histogram)

    # Expected foreground counts based on background distribution
    # There will always be at least one background cell for there to be a foreground cell
    # as foreground cells are a subset of background cells
    expected_foreground = background_histogram * (n_f / n_b)

    # Calculate absolute differences per bin
    rsp_abs_bin = foreground_histogram - expected_foreground

    # Compute cumulative sums for CDFs
    if n_f > 0:
        cdf_f = np.cumsum(foreground_histogram) / n_f
        cdf_expected_foreground = np.cumsum(expected_foreground) / n_f
    else:
        cdf_f = np.zeros_like(foreground_histogram)
        cdf_expected_foreground = np.zeros_like(expected_foreground)

    # Compute area between CDFs using trapezoidal rule
    area_cdf_diff = np.trapezoid(
        np.abs(cdf_f - cdf_expected_foreground), dx=1.0 / len(cdf_f)
    )

    # Compute maximum absolute difference across CDFs
    d_max_abs = np.max(np.abs(cdf_f - cdf_expected_foreground))

    bin_width = 1.0 / len(cdf_f)
    area_cdf_diff = np.trapz(np.abs(cdf_f - cdf_expected_foreground), dx=bin_width)

    # Compute inversely complementary RSP Diff
    rsp_diff = 1.0 - area_cdf_diff

    return rsp_abs_bin, d_max_abs, rsp_diff


def compute_relative_mode(
    normalized_foreground_histogram: np.ndarray,
    normalized_background_histogram: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """


    Parameters
    ----------
    normalized_foreground_histogram : np.ndarray
        Normalized histogram (fractions) of foreground points within each
        angular bin of the scanning window.
    normalized_background_histogram : np.ndarray
        Normalized histogram (fractions) of background points within each
        angular bin of the scanning window.

    Returns
    -------
    rsp_rel_bin : np.ndarray
        Relative enrichment ratio per bin.
        Calculated as:
            rsp_rel_bin[k] = (normalized_foreground_histogram[k] / normalized_background_histogram[k])
            if normalized_background_histogram[k] > 0
            rsp_rel_bin[k] = 1.0 if normalized_background_histogram[k] == 0
    deviation_rel : np.ndarray
        Absolute deviation of the relative enrichment ratio from neutrality (1).
        Calculated as:
            deviation_rel[k] = |rsp_rel_bin[k] - 1|

    Notes
    -----
    - If normalized_background_histogram[k] is zero, the relative ratio is set to 1.0 to
      denote neutrality.
    - A ratio > 1 indicates enrichment of foreground relative to background in bin k.
    - A ratio < 1 indicates depletion of foreground relative to background in bin k.

    Example
    -------
    >>> normalized_foreground_histogram = np.array([0.1, 0.2, 0.3, 0.4])
    >>> normalized_background_histogram = np.array([0.2, 0.2, 0.2, 0.2])
    >>> rsp_rel_bin, deviation_rel = compute_relative_mode(normalized_foreground_histogram,
        normalized_background_histogram)
    >>> rsp_rel_bin
    array([0.5, 1.0, 1.5, 2.0])
    >>> deviation_rel
    array([0.5, 0. , 0.5, 1. ])
    """
    normalized_foreground_histogram = np.asarray(normalized_foreground_histogram)
    normalized_background_histogram = np.asarray(normalized_background_histogram)

    # Initialize rsp_rel_bin with ones (neutral)
    rsp_rel_bin = np.ones_like(normalized_foreground_histogram)

    # Identify bins where background fraction is greater than zero
    valid_bins = normalized_background_histogram > 0

    # Compute relative ratios where background is present
    rsp_rel_bin[valid_bins] = (
        normalized_foreground_histogram[valid_bins]
        / normalized_background_histogram[valid_bins]
    )

    # Compute deviation from neutrality
    deviation_rel = np.abs(rsp_rel_bin - 1)

    return rsp_rel_bin, deviation_rel
