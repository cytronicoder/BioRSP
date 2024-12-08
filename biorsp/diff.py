"""Compute the RSP diff value based on foreground and background CDFs."""

import numpy as np


def compute_diff(
    fg_cdf: np.ndarray,
    bg_cdf: np.ndarray,
    bin_edges: np.ndarray,
    mode: str = "absolute",
) -> float:
    """
    Compute the RSP diff value based on foreground and background CDFs.

    RSP_diff = union_area - intersection_area

    Where:
    - union_area is the integral (sum) of the maximum of fg_cdf and bg_cdf across all bins.
    - intersection_area is the integral (sum) of the minimum of fg_cdf and bg_cdf across all bins.

    Parameters:
    - fg_cdf (np.ndarray): 1D array representing the foreground CDF.
    - bg_cdf (np.ndarray): 1D array representing the background CDF.
    - bin_edges (np.ndarray): 1D array of bin edges used to compute the CDFs.
    - mode (str): 'absolute' or 'relative' to specify the type of RSP diff calculation.

    Returns:
    - float: The computed RSP diff value.

    Raises:
    - TypeError: If any input is not a NumPy array.
    - ValueError: If input arrays have incompatible shapes.
    """
    if not isinstance(fg_cdf, np.ndarray):
        raise TypeError("fg_cdf must be a NumPy array.")
    if not isinstance(bg_cdf, np.ndarray):
        raise TypeError("bg_cdf must be a NumPy array.")
    if not isinstance(bin_edges, np.ndarray):
        raise TypeError("bin_edges must be a NumPy array.")
    if fg_cdf.ndim != 1:
        raise ValueError("fg_cdf must be a one-dimensional array.")
    if bg_cdf.ndim != 1:
        raise ValueError("bg_cdf must be a one-dimensional array.")
    if bin_edges.ndim != 1:
        raise ValueError("bin_edges must be a one-dimensional array.")
    if len(bin_edges) != len(fg_cdf) + 1:
        raise ValueError(
            "Length of bin_edges must be one more than length of fg_cdf and bg_cdf."
        )

    bin_widths = np.diff(bin_edges)
    union_cdf = np.maximum(fg_cdf, bg_cdf)
    union_area = np.sum(union_cdf * bin_widths)

    intersection_cdf = np.minimum(fg_cdf, bg_cdf)
    intersection_area = np.sum(intersection_cdf * bin_widths)

    rsp_diff = union_area - intersection_area

    if mode == "relative":
        rsp_diff /= union_area

    return rsp_diff
