"""Compute the RSP diff value based on foreground and background CDFs."""

import numpy as np


def compute_diff(
    fg_cdf: np.ndarray,
    bg_cdf: np.ndarray,
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
    if fg_cdf.ndim != 1:
        raise ValueError("fg_cdf must be a one-dimensional array.")
    if bg_cdf.ndim != 1:
        raise ValueError("bg_cdf must be a one-dimensional array.")
    if fg_cdf.shape != bg_cdf.shape:
        raise ValueError("fg_cdf and bg_cdf must have the same shape.")

    # Assuming the CDFs are defined over [0, 1] with equal spacing:
    n = len(fg_cdf)
    if n < 2:
        raise ValueError("CDF arrays must contain at least two points.")

    # Compute spacing between CDF points
    delta_x = 1.0 / (n - 1)

    # Compute union and intersection
    union_cdf = np.maximum(fg_cdf, bg_cdf)
    intersection_cdf = np.minimum(fg_cdf, bg_cdf)

    # Integrate by summation
    union_area = np.sum(union_cdf) * delta_x
    intersection_area = np.sum(intersection_cdf) * delta_x

    rsp_diff = union_area - intersection_area

    if mode == "relative":
        # Avoid division by zero if union_area is zero
        if union_area != 0:
            rsp_diff /= union_area

    return rsp_diff


def compute_rsp_area(diffs: np.ndarray) -> float:
    """
    Compute the area enclosed by the RSP diff values.

    Parameters:
    - diffs (np.ndarray): 1D array of radii (e.g., diff values).
    - resolution (float): Angular resolution in radians.

    Returns:
    - float: Enclosed area.
    """
    n = len(diffs)
    delta_theta = 2 * np.pi / n
    area = 0.5 * np.sum(diffs**2 * delta_theta)
    return area


def compute_deviation_score(rsp_area, differences):
    """
    Calculate the deviation score based on the RSP area.

    Parameters:
    - rsp_area: The calculated RSP area.
    - differences: Numpy array of differences between the foreground and background CDFs.

    Returns:
    - deviation_score: The calculated deviation score.
    """
    radius = np.sqrt(rsp_area / np.pi)
    delta_theta = 2 * np.pi / len(differences)

    intersection_area = np.sum(np.minimum(differences, radius)) * delta_theta
    total_area = np.sum(np.maximum(differences, radius)) * delta_theta

    if rsp_area != 0 and total_area != 0:
        deviation_score = (total_area - intersection_area) / total_area
        return deviation_score
    return 0
