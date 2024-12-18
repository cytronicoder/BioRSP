"""Compute the RSP diff value based on foreground and background CDFs."""

import numpy as np


def compute_diff(
    fg_cdf: np.ndarray,
    bg_cdf: np.ndarray,
    mode: str = "absolute",
    epsilon: float = 1e-9,
) -> float:
    """
    Compute the RSP diff value based on foreground and background CDFs.

    Modes:
    - "absolute": Use the integral of the absolute differences.
    - "relative": Normalize differences by bg_cdf + epsilon to emphasize differences
                  where background is small.

    Parameters:
    - fg_cdf (np.ndarray): 1D array representing the foreground CDF.
    - bg_cdf (np.ndarray): 1D array representing the background CDF.
    - mode (str): 'absolute' or 'relative' for the type of RSP diff calculation.
    - epsilon (float): A small value to avoid division by zero in relative mode.

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

    n = len(fg_cdf)
    if n < 2:
        raise ValueError("CDF arrays must contain at least two points.")

    delta_x = 1.0 / (n - 1)

    # Compute absolute differences
    abs_diff = np.abs(fg_cdf - bg_cdf)

    if mode == "absolute":
        # Integrate absolute differences
        rsp_diff = np.sum(abs_diff) * delta_x
    elif mode == "relative":
        # Normalize differences by bg_cdf + epsilon
        rel_diff = abs_diff / (bg_cdf + epsilon)
        rsp_diff = np.sum(rel_diff) * delta_x
    else:
        raise ValueError("mode must be either 'absolute' or 'relative'")

    return rsp_diff


def compute_rsp_area(diffs: np.ndarray) -> float:
    """
    Compute the area enclosed by the RSP diff values.

    Parameters:
    - diffs (np.ndarray): 1D array of radii (e.g., diff values).

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
