from typing import Tuple, Union
import numpy as np


def within_window(
    theta: Union[float, np.ndarray],
    center_theta: float,
    window_width: float,
    verbosity: int = 0,
) -> Union[bool, np.ndarray]:
    """
    Check if an angle or array of angles lies within a specified angular window.

    Parameters
    ----------
    theta : float or np.ndarray
        The angle(s) in radians to check. Can be a single float or an array.
    center_theta : float
        The center of the angular window in radians.
    window_width : float
        The width of the angular window in radians.
    verbosity : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    bool or np.ndarray
        - `True` if a single angle is within the window.
        - Boolean array indicating if each angle is within the window for arrays.

    Raises
    ------
    ValueError
        If `window_width` is not within the range (0, 2π].

    Example
    -------
    >>> within_window(np.pi / 3, np.pi / 2, np.pi / 4)
    True
    >>> angles = np.array([0, np.pi / 4, np.pi / 2, 3*np.pi / 4, np.pi])
    >>> within_window(angles, np.pi / 2, np.pi / 2)
    array([False,  True,  True, False, False])
    """
    if not 0 < window_width <= 2 * np.pi:
        raise ValueError(
            "`window_width` must be greater than 0 and at most 2π radians."
        )

    if verbosity:
        print(
            f"Checking angles within window centered at {center_theta} radians "
            f"with width {window_width} radians."
        )

    theta_difference = np.abs((theta - center_theta + np.pi) % (2 * np.pi) - np.pi)
    is_within_window = theta_difference <= (window_width / 2)

    if verbosity and isinstance(theta, np.ndarray):
        print(
            f"Angles checked: {len(theta)}. Within window: {np.sum(is_within_window)}."
        )

    return is_within_window


def calculate_histograms(
    foreground_angles: np.ndarray,
    background_angles: np.ndarray,
    center_theta: float,
    scanning_window: float = np.pi,
    resolution: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes foreground and background histograms within a specified angular window
    centered at a given central angle. Handles wraparound by shifting angles relative
    to the central angle.
    """
    if not 0 < scanning_window <= 2 * np.pi:
        raise ValueError("`scanning_window` must be within (0, 2π] radians.")
    if not 0 <= center_theta < 2 * np.pi:
        raise ValueError("`center_theta` must be within [0, 2π) radians.")

    def shift_angles(angles: np.ndarray, central: float) -> np.ndarray:
        return (angles - central + np.pi) % (2 * np.pi) - np.pi

    # Shift angles relative to center_theta
    shifted_foreground_angles = shift_angles(foreground_angles, center_theta)
    shifted_background_angles = shift_angles(background_angles, center_theta)

    # Define window boundaries
    window_start = -scanning_window / 2
    window_end = scanning_window / 2

    # Select angles within the window
    foreground_within_window = (
        (shifted_foreground_angles >= window_start)
        & (shifted_foreground_angles < window_end)
        if len(shifted_foreground_angles) > 0
        else np.array([])
    )

    background_within_window = (shifted_background_angles >= window_start) & (
        shifted_background_angles < window_end
    )

    # Handle empty foreground
    if len(foreground_within_window) == 0:
        foreground_histogram = np.zeros(int(resolution * scanning_window / (2 * np.pi)))
    else:
        foreground_histogram, _ = np.histogram(
            shifted_foreground_angles[foreground_within_window],
            bins=np.linspace(window_start, window_end, resolution + 1),
        )

    background_histogram, _ = np.histogram(
        shifted_background_angles[background_within_window],
        bins=np.linspace(window_start, window_end, resolution + 1),
    )

    bin_edges = np.linspace(window_start, window_end, resolution + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return foreground_histogram, background_histogram, bin_edges, bin_centers


def normalize_histograms(
    foreground_histogram: np.ndarray, background_histogram: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes histograms.
    """
    normalized_foreground_histogram = foreground_histogram / np.sum(
        foreground_histogram
    )
    normalized_background_histogram = background_histogram / np.sum(
        background_histogram
    )

    return normalized_foreground_histogram, normalized_background_histogram


def calculate_cdfs(
    normalized_foreground_histogram: np.ndarray,
    normalized_background_histogram: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate cumulative distribution functions (CDFs) for foreground and background histograms.

    Parameters
    ----------
    normalized_foreground_histogram : np.ndarray
        Normalized histogram counts of foreground points within each angular bin of the
        scanning window.
    normalized_background_histogram : np.ndarray
        Normalized histogram counts of background points within each angular bin of the
        scanning window.

    Returns
    -------
    foreground_cdf : np.ndarray
        Cumulative distribution function (CDF) for foreground points.
    background_cdf : np.ndarray
        Cumulative distribution function (CDF) for background points.
    """
    foreground_cdf = np.cumsum(normalized_foreground_histogram)
    background_cdf = np.cumsum(normalized_background_histogram)

    return foreground_cdf, background_cdf
