"""Utility functions for working with Cartesian and polar coordinates."""

from typing import Tuple, Union
import numpy as np


def cartesian_to_polar(
    data: np.ndarray, center: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts Cartesian coordinates to polar coordinates relative to a given center.

    Parameters:
    - data (np.ndarray): Array of shape (n_points, 2), where each row is (x, y).
    - center (Tuple[float, float]): (x, y) coordinates representing the new origin.

    Returns:
    - radii (np.ndarray): Array of radii from the center.
    - angles (np.ndarray): Array of angles in radians from the center.
    """
    x_c, y_c = center
    x_shifted = data[:, 0] - x_c
    y_shifted = data[:, 1] - y_c

    radii = np.sqrt(x_shifted**2 + y_shifted**2)
    angles = np.arctan2(y_shifted, x_shifted)

    return radii, angles


def within_angle(
    angle: Union[float, np.ndarray], center: float, width: float
) -> Union[bool, np.ndarray]:
    """
    Checks if an angle or array of angles lies within a specified angular window.

    Parameters:
    - angle (float or np.ndarray): The angle(s) in radians to check.
    - center (float): The center of the angular window in radians.
    - width (float): The width of the angular window in radians.

    Returns:
    - bool or np.ndarray:
        - `True` if a single angle is within the window.
        - Boolean array indicating if each angle is within the window for arrays.

    Raises:
    - ValueError: If `width` is not within the range (0, 2π].

    Example:
    >>> within_angle(np.pi / 3, np.pi / 2, np.pi / 4)
    True
    >>> angles = np.array([0, np.pi / 4, np.pi / 2, 3*np.pi / 4, np.pi])
    >>> within_angle(angles, np.pi / 2, np.pi / 2)
    array([False,  True,  True, False, False])
    """
    if not (0 < width <= 2 * np.pi):
        raise ValueError("width must be greater than 0 and at most 2*pi radians.")

    difference = np.abs((angle - center + np.pi) % (2 * np.pi) - np.pi)
    return difference <= (width / 2)
