"""Histogram computation for foreground and background angles."""

from typing import Tuple
import numpy as np


def compute_histogram(
    foreground: np.ndarray, background: np.ndarray, bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histograms for foreground and background angles based on angular bins.

    Parameters:
    - foreground (np.ndarray): Array of angles in radians for the foreground points.
    - background (np.ndarray): Array of angles in radians for the background points.
    - bins (np.ndarray): Array of bin edges for angular bins.

    Returns:
    - foreground_histogram (np.ndarray): Array representing the foreground histogram counts.
    - background_histogram (np.ndarray): Array representing the background histogram counts.
    """
    foreground_histogram, _ = np.histogram(foreground, bins=bins)
    background_histogram, _ = np.histogram(background, bins=bins)
    return foreground_histogram, background_histogram
