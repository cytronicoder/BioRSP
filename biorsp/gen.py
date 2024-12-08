"""Simulate background and foreground points in a 2D space."""

from typing import List, Tuple, Optional
import numpy as np


def generate_points(
    bg_size: int = 10000,
    coverage: float = 0.1,
    distribution: str = "uniform",
    centroids: Optional[List[Tuple[float, float]]] = None,
    spread: float = 0.2,
    noise: float = 0.05,
    retries: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates background and foreground points in a 2D space.

    Background points represent uniformly distributed cells within a unit circle,
    while foreground points simulate cells with gene expression above a certain
    threshold, optionally clustered around specified centroids.

    Parameters:
    - bg_size (int): Total number of background points (cells) to generate.
    - coverage (float): Ratio of foreground points (expressed cells) to background points.
    - distribution (str): Distribution for foreground points ('uniform' or 'clustered').
    - centroids (Optional[List[Tuple[float, float]]]): Centroids for clusters.
    - spread (float): Spread (radius) around each centroid for clustered points.
    - noise (float): Standard deviation of Gaussian noise for cluster points.
    - retries (int): Maximum retries for generating valid cluster points.
    - seed (Optional[int]): Random seed for reproducibility.

    Returns:
    - background_points (np.ndarray): Array of shape (background, 2) with background points.
    - foreground_points (np.ndarray): Array of shape (foreground, 2) with foreground points.
    """
    if centroids is None:
        centroids = [(0.0, 0.0)]

    if seed is not None:
        np.random.seed(seed)

    # Simulate background points uniformly within a unit circle
    angles_bg = np.random.uniform(0, 2 * np.pi, bg_size)
    radii_bg = np.sqrt(np.random.uniform(0, 1, bg_size))
    x_bg = radii_bg * np.cos(angles_bg)
    y_bg = radii_bg * np.sin(angles_bg)
    background_points = np.vstack((x_bg, y_bg)).T

    # Calculate the number of foreground points
    foreground = int(bg_size * coverage)
    foreground_points = []

    if distribution == "uniform":
        indices = np.random.choice(
            background_points.shape[0], foreground, replace=False
        )
        foreground_points = background_points[indices]

    elif distribution == "clustered":
        if not centroids:
            raise ValueError(
                "At least one centroid must be provided for clustered distribution."
            )

        clusters = len(centroids)
        points_per_cluster = foreground // clusters
        extra_points = foreground % clusters

        for i, centroid in enumerate(centroids):
            points_for_this_cluster = points_per_cluster + (
                1 if i < extra_points else 0
            )
            attempts = 0
            cluster_points = []

            while len(cluster_points) < points_for_this_cluster and attempts < retries:
                remaining = points_for_this_cluster - len(cluster_points)
                angles_cluster = np.random.uniform(0, 2 * np.pi, remaining)
                radii_cluster = spread * np.sqrt(np.random.uniform(0, 1, remaining))
                x_cluster = centroid[0] + radii_cluster * np.cos(angles_cluster)
                y_cluster = centroid[1] + radii_cluster * np.sin(angles_cluster)
                raw_points = np.vstack((x_cluster, y_cluster)).T

                # Add Gaussian noise
                noise_matrix = np.random.normal(0, noise, raw_points.shape)
                noisy_points = raw_points + noise_matrix

                # Filter points outside the unit circle
                distances = np.linalg.norm(noisy_points, axis=1)
                valid_points = noisy_points[distances <= 1.0]
                cluster_points.extend(valid_points.tolist())
                attempts += 1

            foreground_points.extend(cluster_points[:points_for_this_cluster])

        # Fill remaining foreground points if clusters are incomplete
        if len(foreground_points) < foreground:
            remaining = foreground - len(foreground_points)
            additional_indices = np.random.choice(
                background_points.shape[0], remaining, replace=False
            )
            foreground_points.extend(background_points[additional_indices].tolist())

    else:
        raise ValueError("Invalid distribution type. Choose 'uniform' or 'clustered'.")

    foreground_points = np.array(foreground_points)

    return background_points, foreground_points
