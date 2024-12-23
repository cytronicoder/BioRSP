import numpy as np


def set_vantage_point(coords, vantage_point=None):
    """
    Set the vantage point for polar transformation.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates of the data points; typically the background points are used.
    vantage_point : np.ndarray, optional
        The coordinates of the vantage point. If not provided, the centroid of the points is used.

    Returns
    -------
    vantage_point : np.ndarray
        The coordinates of the vantage point.
    """

    if vantage_point is None:
        vantage_point = np.mean(coords, axis=0)
    return vantage_point


def transform_to_polar(coords, vantage_point):
    """
    Transform Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    coords : np.ndarray
        The Cartesian coordinates to transform.
    vantage_point : np.ndarray
        The coordinates of the vantage point.

    Returns
    -------
    r : np.ndarray
        The radial distance from the vantage point.
    theta : np.ndarray
        The angle in radians.
    """
    delta = coords - vantage_point
    r = np.sqrt((delta[:, 0]) ** 2 + (delta[:, 1]) ** 2)
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    theta = np.mod(theta, 2 * np.pi)
    return r, theta
