from typing import Dict, Any, Optional
import numpy as np

from .gen import generate_points
from .coord import cartesian_to_polar, within_angle
from .hist import compute_histogram
from .cdf import compute_cdf, compute_cdfs
from .diff import compute_diff

__all__ = [
    "generate_points",
    "cartesian_to_polar",
    "within_angle",
    "compute_histogram",
    "compute_cdf",
    "compute_cdfs",
    "compute_diff",
]


class Config:
    """
    Global configuration for biorsp.
    """

    def __init__(self):
        self.params = {
            "MODE": "absolute",  # 'absolute' or 'relative'
            "SCANNING_WINDOW": np.pi,  # Default scanning window in radians
            "RESOLUTION": 1000,  # Default resolution
        }

    def update(self, custom_params: Dict[str, Any]) -> None:
        """
        Updates the default parameters with custom values.

        Parameters:
        - custom_params (Dict[str, Any]): Dictionary of custom parameters to update.
        """
        self.params.update(custom_params)

    def get(self, key: str) -> Any:
        """
        Retrieves a parameter value by key.

        Parameters:
        - key (str): Parameter name.

        Returns:
        - The value of the parameter.
        """
        return self.params.get(key, None)


_config = Config()


def init(custom_params: Optional[Dict[str, Any]] = None) -> None:
    """
    Initializes configuration parameters.

    Parameters:
    - custom_params (Optional[Dict[str, Any]]): Dictionary of custom parameters.
    """
    if custom_params:
        _config.update(custom_params)


def get_param(key: str) -> Any:
    """
    Retrieves a configuration parameter.

    Parameters:
    - key (str): Parameter name.

    Returns:
    - The value of the parameter.
    """
    return _config.get(key)
