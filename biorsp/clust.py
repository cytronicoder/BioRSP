import numpy as np
from sklearn.cluster import DBSCAN


def compute_dbscan(
    embedding_results: np.ndarray, eps: float = 4.0, min_samples: int = 50
) -> np.ndarray:
    """
    Run DBSCAN on embedding results.

    Parameters:
    - embedding_results (np.ndarray): 2D embedding coordinates (cells x components).
    - eps (float): Epsilon parameter for DBSCAN.
    - min_samples (int): Minimum samples for DBSCAN.

    Returns:
    - np.ndarray: DBSCAN cluster labels.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(embedding_results)
    dbscan_labels += 1  # Shift labels to make 0 a valid cluster
    return dbscan_labels
