from typing import Optional

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP


def compute_tsne(
    dge_matrix: pd.DataFrame,
    n_components: int = 2,
    random_state: Optional[int] = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    """
    Perform t-SNE on the filtered DGE matrix.

    Parameters:
    - dge_matrix (pd.DataFrame): Filtered DGE matrix.
    - n_components (int): Number of embedding dimensions.
    - random_state (Optional[int]): Random state for reproducibility.
    - perplexity (float): Perplexity parameter for t-SNE.

    Returns:
    - np.ndarray: t-SNE embedding coordinates (cells x n_components).
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
    )
    return tsne.fit_transform(dge_matrix.T)


def compute_umap(
    dge_matrix: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Perform UMAP on the filtered DGE matrix.

    Parameters:
    - dge_matrix (pd.DataFrame): Filtered DGE matrix.
    - n_neighbors (int): Number of neighbors for UMAP.
    - min_dist (float): Minimum distance for UMAP.
    - random_state (Optional[int]): Random state for reproducibility.

    Returns:
    - np.ndarray: UMAP embedding coordinates (cells x n_components).
    """
    umap_reducer = UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    )
    return umap_reducer.fit_transform(dge_matrix.T)
