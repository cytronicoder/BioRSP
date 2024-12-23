from typing import Optional

import numpy as np
import scanpy as sc
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import umap


def load_data(
    path: str,
    ftype: str = "txt",
    delimiter: Optional[str] = None,
    transpose: bool = False,
    obs_col: str = "obs",
    var_col: str = "var",
    first_column_names: bool = True,
    verbose: int = 0,
) -> sc.AnnData:
    """
    Load scRNA-seq data into an AnnData object.

    Parameters
    ----------
    path : str
        File path to the scRNA-seq data.
    ftype : str, optional
        File type of the input data. Supported types are 'txt', 'csv', and 'h5ad'.
        Default is 'txt'.
    delimiter : str, optional
        Delimiter used in 'txt' or 'csv' files. If `None`, auto-detected. Default is `None`.
    transpose : bool, optional
        Transpose data matrix upon loading. Default is `False`.
    obs_col : str, optional
        Column name to use as cell identifiers in `.obs`. Default is "obs".
    var_col : str, optional
        Column name to use as gene identifiers in `.var`. Default is "var".
    first_column_names : bool, optional
        Indicates whether the first column of the file contains row names. Default is `True`.
    verbose : int, optional
        Verbosity level. Default is `0`.

    Returns
    -------
    AnnData
        Annotated data matrix containing the loaded scRNA-seq data.

    Raises
    ------
    ValueError
        If an unsupported file type is specified.

    Example
    -------
    >>> adata = load_data("data/dge.txt", ftype="txt", verbose=1)
    """
    if verbose:
        print(f"Loading data from {path} as {ftype} file...")

    if ftype == "txt":
        data = sc.read_csv(
            path, delimiter=delimiter or "\t", first_column_names=first_column_names
        )
    elif ftype == "csv":
        data = sc.read_csv(
            path, delimiter=delimiter or ",", first_column_names=first_column_names
        )
    elif ftype == "h5ad":
        data = sc.read_h5ad(path)
    else:
        raise ValueError("Unsupported file type. Use 'txt', 'csv', or 'h5ad'.")

    if transpose:
        if verbose:
            print("Transposing data matrix.")
        data = data.T

    if obs_col != "obs" and obs_col in data.obs.columns:
        data.obs_names = data.obs[obs_col]
    elif obs_col != "obs":
        raise ValueError(f"obs_col '{obs_col}' not found in .obs.")

    if var_col != "var" and var_col in data.var.columns:
        data.var_names = data.var[var_col]
    elif var_col != "var":
        raise ValueError(f"var_col '{var_col}' not found in .var.")

    if verbose:
        print(f"Data loaded with shape: {data.shape}.")
    return data


def filter_data(
    adata: sc.AnnData,
    mito_prefix: str = "MT-",
    mito_frac: float = 0.1,
    min_genes: int = 200,
    min_cells: int = 3,
    max_counts: Optional[int] = None,
    min_counts: Optional[int] = None,
    max_genes: Optional[int] = None,
    verbose: int = 0,
) -> sc.AnnData:
    """
    Perform quality control filtering on scRNA-seq data.

    Parameters
    ----------
    adata : AnnData
        Input annotated data matrix.
    mito_prefix : str, optional
        Prefix used to identify mitochondrial genes. Default is "MT-".
    mito_frac : float, optional
        Maximum fraction of counts attributed to mitochondrial genes per cell. Default is 0.1.
    min_genes : int, optional
        Minimum number of genes expressed in a cell for it to be retained. Default is 200.
    min_cells : int, optional
        Minimum number of cells that must express a gene for it to be retained. Default is 3.
    max_counts : int, optional
        Maximum total counts per cell. Default is `None`.
    min_counts : int, optional
        Minimum total counts per cell. Default is `None`.
    max_genes : int, optional
        Maximum number of genes expressed per cell. Default is `None`.
    verbose : int, optional
        Verbosity level. Default is `0`.

    Returns
    -------
    AnnData
        Filtered annotated data matrix after applying quality control criteria.

    Example
    -------
    >>> adata_filtered = filter_data(adata, mito_frac=0.05, min_genes=500, verbose=1)
    """
    if verbose:
        print("Performing quality control filtering...")

    mito_genes = adata.var_names.str.startswith(mito_prefix)
    if not np.any(mito_genes) and verbose:
        print(f"No mitochondrial genes found with prefix '{mito_prefix}'.")
        adata.obs["mito_counts"] = 0
    else:
        adata.obs["mito_counts"] = adata[:, mito_genes].X.sum(axis=1)

    adata.obs["n_genes"] = (adata.X > 0).sum(axis=1)
    adata.obs["total_counts"] = adata.X.sum(axis=1)
    adata.obs["mito_frac"] = adata.obs["mito_counts"] / adata.obs["total_counts"]

    cell_mask = (adata.obs["mito_frac"] <= mito_frac) & (
        adata.obs["n_genes"] >= min_genes
    )
    if max_counts is not None:
        cell_mask &= adata.obs["total_counts"] <= max_counts
    if min_counts is not None:
        cell_mask &= adata.obs["total_counts"] >= min_counts
    if max_genes is not None:
        cell_mask &= adata.obs["n_genes"] <= max_genes

    filtered_cells = adata[cell_mask].copy()
    gene_mask = (filtered_cells.X > 0).sum(axis=0) >= min_cells
    filtered_data = filtered_cells[:, gene_mask].copy()

    if verbose:
        print(f"Filtered cells: {filtered_cells.n_obs} / {adata.n_obs}")
        print(f"Filtered genes: {filtered_data.n_vars} / {adata.n_vars}")

    return filtered_data


def dim_reduce(
    adata: sc.AnnData,
    method: str = "umap",
    embedding_key: Optional[str] = None,
    # UMAP-specific parameters
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    umap_random_state: Optional[int] = 42,
    # t-SNE-specific parameters
    tsne_n_components: int = 2,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: float = 200.0,
    tsne_early_exaggeration: float = 12.0,
    tsne_n_iter: int = 1000,
    tsne_init: str = "random",
    tsne_random_state: Optional[int] = 42,
    verbose: int = 0,
) -> sc.AnnData:
    """
    Perform dimensionality reduction using t-SNE or UMAP.

    Parameters
    ----------
    adata : AnnData
        Input annotated data matrix.
    method : str, optional
        Dimensionality reduction method to use. Options are "umap" or "tsne".
        Default is "umap".
    embedding_key : str, optional
        Key to store the embedding coordinates in `.obsm`. If not provided,
        defaults to `"X_{method}"`.

    UMAP Parameters
    ---------------
    umap_n_neighbors : int, optional
        Number of neighbors to consider in UMAP. Default is 15.
    umap_min_dist : float, optional
        Minimum distance between points in UMAP embedding. Default is 0.1.
    umap_metric : str, optional
        Metric for UMAP distance computation. Default is "euclidean".
    umap_random_state : int, optional
        Random seed for reproducibility in UMAP. Default is 42.

    t-SNE Parameters
    -----------------
    tsne_n_components : int, optional
        Number of dimensions for the t-SNE embedding. Default is 2.
    tsne_perplexity : float, optional
        Perplexity for t-SNE. Default is 30.0.
    tsne_learning_rate : float, optional
        Learning rate for t-SNE. Default is 200.0.
    tsne_early_exaggeration : float, optional
        Early exaggeration factor for t-SNE. Default is 12.0.
    tsne_n_iter : int, optional
        Number of iterations for t-SNE optimization. Default is 1000.
    tsne_init : str, optional
        Initialization method for t-SNE. Options are "random" or "pca". Default is "random".
    tsne_random_state : int, optional
        Random seed for reproducibility in t-SNE. Default is 42.

    verbose : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    AnnData
        Annotated data matrix with embedding coordinates saved to `.obsm[embedding_key]`.

    Raises
    ------
    ValueError
        If an invalid `method` is specified.

    Example
    -------
    >>> adata_embedded = dim_reduce(
    ...     adata_filtered, method="umap", n_neighbors=10, min_dist=0.05, verbose=1
    ... )
    Performing dimensionality reduction using UMAP...
    Embedding saved to `.obsm['X_umap']` with shape: (1000, 2).
    """
    if embedding_key is None:
        embedding_key = f"X_{method.lower()}"

    if verbose:
        print(f"Performing dimensionality reduction using {method.upper()}...")

    data = adata.X

    if method.lower() == "umap":
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            n_components=2,
            metric=umap_metric,
            random_state=umap_random_state,
            verbose=verbose,
        )
    elif method.lower() == "tsne":
        reducer = TSNE(
            n_components=tsne_n_components,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            early_exaggeration=tsne_early_exaggeration,
            n_iter=tsne_n_iter,
            init=tsne_init,
            random_state=tsne_random_state,
            verbose=verbose,
        )
    else:
        raise ValueError("Invalid method specified. Use 'umap' or 'tsne'.")

    adata.obsm[embedding_key] = reducer.fit_transform(data)

    if verbose:
        print(
            f"Dimensionality reduction completed. Embedding saved to `.obsm['{embedding_key}']` "
            f"with shape: {adata.obsm[embedding_key].shape}"
        )

    return adata


def dbscan_cluster(
    adata: sc.AnnData,
    method: Optional[str] = None,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    algorithm: str = "auto",
    verbose: int = 0,
    cluster_key: str = "dbscan_cluster",
) -> sc.AnnData:
    """
    Perform DBSCAN clustering on a specified embedding within an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix containing scRNA-seq data.
    method : str, optional
        Dimensionality reduction method used for embedding. Options are "umap" or "tsne".
        If not specified, the function will check for embeddings in `.obsm` and raise an error
        if multiple or no embeddings are found.
    eps : float, optional
        The maximum distance between two samples for them to be considered as in
        the same neighborhood. Default is 0.5.
    min_samples : int, optional
        The number of samples in a neighborhood for a point to be considered as
        a core point. Default is 5.
    metric : str, optional
        The metric to use when calculating distance. Default is "euclidean".
    algorithm : str, optional
        Algorithm to compute the neighborhood. Default is "auto".
    verbose : int, optional
        Verbosity level. Default is 0.
    cluster_key : str, optional
        The key under which to store the cluster labels in `.obs`. Default is "dbscan_cluster".

    Returns
    -------
    AnnData
        The input AnnData object with cluster labels added to `.obs[cluster_key]`.

    Raises
    ------
    KeyError
        If no embedding is found in `.obsm` or if multiple embeddings are found but
        no `method` is specified.

    Example
    -------
    >>> adata_clustered = dbscan_cluster(
    ...     adata_embedded, method="umap", eps=0.5, min_samples=5, verbose=1
    ... )
    """
    if method is None:
        available_embeddings = [
            key for key in adata.obsm.keys() if key.startswith("X_")
        ]
        if len(available_embeddings) == 0:
            raise KeyError(
                "No embedding found in `.obsm`. "
                "Please run `dim_reduce` first to generate an embedding."
            )
        elif len(available_embeddings) > 1:
            raise KeyError(
                f"Multiple embeddings found: {', '.join(available_embeddings)}. "
                "Please specify the method (e.g., 'umap' or 'tsne') to select the embedding."
            )
        else:
            method = available_embeddings[0].replace("X_", "")

    embedding_key = f"X_{method}"
    if embedding_key not in adata.obsm.keys():
        raise KeyError(
            f"Embedding for method '{method}' not found in `.obsm`. "
            "Please run `dim_reduce` with method='{method}' first."
        )

    if verbose:
        print(f"Starting DBSCAN clustering using embedding '{embedding_key}'...")

    embedding = adata.obsm[embedding_key]

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
    )

    cluster_labels = dbscan.fit_predict(embedding)
    adata.obs[cluster_key] = cluster_labels

    if verbose:
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(
            f"DBSCAN clustering completed. {n_clusters} clusters found, {n_noise} noise points."
        )

    return adata
