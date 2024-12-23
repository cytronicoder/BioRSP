from typing import List, Union, Tuple

import numpy as np
import scanpy as sc


def define_foreground_and_background(
    adata: sc.AnnData,
    genes: Union[str, List[str]],
    threshold: Union[float, List[float]],
    mode: str = "any",
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define foreground and background sets of cells based on gene expression thresholds.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix containing scRNA-seq data.
    genes : str or List[str]
        The gene(s) to use for defining foreground cells.
    threshold : float or List[float]
        The expression threshold(s) for the gene(s). If multiple genes are provided,
        this should be a list of the same length as `genes`.
    mode : str, optional
        How to combine thresholds for multiple genes. Options are:
        - "any" (default): A cell is foreground if it meets the threshold for any gene.
        - "all": A cell is foreground only if it meets the threshold for all genes.
    verbose : int, optional
        Verbosity level. Default is `0`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Foreground indices: Indices of cells meeting the threshold criteria.
        - Background indices: Indices of all cells (equal to all available indices).

    Raises
    ------
    ValueError
        If `genes` and `threshold` lengths do not match (for multiple genes).
        If any specified gene is not found in the dataset.
        If an invalid `mode` is specified.

    Example
    -------
    >>> fg_indices, bg_indices = define_foreground_and_background(
    ...     adata, genes=["GeneA", "GeneB"], threshold=[1.5, 2.0], mode="any", verbose=1
    ... )
    Foreground cells identified: 150 / 1000.
    Background cells identified: 1000 / 1000 (all points).
    """
    # Ensure genes is a list
    if isinstance(genes, str):
        genes = [genes]
    if isinstance(threshold, (int, float)):
        threshold = [threshold] * len(genes)

    # Validate lengths
    if len(genes) != len(threshold):
        raise ValueError("Length of `genes` and `threshold` must match.")

    if verbose:
        print(
            f"Defining foreground cells using {len(genes)} gene(s) with thresholds {threshold}."
        )

    # Initialize an empty list to store expression data
    gene_expr_list = []

    for _, gene in enumerate(genes):
        if gene not in adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in the dataset.")

        # Extract expression data for the gene
        expr = adata[:, gene].X
        expr = np.asarray(expr).flatten()
        gene_expr_list.append(expr)

    gene_expr = np.vstack(gene_expr_list)  # Shape: (n_genes, n_cells)
    threshold = np.array(threshold).reshape(-1, 1)  # Shape: (n_genes, 1)

    # Apply thresholds
    if mode == "any":
        foreground_mask = np.any(gene_expr >= threshold, axis=0)
    elif mode == "all":
        foreground_mask = np.all(gene_expr >= threshold, axis=0)
    else:
        raise ValueError("Invalid `mode`. Use 'any' or 'all'.")

    fg_indices = np.where(foreground_mask)[0]
    bg_indices = np.arange(adata.n_obs)

    if verbose:
        print(f"Foreground cells identified: {len(fg_indices)} / {adata.n_obs}.")
        print(
            f"Background cells identified: {len(bg_indices)} / {adata.n_obs} (all points)."
        )

    return fg_indices, bg_indices
