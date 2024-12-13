import pandas as pd
import matplotlib.pyplot as plt


def load_dge(filepath: str, sep: str = "\t", index_col: int = 0) -> pd.DataFrame:
    """
    Load a Differential Gene Expression (DGE) matrix from a file.

    Parameters:
    - filepath (str): Path to the DGE file.
    - sep (str): Delimiter for the file (default: tab-separated).
    - index_col (int): Column to use as row labels (default: 0).

    Returns:
    - pd.DataFrame: Loaded DGE matrix.
    """
    try:
        dge_matrix = pd.read_csv(filepath, sep=sep, index_col=index_col)
        print(f"Loaded DGE matrix with shape: {dge_matrix.shape}")
        return dge_matrix
    except Exception as e:
        raise FileNotFoundError(f"Error reading the file at {filepath}: {e}") from e


def filter_cells_by_umi(
    dge_matrix: pd.DataFrame, threshold_umi: int, plot: bool = False
) -> pd.DataFrame:
    """
    Filter cells by UMI count.

    Parameters:
    - dge_matrix (pd.DataFrame): Gene expression data (rows = genes, columns = cells).
    - threshold_umi (int): Minimum UMI count threshold.
    - plot (bool): Whether to plot the UMI count histogram.

    Returns:
    - pd.DataFrame: Filtered DGE matrix.
    """
    umi_counts_per_cell = dge_matrix.sum(axis=0)

    if plot:
        plt.hist(umi_counts_per_cell, bins=50)
        plt.xlabel("UMI counts")
        plt.ylabel("Number of cells")
        plt.show()

    filtered_cells = umi_counts_per_cell[umi_counts_per_cell > threshold_umi].index
    return dge_matrix[filtered_cells]


def filter_genes_by_expression(
    dge_matrix: pd.DataFrame, threshold_gene: int
) -> pd.DataFrame:
    """
    Filter genes by expression count.

    Parameters:
    - dge_matrix (pd.DataFrame): Gene expression data.
    - threshold_gene (int): Minimum gene expression count threshold.

    Returns:
    - pd.DataFrame: Filtered DGE matrix.
    """
    gene_counts_per_cell = (dge_matrix > 0).sum(axis=1)
    filtered_genes = gene_counts_per_cell[gene_counts_per_cell > threshold_gene].index
    return dge_matrix.loc[filtered_genes]


def preprocess_dge_matrix(
    filepath: str, threshold_umi: int = 500, threshold_gene: int = 1, sep: str = "\t"
) -> pd.DataFrame:
    """
    Preprocess the DGE matrix by loading it and filtering cells and genes.

    Parameters:
    - filepath (str): Path to the DGE matrix file.
    - threshold_umi (int): Minimum UMI count threshold.
    - threshold_gene (int): Minimum gene expression count threshold.
    - sep (str): Delimiter for the file (default: tab-separated).

    Returns:
    - pd.DataFrame: Filtered DGE matrix.
    """
    # Load DGE matrix
    dge_matrix = load_dge(filepath, sep=sep)

    # Filter cells and genes
    dge_matrix = filter_cells_by_umi(dge_matrix, threshold_umi)
    dge_matrix = filter_genes_by_expression(dge_matrix, threshold_gene)
    print(f"Filtered DGE matrix shape: {dge_matrix.shape}")
    return dge_matrix
