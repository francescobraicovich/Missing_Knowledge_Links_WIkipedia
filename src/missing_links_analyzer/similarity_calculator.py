"""
Module for calculating node similarity in a graph using Jaccard coefficient.

This module provides functions to compute:
- Common neighbors between nodes from an adjacency matrix.
- The size of the union of neighbor sets for pairs of nodes.
- Jaccard similarity coefficient based on common and total unique neighbors.

It also includes a facade function `calculate_jaccard_similarity` to orchestrate 
these calculations directly from a NetworkX graph.
"""

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix # csc_matrix not strictly used in final version but kept from original
import networkx as nx
import logging
from threadpoolctl import threadpool_limits 

logger = logging.getLogger(__name__)

def get_common_neighbors(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the number of common neighbors between all pairs of nodes in a graph.

    Assumes an undirected graph where `adj_matrix` is symmetric and binary.
    The common neighbors between node `i` and `j` are nodes `k` such that
    an edge `(i,k)` and `(j,k)` both exist. This is `(adj_matrix @ adj_matrix)[i,j]`.

    Args:
        adj_matrix (np.ndarray): A 2D NumPy array representing the symmetric 
                                 adjacency matrix of the graph.

    Returns:
        np.ndarray: A 2D NumPy array `C` where `C[i,j]` is the number of
                    common neighbors between node `i` and node `j`.
                    The diagonal `C[i,i]` (which would be the degree of node `i`)
                    is set to 0. This is because common neighbors are typically
                    considered for distinct pairs, and J(i,i) = 1 is handled separately.
    """
    if not isinstance(adj_matrix, np.ndarray) or adj_matrix.ndim != 2:
        raise ValueError("adj_matrix must be a 2D NumPy array.")
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adj_matrix must be square.")

    logger.debug(f"Calculating common neighbors for matrix of shape {adj_matrix.shape}.")
    # Original code used adj_matrix_csc.dot(adj_matrix_csr). 
    # For a symmetric adj_matrix, adj_matrix_csr.dot(adj_matrix_csr) is equivalent and common.
    # This computes A*A, where A is the adjacency matrix. (A*A)[i,j] is the number of paths of length 2.
    adj_matrix_csr = csr_matrix(adj_matrix.astype(int)) # Use int for safety
    
    common_neighbors_matrix = adj_matrix_csr.dot(adj_matrix_csr)
    
    common_neighbors_matrix.setdiag(0) # Set diagonal to 0
    
    logger.debug("Common neighbors calculation complete.")
    return common_neighbors_matrix.toarray()


def get_total_neighbors(adj_matrix: np.ndarray, common_neighbors_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the size of the union of neighbor sets for each pair of nodes.

    For nodes `i` and `j`, this is `|N(i) U N(j)| = |N(i)| + |N(j)| - |N(i) intersect N(j)|`.
    `|N(i)|` is the degree of node `i`.
    `|N(i) intersect N(j)|` is `common_neighbors_matrix[i,j]`.

    Args:
        adj_matrix (np.ndarray): A 2D NumPy array representing the symmetric 
                                 adjacency matrix of the graph.
        common_neighbors_matrix (np.ndarray): A 2D NumPy array where `C[i,j]` is the
                                              number of common neighbors between `i` and `j`.

    Returns:
        np.ndarray: A 2D NumPy array `U` where `U[i,j]` is `|N(i) U N(j)|`.
                    The diagonal `U[i,i]` is set to 1. This is a convention to help
                    avoid 0/0 in `get_jaccard_coefficient` when `common_neighbors_matrix[i,i]`
                    is 0. The J(i,i)=1 case is handled by the facade.
    """
    if adj_matrix.shape != common_neighbors_matrix.shape:
        raise ValueError("adj_matrix and common_neighbors_matrix must have the same shape.")

    logger.debug("Calculating total unique neighbors (union of neighbor sets).")
    # Degree of each node: sum of connections along rows (or columns if symmetric)
    degrees = np.sum(adj_matrix, axis=1).astype(int) 

    # |N(i)| + |N(j)| for each pair (i,j)
    sum_of_degrees_matrix = np.add.outer(degrees, degrees)

    # |N(i) U N(j)| = |N(i)| + |N(j)| - |N(i) intersect N(j)|
    total_neighbors_union_matrix = sum_of_degrees_matrix - common_neighbors_matrix
    
    # Ensure matrix values are not negative (can happen if common_neighbors_matrix was not calculated correctly
    # or if adj_matrix was not binary/symmetric as expected).
    np.maximum(total_neighbors_union_matrix, 0, out=total_neighbors_union_matrix)

    # For J(i,i), if common_neighbors_matrix[i,i]=0, setting total_neighbors_union_matrix[i,i]=1
    # makes J(i,i)=0/1=0. The facade function `calculate_jaccard_similarity` will set J(i,i)=1.
    # This diagonal setting is a convention from the original code to avoid 0/0 if common_neighbors_matrix[i,i] was 0.
    np.fill_diagonal(total_neighbors_union_matrix, 1) # Avoid 0/0 for diagonal in next step if C[i,i]=0
    
    logger.debug("Total unique neighbors (union) calculation complete.")
    return total_neighbors_union_matrix


def get_jaccard_coefficient(common_neighbors_matrix: np.ndarray, total_neighbors_union_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the Jaccard coefficient J(i,j) for all pairs of nodes.

    J(i, j) = |N(i) intersect N(j)| / |N(i) U N(j)|.

    Args:
        common_neighbors_matrix (np.ndarray): Matrix of common neighbor counts.
        total_neighbors_union_matrix (np.ndarray): Matrix of the size of the union of neighbor sets.

    Returns:
        np.ndarray: Matrix of Jaccard coefficients. NaNs (from 0/0) or Infs may occur
                    if `total_neighbors_union_matrix` has zeros. These are handled
                    by the calling facade function.
    """
    logger.debug("Calculating Jaccard coefficient matrix.")
    # Suppress warnings for division by zero or invalid values (NaNs)
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard_matrix = common_neighbors_matrix / total_neighbors_union_matrix
    # Diagonal J(i,i) will be 0/1=0 due to how helper functions prepare their diagonals.
    # This is overridden to 1.0 in the facade.
    logger.debug("Jaccard coefficient calculation complete.")
    return jaccard_matrix


def calculate_jaccard_similarity(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates Jaccard similarity and related matrices for a graph.

    The Jaccard index for a node with itself, J(i,i), is set to 1.0.
    For pairs (i,j) where `|N(i) U N(j)|` is 0, J(i,j) is 0.0.

    Args:
        graph (nx.Graph): A NetworkX graph. Should be unweighted and undirected for
                          standard Jaccard index interpretation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - adjacency_matrix (np.ndarray): Dense adjacency matrix (nodes x nodes).
            - common_neighbors_matrix (np.ndarray): Matrix of common neighbors (C[i,i]=0).
            - total_neighbors_matrix (np.ndarray): Matrix of total unique neighbor counts (U[i,i]=1).
                                                    (Named `total_neighbors_matrix` in output tuple as per prompt)
            - jaccard_similarity_matrix (np.ndarray): Matrix of Jaccard similarity (J[i,i]=1.0).
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX graph.")

    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        logger.warning("Graph is empty. Returning empty arrays of shape (0,0).")
        empty_float = np.empty((0,0), dtype=float)
        empty_int = np.empty((0,0), dtype=int)
        return empty_int, empty_float, empty_float, empty_float

    logger.info(f"Calculating Jaccard similarity for graph with {n_nodes} nodes and {graph.number_of_edges()} edges.")
    
    # Consistent node ordering for matrix
    # Using sorted list of nodes for deterministic output if graph node order is not guaranteed
    node_list = sorted(list(graph.nodes()))

    # threadpool_limits is kept as per original structure, might be useful for very large computations
    # to manage underlying BLAS/LAPACK threading. limits=None usually means use default.
    with threadpool_limits(limits=None): 
        logger.debug("Converting graph to dense NumPy adjacency matrix.")
        # Using int dtype for adjacency matrix as it's binary.
        adjacency_matrix = nx.to_numpy_array(graph, nodelist=node_list, dtype=int)

        logger.info("Calculating common neighbors matrix.")
        common_neighbors_matrix = get_common_neighbors(adjacency_matrix)

        logger.info("Calculating total neighbors matrix.")
        # Renaming variable to match return tuple name in docstring of facade
        total_neighbors_matrix = get_total_neighbors(adjacency_matrix, common_neighbors_matrix) 
    
        logger.info("Calculating Jaccard similarity matrix.")
        jaccard_similarity_matrix = get_jaccard_coefficient(common_neighbors_matrix, total_neighbors_matrix)
    
    # Handle NaNs (0/0) -> 0.0. Infs (x/0) -> 0.0 (should not happen with current logic if U is non-zero when C is non-zero).
    jaccard_similarity_matrix = np.nan_to_num(jaccard_similarity_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Jaccard index of a node with itself is 1.0.
    # This overrides the 0.0 that would result from C[i,i]=0 and U[i,i]=1.
    if n_nodes > 0:
        np.fill_diagonal(jaccard_similarity_matrix, 1.0)

    logger.info("Jaccard similarity calculation complete.")
    return adjacency_matrix, common_neighbors_matrix, total_neighbors_matrix, jaccard_similarity_matrix