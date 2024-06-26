import numpy as np
from threadpoolctl import threadpool_limits
from scipy.sparse import csr_matrix, csc_matrix


def get_common_neighbors(adj_matrix):
    """
    Computes the common neighbors between nodes in a given adjacency matrix.

    Parameters:
    adj_matrix (array-like): The adjacency matrix representing the graph.

    Returns:
    array-like: The matrix of common neighbors between nodes.
    """

    # convert it into a csr
    adj_matrix_csr = csr_matrix(adj_matrix)
    adj_matrix_csc = csc_matrix(adj_matrix)

    # compute the common neighbors
    common_neighbors = adj_matrix_csc.dot(adj_matrix_csr)

    # set the diagonal to 0
    common_neighbors.setdiag(0)

    # convert it to a dense matrix
    common_neighbors = common_neighbors.toarray()

    return common_neighbors


def get_total_neighbors(adj_matrix, common_neighbors):
    """
    Calculates the total number of neighbors for each node in a graph.

    Parameters:
    - adj_matrix (numpy.ndarray): The adjacency matrix of the graph.
    - common_neighbors (numpy.ndarray): The matrix representing the number of common neighbors between nodes.

    Returns:
    - total_neighbors_matrix (numpy.ndarray): The matrix representing the total number of neighbors for each node.
    """

    # compute the sum of the neighbors
    col_sum = np.sum(adj_matrix, axis=0)

    # compute the outer sum
    outer_sum_matrix = np.add.outer(col_sum, col_sum)

    total_neighbors_matrix = outer_sum_matrix - common_neighbors - 2*adj_matrix

    # set the diagonal to 1 to avoid division by 0
    np.fill_diagonal(total_neighbors_matrix, 1)

    return total_neighbors_matrix


def get_jaccard_coefficient(common_neighbors, total_neighbors):
    """
    Calculate the Jaccard coefficient.

    The Jaccard coefficient is a measure of similarity between two sets. In the context of graph theory,
    it is often used to measure the similarity between two nodes based on their common neighbors. The Jaccard
    coefficient is defined as the ratio of the number of common neighbors between two nodes to the total number
    of neighbors of the two nodes.

    Parameters:
    - common_neighbors (int): The number of common neighbors between two nodes.
    - total_neighbors (int): The total number of neighbors of the two nodes.

    Returns:
    - jaccard_matrix (float): The Jaccard coefficient between the two nodes.
    """
    # compute the jaccard coefficient
    jaccard_matrix = common_neighbors / total_neighbors

    return jaccard_matrix