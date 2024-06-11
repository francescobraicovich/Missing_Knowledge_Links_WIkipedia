import numpy as np
from threadpoolctl import threadpool_limits
from scipy.sparse import csr_matrix, csc_matrix


def get_common_neighbors(adj_matrix):

    # convert it into a csr
    adj_matrix_csr = csr_matrix(adj_matrix)
    adj_matrix_csc = csc_matrix(adj_matrix)

    # compute the common neighbors
    common_neighbors = adj_matrix_csc.dot(adj_matrix_csr)

    # set the diagonal to 0
    common_neighbors.setdiag(0)

    # convert it to a dense matrix
    common_neighbors = common_neighbors.toarray()

    return common_neighbors


def get_total_neighbors(adj_matrix, common_neighbors):

    # compute the sum of the neighbors
    col_sum = np.sum(adj_matrix, axis=0)

    # compute the outer sum
    outer_sum_matrix = np.add.outer(col_sum, col_sum)

    total_neighbors_matrix = outer_sum_matrix - common_neighbors - 2*adj_matrix

    # set the diagonal to 1 to avoid division by 0
    np.fill_diagonal(total_neighbors_matrix, 1)

    return total_neighbors_matrix


def get_jaccard_coefficient(common_neighbors, total_neighbors):

    # compute the jaccard coefficient
    jaccard_matrix = common_neighbors / total_neighbors

    return jaccard_matrix
