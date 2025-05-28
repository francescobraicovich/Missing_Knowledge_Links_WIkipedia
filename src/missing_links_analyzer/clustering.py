"""
Module for performing node clustering using HDBSCAN algorithm.

This module takes a similarity matrix as input, converts it to a distance
matrix, and then applies HDBSCAN to find clusters of nodes. It includes
functionality to iterate over different epsilon values to find an optimal
clustering based on silhouette score and provides an option to recursively
subcluster large noise groups.
"""

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

# Default epsilon range if not provided by settings or caller
DEFAULT_EPSILON_VALUES = np.linspace(0.01, 0.9, 30) 
# Hardcoded constant for sub-clustering check, could be a parameter or from settings in future
MIN_ABSOLUTE_NOISE_SIZE_FOR_SUBCLUSTERING = 50
# Hardcoded max_cluster_size for HDBSCAN, could be a parameter or from settings in future
HDBSCAN_MAX_CLUSTER_SIZE = 300
# Number of iterations silhouette score can not improve before stopping early
SILHOUETTE_IMPROVEMENT_PATIENCE = 3


def cluster_nodes_hdbscan(
    similarity_matrix: np.ndarray,
    cluster_selection_epsilon_values: list[float],
    noise_threshold_ratio: float,
    subcluster_noise_enabled: bool,
    min_cluster_size: int = 5, 
    min_samples: int | None = None
) -> np.ndarray:
    """
    Performs HDBSCAN clustering on nodes based on a similarity matrix.

    The function iterates through a list of `cluster_selection_epsilon` values for HDBSCAN,
    aiming to find the best clustering based on the silhouette score. It normalizes the
    similarity matrix, converts it to a distance matrix, and then applies HDBSCAN.
    If enabled, large noise clusters can be recursively sub-clustered.

    Args:
        similarity_matrix (np.ndarray): A square 2D NumPy array where `similarity_matrix[i, j]`
                                        is the similarity score between node i and node j.
        cluster_selection_epsilon_values (list[float]): A list of epsilon values to try for
                                                        HDBSCAN's `cluster_selection_epsilon`
                                                        parameter. If empty, a default range
                                                        will be used with a warning.
        noise_threshold_ratio (float): A ratio of the total number of nodes. If the size of
                                       the noise cluster (nodes labeled -1) exceeds this ratio
                                       of total nodes, it may be considered for sub-clustering.
        subcluster_noise_enabled (bool): If True, enables the recursive sub-clustering of
                                         large noise clusters.
        min_cluster_size (int, optional): The minimum number of samples in a group for that
                                          group to be considered a cluster. Passed to HDBSCAN.
                                          Defaults to 5.
        min_samples (int | None, optional): The number of samples in a neighborhood for a
                                            point to be considered as a core point. Passed to HDBSCAN.
                                            Defaults to None (HDBSCAN will then use `min_cluster_size`).

    Returns:
        np.ndarray: An array of cluster labels for each node. Noise points are labeled -1.
    """
    if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.ndim != 2 or \
       similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be a square 2D NumPy array.")
    if not (0.0 <= noise_threshold_ratio <= 1.0): # Check ratio bounds
        raise ValueError("noise_threshold_ratio must be between 0.0 and 1.0.")

    n_nodes = similarity_matrix.shape[0]
    if n_nodes == 0:
        logger.warning("Input similarity_matrix is empty. Returning empty array for labels.")
        return np.array([], dtype=int)

    # Normalize the similarity matrix (max value could be 0 if matrix is all zeros)
    max_similarity = np.max(similarity_matrix)
    if max_similarity > 0:
        # Create a copy to avoid modifying the original matrix if it's passed by reference
        similarity_matrix_normalized = similarity_matrix / max_similarity
    else:
        # If all similarities are 0, distance matrix will be all 1s.
        similarity_matrix_normalized = np.copy(similarity_matrix)
    
    # Convert the similarity matrix to a distance matrix
    distance_matrix = 1 - similarity_matrix_normalized

    eps_values_to_iterate = cluster_selection_epsilon_values
    if not eps_values_to_iterate: # Check if the list is empty
        logger.warning(f"cluster_selection_epsilon_values is empty. "
                       f"Using default range: {DEFAULT_EPSILON_VALUES.min():.2f} to {DEFAULT_EPSILON_VALUES.max():.2f} "
                       f"with {len(DEFAULT_EPSILON_VALUES)} steps.")
        eps_values_to_iterate = DEFAULT_EPSILON_VALUES.tolist() # Use tolist() if DEFAULT_EPSILON_VALUES is np.array

    best_silhouette = -1.0  # Silhouette score is between -1 and 1
    # Initialize best_labels as all noise, in case no valid clusters are found
    best_labels = np.full(n_nodes, -1, dtype=int) 
    times_silhouette_not_improved = 0
    
    logger.info(f"Starting HDBSCAN epsilon search. Nodes: {n_nodes}, Min cluster size: {min_cluster_size}, Min samples: {min_samples}")

    for eps_val in eps_values_to_iterate: # Renamed eps to eps_val to avoid confusion
        hdbscan_model = HDBSCAN(
            metric='precomputed', 
            cluster_selection_epsilon=eps_val,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples, # Use the passed min_samples
            cluster_selection_method='eom', # Excess of Mass
            # n_jobs=-1, # Using all available cores, good for performance. Kept from original.
            max_cluster_size=HDBSCAN_MAX_CLUSTER_SIZE # Hardcoded, as per original
        )
        
        current_labels = hdbscan_model.fit_predict(distance_matrix.astype(np.double)) # Ensure double for HDBSCAN
        
        # Number of actual clusters (excluding noise points labeled -1)
        unique_labels_no_noise = np.unique(current_labels[current_labels != -1])
        num_clusters = len(unique_labels_no_noise)

        # Silhouette score requires at least 2 clusters and at most n_samples-1 clusters.
        # Or if all points are in one cluster (possibly with noise), it's not well-defined.
        if num_clusters < 2:
            logger.debug(f'Epsilon: {eps_val:.3f}, N Clusters: {num_clusters}. Silhouette not calculated (requires >= 2 clusters).')
            # If best_labels is still all -1s, and this configuration found at least one cluster, it's an improvement.
            if num_clusters == 1 and np.all(best_labels == -1):
                 logger.info(f"Found 1 cluster with epsilon {eps_val:.3f} (plus noise). Updating best_labels tentatively.")
                 best_labels = current_labels
            continue

        try:
            # Use distance_matrix for silhouette score as metric='precomputed' was used for HDBSCAN
            silhouette = silhouette_score(distance_matrix, current_labels, metric='precomputed')
            logger.info(f'Epsilon: {eps_val:.3f}, N Clusters: {num_clusters}, Silhouette Score: {silhouette:.3f}')
        except ValueError as e: # Handles cases like all samples in one cluster after filtering noise
            logger.warning(f'Epsilon: {eps_val:.3f}, N Clusters: {num_clusters}. Could not calculate Silhouette Score: {e}')
            if num_clusters == 1 and np.all(best_labels == -1) and len(np.unique(current_labels)) > 1 :
                 logger.info(f"Found 1 cluster + noise with epsilon {eps_val:.3f}. Updating best_labels tentatively.")
                 best_labels = current_labels
            continue

        if silhouette >= best_silhouette: # Use >= to prefer solutions with more clusters if silhouette is same
            best_silhouette = silhouette
            times_silhouette_not_improved = 0
            best_labels = current_labels
        else:
            times_silhouette_not_improved += 1

        # Heuristic to stop early if score doesn't improve and some reasonable clustering already found
        if times_silhouette_not_improved > SILHOUETTE_IMPROVEMENT_PATIENCE and best_silhouette > 0: 
            logger.info(f"Silhouette score has not improved for {times_silhouette_not_improved} iterations. Stopping epsilon search early.")
            break
    
    # Sub-clustering noise if conditions are met
    noise_indices = np.where(best_labels == -1)[0]
    noise_size = len(noise_indices)
    # Calculate actual threshold count based on ratio
    dynamic_noise_threshold = int(noise_threshold_ratio * n_nodes)

    if subcluster_noise_enabled and \
       noise_size > dynamic_noise_threshold and \
       noise_size > MIN_ABSOLUTE_NOISE_SIZE_FOR_SUBCLUSTERING: # Check against both dynamic and absolute thresholds
        
        logger.warning(f'The noise cluster is large. Noise size: {noise_size} '
                       f'(Thresholds: >{dynamic_noise_threshold} (ratio-based) and >{MIN_ABSOLUTE_NOISE_SIZE_FOR_SUBCLUSTERING} (absolute)). '
                       f'Attempting to subcluster.')

        # Extract similarity matrix for noise points only
        # Use original similarity_matrix, not the normalized one, for recursive call consistency
        noise_similarity_sub_matrix = similarity_matrix[noise_indices, :][:, noise_indices]
        
        logger.info(f"Recursively calling clustering for {noise_size} noise points.")
        # Recursive call passes most parameters through.
        # cluster_selection_epsilon_values is passed as is.
        noise_labels_recursive = cluster_nodes_hdbscan(
            noise_similarity_sub_matrix, # Pass sub-matrix
            cluster_selection_epsilon_values, 
            noise_threshold_ratio, 
            subcluster_noise_enabled, # Could be set to False for deeper levels
            min_cluster_size,
            min_samples
        )

        # Re-map sub-cluster labels to new unique labels in the main set
        successfully_subclustered_mask = noise_labels_recursive != -1
        actual_noise_indices_to_update = noise_indices[successfully_subclustered_mask]
        
        if np.any(successfully_subclustered_mask):
            max_existing_label = np.max(best_labels) if len(best_labels[best_labels != -1]) > 0 else -1
            
            new_labels_for_noise = noise_labels_recursive[successfully_subclustered_mask] + max_existing_label + 1
            best_labels[actual_noise_indices_to_update] = new_labels_for_noise
            logger.info(f"Integrated {len(actual_noise_indices_to_update)} points from noise into new subclusters.")
        else:
            logger.info("Subclustering of noise did not yield any new clusters.")
    
    # Final silhouette score calculation on potentially modified labels
    final_unique_labels_no_noise = np.unique(best_labels[best_labels != -1])
    final_num_clusters = len(final_unique_labels_no_noise)

    if final_num_clusters >= 2: # Silhouette needs at least 2 clusters
        try:
            # Use distance_matrix corresponding to the full original set of nodes
            final_silhouette = silhouette_score(distance_matrix, best_labels, metric='precomputed')
            logger.info('-'*50)
            logger.info(f'Final Clustering Results: N Clusters: {final_num_clusters}, Silhouette Score: {final_silhouette:.3f}')
            logger.info('-'*50)
        except ValueError as e:
            logger.warning(f"Could not calculate final silhouette score (N Clusters: {final_num_clusters}): {e}")
    else:
        logger.info('-'*50)
        logger.info(f'Final Clustering Results: N Clusters: {final_num_clusters}. Silhouette score not applicable or not calculated.')
        logger.info('-'*50)

    return best_labels