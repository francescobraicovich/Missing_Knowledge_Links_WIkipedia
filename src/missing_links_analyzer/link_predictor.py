"""
Module for identifying candidate missing links in a graph based on node similarity
and clustering information.
"""
import logging
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

def _process_cluster_for_candidates(
    cluster_id: int, # For logging/tracking purposes if needed, not used in core logic
    node_names_list: list[str],
    nodes_in_cluster_indices: np.ndarray,
    similarity_matrix: np.ndarray,
    weak_threshold: float,
    adjacency_matrix: np.ndarray
) -> tuple[dict[tuple[str, str], float], list[tuple[int, int, float]]]:
    """
    Processes a single cluster to find candidate links based on a weak similarity threshold.

    Iterates through pairs of nodes within the given cluster. If a pair's similarity
    score meets the weak_threshold and they are not already linked, they are considered
    candidate links.

    Args:
        cluster_id (int): The ID of the cluster being processed.
        node_names_list (list[str]): Ordered list of all node names in the graph.
        nodes_in_cluster_indices (np.ndarray): Array of matrix indices for nodes
                                               belonging to this cluster.
        similarity_matrix (np.ndarray): Full similarity matrix for all nodes.
        weak_threshold (float): Minimum similarity score for a pair to be considered
                                a candidate link within a cluster.
        adjacency_matrix (np.ndarray): Full adjacency matrix of the graph.

    Returns:
        tuple[dict[tuple[str, str], float], list[tuple[int, int, float]]]:
            - A dictionary `{(node_i_name, node_j_name): score}` for candidates found.
            - A list of tuples `[(index_i, index_j, score)]` for updating the
              candidate_links_matrix.
    """
    # logger.debug(f"Processing cluster {cluster_id} with {len(nodes_in_cluster_indices)} nodes.")
    cluster_candidate_links_dict: dict[tuple[str, str], float] = {}
    cluster_candidate_matrix_updates: list[tuple[int, int, float]] = []

    for i in range(len(nodes_in_cluster_indices)):
        index_i = nodes_in_cluster_indices[i]
        node_i_name = node_names_list[index_i]
        
        for j in range(i + 1, len(nodes_in_cluster_indices)):
            index_j = nodes_in_cluster_indices[j]
            node_j_name = node_names_list[index_j]

            similarity_score = similarity_matrix[index_i, index_j]

            if similarity_score >= weak_threshold and adjacency_matrix[index_i, index_j] == 0:
                # Ensure consistent ordering for dict keys if graph is undirected
                # Not strictly necessary if consumption doesn't rely on specific order, but good practice.
                # For now, (node_i_name, node_j_name) from loop order is used.
                cluster_candidate_links_dict[(node_i_name, node_j_name)] = similarity_score
                cluster_candidate_matrix_updates.append((index_i, index_j, similarity_score))
    
    # logger.debug(f"Found {len(cluster_candidate_links_dict)} candidates in cluster {cluster_id}.")
    return cluster_candidate_links_dict, cluster_candidate_matrix_updates


def identify_candidate_links_by_similarity(
    graph: nx.Graph,
    similarity_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    weak_threshold: float,
    strong_threshold: float
) -> tuple[dict[tuple[str, str], float], np.ndarray]:
    """
    Identifies candidate missing links in a graph.

    Two types of candidates are identified:
    1. Within-cluster candidates: Pairs of nodes in the same cluster with similarity
       above `weak_threshold` and not already linked. Uses multithreading.
    2. Strong threshold candidates: Any pair of nodes (regardless of cluster) with
       similarity above `strong_threshold` and not already linked.

    Args:
        graph (nx.Graph): The input graph.
        similarity_matrix (np.ndarray): Matrix of similarity scores between nodes.
        cluster_labels (np.ndarray): Array of cluster labels for each node.
        weak_threshold (float): Minimum similarity for within-cluster candidates.
        strong_threshold (float): Minimum similarity for overall strong candidates.

    Returns:
        tuple[dict[tuple[str, str], float], np.ndarray]:
            - A dictionary of candidate links `{(node_name1, node_name2): similarity_score}`,
              sorted by score in descending order.
            - A NumPy array (matrix) where `matrix[i,j]` stores the similarity score
              if pair (i,j) is a candidate, otherwise 0. This matrix is symmetric.
    """
    logger.info(f"Identifying candidate links. Weak threshold: {weak_threshold}, Strong threshold: {strong_threshold}")
    
    # Ensure consistent node ordering for matrix indexing
    node_names_list = list(graph.nodes())
    adj_matrix = nx.to_numpy_array(graph, nodelist=node_names_list, dtype=int)
    num_nodes = graph.number_of_nodes()

    candidate_links_dict: dict[tuple[str, str], float] = {}
    candidate_links_matrix = np.zeros_like(similarity_matrix)

    # --- Within-Cluster Candidates (ThreadPoolExecutor) ---
    # Get unique cluster IDs, excluding noise cluster (-1)
    unique_cluster_ids = np.unique(cluster_labels[cluster_labels != -1])
    logger.info(f"Processing {len(unique_cluster_ids)} non-noise clusters for weak threshold candidates.")

    with ThreadPoolExecutor() as executor:
        futures_to_cluster_id = {}
        for cluster_id_val in unique_cluster_ids:
            nodes_in_cluster_indices_arr = np.where(cluster_labels == cluster_id_val)[0]
            if len(nodes_in_cluster_indices_arr) < 2: # Need at least 2 nodes to form a pair
                logger.debug(f"Skipping cluster {cluster_id_val}, not enough nodes ({len(nodes_in_cluster_indices_arr)}).")
                continue
            
            future = executor.submit(
                _process_cluster_for_candidates,
                cluster_id_val,
                node_names_list,
                nodes_in_cluster_indices_arr,
                similarity_matrix,
                weak_threshold,
                adj_matrix
            )
            futures_to_cluster_id[future] = cluster_id_val
        
        for future in as_completed(futures_to_cluster_id):
            cluster_id_completed = futures_to_cluster_id[future]
            try:
                cluster_dict, cluster_matrix_updates = future.result()
                candidate_links_dict.update(cluster_dict)
                for r, c, score_val in cluster_matrix_updates:
                    candidate_links_matrix[r, c] = score_val
                    candidate_links_matrix[c, r] = score_val # Ensure symmetry
                logger.debug(f"Collected {len(cluster_dict)} candidates from cluster {cluster_id_completed}.")
            except Exception as e:
                logger.error(f"Error processing cluster {cluster_id_completed}: {e}", exc_info=True)

    logger.info(f"Found {len(candidate_links_dict)} candidates from within-cluster processing.")

    # --- Strong Threshold Candidates (across all pairs) ---
    logger.info("Processing all pairs for strong threshold candidates.")
    # Iterate through the upper triangle of the similarity matrix
    for r_idx in range(num_nodes):
        for c_idx in range(r_idx + 1, num_nodes):
            current_similarity_score = similarity_matrix[r_idx, c_idx]
            if current_similarity_score >= strong_threshold and adj_matrix[r_idx, c_idx] == 0:
                node_r_name = node_names_list[r_idx]
                node_c_name = node_names_list[c_idx]
                
                # Check if already added (e.g. by weak threshold or reverse order)
                # Using a consistent key order (e.g. sorted names) could simplify this check
                # For now, check both orders if necessary, though dict keys are (n1, n2) from loop.
                pair_key = (node_r_name, node_c_name)
                # The _process_cluster_for_candidates also uses (node_i_name, node_j_name) order from its loops.
                # If a pair was found by weak threshold, it's already in candidate_links_dict.
                
                if pair_key not in candidate_links_dict:
                    candidate_links_dict[pair_key] = current_similarity_score
                    candidate_links_matrix[r_idx, c_idx] = current_similarity_score
                    candidate_links_matrix[c_idx, r_idx] = current_similarity_score # Ensure symmetry
                    # logger.debug(f"Added strong candidate {pair_key} with score {current_similarity_score:.3f}")
                elif candidate_links_dict[pair_key] < current_similarity_score: # If found by weak, but this score is higher (unlikely if same matrix)
                    candidate_links_dict[pair_key] = current_similarity_score
                    candidate_links_matrix[r_idx, c_idx] = current_similarity_score
                    candidate_links_matrix[c_idx, r_idx] = current_similarity_score


    # Sort the final dictionary by similarity score in descending order
    sorted_candidate_links_dict = dict(
        sorted(candidate_links_dict.items(), key=lambda item: item[1], reverse=True)
    )
    
    logger.info(f"Total candidate links identified: {len(sorted_candidate_links_dict)}")
    return sorted_candidate_links_dict, candidate_links_matrix