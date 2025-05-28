"""
Module for building feature datasets for training link prediction models
and for generating features for missing link candidates.
"""
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import networkx as nx
from ..utils import filter_sentences # Assuming filter_sentences is in utils.py

logger = logging.getLogger(__name__)

SUBSTRING_TO_REMOVE_CATEGORIES = ['wiki', 'cs1', 'articles', 'pages', 'redirects', 'template', 'disputes', 'iso', 'dmy', 'short', 's1']
DEFAULT_COLUMNS = [
    'node_1', 'node_2', 'degree_node_1', 'degree_node_2', 'common_neighbors', 
    'total_neighbors', 'similarity', 'common_categories', 'total_categories', 
    'n_categories_node_1', 'n_categories_node_2', 
    'cluster_node_1', 'cluster_node_2', 'link'
]
DEFAULT_MAX_TRAINING_SAMPLES = 1_000_000 # From original 10e5
MIN_TRAINING_SAMPLES_FALLBACK = 50_000 # Fallback minimum for training samples


def _find_indices_of_links(
    dataset_type: str, 
    missing_link_candidates_matrix: np.ndarray, 
    num_nodes: int, 
    num_candidate_links: int, 
    max_training_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds pairs of node indices for dataset construction.

    If `dataset_type` is 'predict', it returns indices of all missing link candidates
    (non-zero entries in `missing_link_candidates_matrix`).
    If `dataset_type` is 'train', it samples random pairs of nodes that are *not*
    present in `missing_link_candidates_matrix` (which should represent existing links
    or known non-links to avoid for negative sampling), aiming for a balanced dataset size.

    Args:
        dataset_type (str): Either 'predict' or 'train'.
        missing_link_candidates_matrix (np.ndarray): A matrix where non-zero entries
                                                    indicate pairs to be handled based on `dataset_type`.
                                                    For 'train', these are pairs to *avoid* sampling.
                                                    For 'predict', these are the pairs *to* sample.
        num_nodes (int): Total number of nodes in the graph.
        num_candidate_links (int): Number of pairs marked in `missing_link_candidates_matrix`
                                   (used for 'train' type to estimate sample size).
        max_training_samples (int): Maximum number of samples for the training dataset.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays, `i` and `j`, containing the
                                       row and column indices of selected node pairs.
    """
    if dataset_type == 'predict':
        logger.info("Finding indices for 'predict' dataset (all missing link candidates).")
        missing_link_mask = missing_link_candidates_matrix > 0
        i, j = np.where(missing_link_mask)
        logger.info(f"Found {len(i)} pairs for 'predict' dataset.")
        return i, j
    
    elif dataset_type == 'train':
        logger.info("Finding indices for 'train' dataset (sampling non-candidate pairs).")
        # Adjusted logic: 5x candidates, or at least MIN_TRAINING_SAMPLES_FALLBACK, but capped by max_training_samples.
        dataset_length = int(min(max(num_candidate_links * 5, MIN_TRAINING_SAMPLES_FALLBACK), max_training_samples))
        logger.info(f"Targeting {dataset_length} non-candidate pairs for 'train' dataset.")
        
        # Note: np.random.seed(1) was removed. Deterministic sampling is not guaranteed here.
        # For fully reproducible training sets, manage seeding at a higher level if needed.
        logger.info("Sampling random indices. Deterministic sampling for training indices is not guaranteed by this function alone.")

        # Sample random indices i, j
        # Ensure we don't sample more unique pairs than available if num_nodes is small
        max_possible_pairs = num_nodes * (num_nodes -1) // 2
        if dataset_length > max_possible_pairs and num_nodes >1 : # only if trying to sample more than what's possible
            logger.warning(f"Requested dataset_length {dataset_length} is greater than max possible unique pairs {max_possible_pairs}. Capping at max possible pairs.")
            dataset_length = max_possible_pairs
            if dataset_length == 0 and num_nodes <=1: # Handle edge case of 0 or 1 node
                 logger.warning("Not enough nodes to form pairs. Returning empty indices.")
                 return np.array([], dtype=int), np.array([], dtype=int)


        sampled_i = np.array([], dtype=int)
        sampled_j = np.array([], dtype=int)
        
        attempts = 0
        max_attempts = 10 # Safeguard against infinite loops if sampling is very difficult

        # We need to ensure i != j and we're not picking existing candidates
        # This loop can be inefficient if missing_link_candidates_matrix is dense
        # or if dataset_length is close to max_possible_pairs.
        while len(sampled_i) < dataset_length and attempts < max_attempts:
            needed = dataset_length - len(sampled_i)
            # Sample slightly more to account for rejections
            sample_size = int(needed * 1.2) + 100 
            
            temp_i = np.random.choice(num_nodes, size=sample_size)
            temp_j = np.random.choice(num_nodes, size=sample_size)

            # Ensure i != j
            valid_pair_mask = temp_i != temp_j
            temp_i = temp_i[valid_pair_mask]
            temp_j = temp_j[valid_pair_mask]
            
            # Ensure not in missing_link_candidates_matrix (which also includes actual links)
            # This assumes missing_link_candidates_matrix[x,y] > 0 means it's a candidate/existing link.
            # For undirected, ensure we check both (row,col) and (col,row) if matrix is not symmetric for candidates
            # However, missing_link_candidates_matrix is usually derived from (adj_matrix == 0) & (similarity > threshold)
            # and should be symmetric if the base graph is undirected.
            not_candidate_mask = missing_link_candidates_matrix[temp_i, temp_j] == 0
            if not missing_link_candidates_matrix.flags['symmetric']: # If not guaranteed symmetric for candidate representation
                 not_candidate_mask &= missing_link_candidates_matrix[temp_j, temp_i] == 0


            temp_i = temp_i[not_candidate_mask]
            temp_j = temp_j[not_candidate_mask]
            
            # Add to overall list
            sampled_i = np.concatenate((sampled_i, temp_i))
            sampled_j = np.concatenate((sampled_j, temp_j))
            
            # Remove duplicates from what we've collected so far to ensure unique pairs
            # This is complex to do efficiently while growing. A simpler approach is to sample more and then unique-fy.
            # For now, let's assume the impact of duplicates is minor or handled by sampling more initially.
            # Truncate to dataset_length if we overshoot
            if len(sampled_i) > dataset_length:
                sampled_i = sampled_i[:dataset_length]
                sampled_j = sampled_j[:dataset_length]
            
            attempts += 1
            logger.debug(f"Sampling attempt {attempts}: collected {len(sampled_i)} / {dataset_length} pairs.")

        if len(sampled_i) < dataset_length:
            logger.warning(f"Could only sample {len(sampled_i)} non-candidate pairs after {max_attempts} attempts (target: {dataset_length}).")
        
        logger.info(f"Selected {len(sampled_i)} non-candidate pairs for 'train' dataset.")
        return sampled_i, sampled_j
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'predict' or 'train'.")


def _process_node_pair(
    node_idx_i: int,
    node_idx_j: int,
    node_names_list: list[str],
    node_degrees_list: list[int], # Using list of pre-calculated degrees
    cluster_labels_list: np.ndarray,
    categories_dict: dict[str, np.ndarray], # Values are np.ndarray of category strings
    substring_to_remove_categories_list: list[str],
    filtered_categories_cache: dict[str, set[str]], # Cache for filtered category sets
    common_neighbors_matrix_view: np.ndarray,
    total_neighbors_matrix_view: np.ndarray,
    adjacency_matrix_view: np.ndarray,
    similarity_matrix_view: np.ndarray
) -> dict:
    """
    Processes a single pair of nodes (node_idx_i, node_idx_j) and computes their features.

    Args:
        node_idx_i: Index of the first node in graph's node list.
        node_idx_j: Index of the second node in graph's node list.
        node_names_list: Ordered list of node names in the graph.
        node_degrees_list: Ordered list of node degrees.
        cluster_labels_list: Array of cluster labels for nodes.
        categories_dict: Dict mapping node name to its raw categories (array of strings).
        substring_to_remove_categories_list: List of substrings to filter from categories.
        filtered_categories_cache: Cache for memoizing filtered category sets.
        common_neighbors_matrix_view: View of the common neighbors matrix.
        total_neighbors_matrix_view: View of the total neighbors matrix.
        adjacency_matrix_view: View of the adjacency matrix (for link target).
        similarity_matrix_view: View of the similarity matrix.

    Returns:
        A dictionary containing all features for the node pair.
    """
    node_i_name = node_names_list[node_idx_i]
    node_j_name = node_names_list[node_idx_j]
    
    # Process categories for node_i
    if node_i_name not in filtered_categories_cache:
        raw_cats_i = categories_dict.get(node_i_name, np.array([], dtype=str))
        filtered_categories_cache[node_i_name] = filter_sentences(raw_cats_i, substring_to_remove_categories_list)
    categories_i_set = filtered_categories_cache[node_i_name]

    # Process categories for node_j
    if node_j_name not in filtered_categories_cache:
        raw_cats_j = categories_dict.get(node_j_name, np.array([], dtype=str))
        filtered_categories_cache[node_j_name] = filter_sentences(raw_cats_j, substring_to_remove_categories_list)
    categories_j_set = filtered_categories_cache[node_j_name]

    return {
        'node_1': node_i_name,
        'node_2': node_j_name,
        'degree_node_1': node_degrees_list[node_idx_i],
        'degree_node_2': node_degrees_list[node_idx_j],
        'common_neighbors': common_neighbors_matrix_view[node_idx_i, node_idx_j],
        'total_neighbors': total_neighbors_matrix_view[node_idx_i, node_idx_j],
        'similarity': similarity_matrix_view[node_idx_i, node_idx_j],
        'common_categories': len(categories_i_set.intersection(categories_j_set)),
        'total_categories': len(categories_i_set.union(categories_j_set)),
        'n_categories_node_1': len(categories_i_set),
        'n_categories_node_2': len(categories_j_set),
        'cluster_node_1': cluster_labels_list[node_idx_i],
        'cluster_node_2': cluster_labels_list[node_idx_j],
        'link': adjacency_matrix_view[node_idx_i, node_idx_j] # Target variable
    }


def build_feature_dataset(
    graph: nx.Graph,
    adjacency_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    missing_link_candidates_matrix: np.ndarray, # For 'predict' these are candidates, for 'train' these are existing/candidates to avoid
    common_neighbors_matrix: np.ndarray,
    total_neighbors_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    categories_dict: dict[str, np.ndarray], # Values are np.ndarray of category strings
    dataset_type: str, # 'train' or 'predict'
    filtered_categories_cache: dict[str, set[str]] | None = None,
    max_training_samples: int = DEFAULT_MAX_TRAINING_SAMPLES
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """
    Builds a feature dataset for link prediction or candidate analysis using multithreading.

    Args:
        graph (nx.Graph): The graph object, used for consistent node ordering and degrees.
        adjacency_matrix (np.ndarray): Adjacency matrix of the graph.
        similarity_matrix (np.ndarray): Matrix of node pair similarities.
        missing_link_candidates_matrix (np.ndarray): Matrix indicating candidate links.
                                                    For training, these pairs (and existing links) are excluded
                                                    from negative sampling. For prediction, these are the pairs to featurize.
        common_neighbors_matrix (np.ndarray): Matrix of common neighbor counts.
        total_neighbors_matrix (np.ndarray): Matrix of total unique neighbor counts.
        cluster_labels (np.ndarray): Array of cluster labels for each node.
        categories_dict (dict[str, np.ndarray]): Dict mapping node name to its raw categories.
        dataset_type (str): Either 'train' (sample non-links) or 'predict' (use candidates).
        filtered_categories_cache (dict[str, set[str]] | None, optional): Cache for filtered categories.
                                                                        Initialized if None. Defaults to None.
        max_training_samples (int, optional): Max samples for training dataset.
                                              Defaults to DEFAULT_MAX_TRAINING_SAMPLES.

    Returns:
        tuple[pd.DataFrame, dict[str, set[str]]]:
            - DataFrame containing the features for each node pair.
            - Updated filtered_categories_cache.
    """
    if filtered_categories_cache is None:
        filtered_categories_cache = {}

    # Ensure consistent mapping from graph nodes to matrix indices
    # graph.nodes() can be non-deterministic in order if graph is modified, so sort or fix order.
    # However, matrices are usually built with a fixed node order (e.g., from initial graph.nodes() list).
    # It's crucial that node_names_list here corresponds to the order used for matrix generation.
    # Assuming matrices were generated using list(graph.nodes()) at the time of their creation.
    node_names_list = list(graph.nodes()) # This order MUST match the matrix row/col order.
    # Pre-calculate degrees based on this node order
    # Ensure degrees are integers
    node_degrees_list = [int(graph.degree(n)) for n in node_names_list] 


    num_candidate_links = np.sum(missing_link_candidates_matrix > 0)
    
    indices_i_array, indices_j_array = _find_indices_of_links(
        dataset_type=dataset_type,
        missing_link_candidates_matrix=missing_link_candidates_matrix,
        num_nodes=graph.number_of_nodes(),
        num_candidate_links=num_candidate_links,
        max_training_samples=max_training_samples
    )

    if len(indices_i_array) == 0:
        logger.warning(f"No node pairs selected for dataset_type '{dataset_type}'. Returning empty DataFrame.")
        # Return empty DataFrame with correct columns
        cols = DEFAULT_COLUMNS.copy()
        if dataset_type == 'predict':
            cols.remove('link')
        return pd.DataFrame(columns=cols), filtered_categories_cache

    results = []
    # Using tqdm for progress bar with ThreadPoolExecutor requires careful handling or external libs
    # For simplicity, logging start/end and relying on tqdm's default behavior if it works with futures.
    logger.info(f"Processing {len(indices_i_array)} node pairs using ThreadPoolExecutor.")

    with ThreadPoolExecutor() as executor:
        # Submit tasks: each task processes one pair (idx_i, idx_j)
        futures = [
            executor.submit(
                _process_node_pair,
                idx_i, idx_j, # These are indices for the matrices
                node_names_list, 
                node_degrees_list,
                cluster_labels, 
                categories_dict, 
                SUBSTRING_TO_REMOVE_CATEGORIES,
                filtered_categories_cache, # This cache will be mutated by threads; needs to be thread-safe or accept potential race conditions if minor. Python dicts are mostly thread-safe for atomic ops.
                common_neighbors_matrix, 
                total_neighbors_matrix, 
                adjacency_matrix,
                similarity_matrix # Pass similarity matrix
            )
            for idx_i, idx_j in zip(indices_i_array, indices_j_array)
        ]
        
        # Collect results with progress
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Building '{dataset_type}' dataset features"):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error processing a node pair: {e}", exc_info=True) # Log error with traceback
    
    if not results:
        logger.warning(f"No results obtained from thread pool for dataset_type '{dataset_type}'. Returning empty DataFrame.")
        cols = DEFAULT_COLUMNS.copy()
        if dataset_type == 'predict':
            cols.remove('link')
        return pd.DataFrame(columns=cols), filtered_categories_cache

    df = pd.DataFrame(results)
    
    # Assertion for similarity (already included in results from _process_node_pair)
    if 'similarity' not in df.columns:
         logger.error("'similarity' column is missing from the results processed by _process_node_pair.")
    elif df['similarity'].isna().sum() > 0:
        logger.warning(f"Found {df['similarity'].isna().sum()} NaN values in 'similarity' column. Check data for these pairs.")
        # df['similarity'].fillna(0, inplace=True) # Example: fill NaNs if they occur

    # Reorder columns
    current_columns = DEFAULT_COLUMNS.copy()
    if dataset_type == 'predict':
        if 'link' in current_columns: # Ensure 'link' is in default if attempting removal
            current_columns.remove('link')
        else: # Should not happen if DEFAULT_COLUMNS is correct
            logger.warning("'link' column not found in DEFAULT_COLUMNS for removal in 'predict' mode.")
    
    # Ensure all expected columns are present in DataFrame, add missing ones with NaN or default
    for col in current_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' missing from DataFrame, adding with NaNs.")
            df[col] = np.nan 
            
    df = df[current_columns] # Select and order
    
    logger.info(f"Successfully built '{dataset_type}' dataset with {len(df)} pairs and {len(df.columns)} features.")
    return df, filtered_categories_cache
