import pandas as pd
import numpy as np
from tqdm import tqdm

# Define the blacklist substrings
substring_to_remove_categories = ['wiki', 'cs1', 'articles', 'pages', 'redirects', 'template', 'disputes', 'iso', 'dmy', 'short', 's1']

columns = ['node_1', 'node_2', 'degree_node_1', 'degree_node_2', 'common_neighbors', 
           'total_neighbors', 'similarity', 'common_categories', 'total_categories', 
           'n_categories_node_1', 'n_categories_node_2', 
           'cluster_node_1', 'cluster_node_2', 'link']

def filter_sentences(sentences, words_to_filter):
    """
    Filters out sentences that contain any of the specified words.

    Args:
        sentences (numpy.ndarray): An array of sentences to filter.
        words_to_filter (list): A list of words to filter out.

    Returns:
        set: A set of filtered sentences.

    """
    # convert the sentences to lowercase
    sentences_lower = np.char.lower(sentences)

    # create a mask 
    mask = np.zeros(len(sentences_lower), dtype=bool)

    # iterate over the words to filter
    for word in words_to_filter:
        current_mask = np.char.find(sentences_lower, word) != -1
        mask |= current_mask

    # filter the sentences
    filtered_sentences = sentences[~mask]

    return set(filtered_sentences)

def find_indices_of_links(df_type, missing_link_matrix):
    """
    Find indices of links based on the given DataFrame type and missing link matrix.

    Parameters:
    - df_type (str): The type of DataFrame. Can be 'missing links' or 'train dataset'.
    - missing_link_matrix (numpy.ndarray): The matrix representing missing links.

    Returns:
    - i (numpy.ndarray): The row indices of the links.
    - j (numpy.ndarray): The column indices of the links.
    """

    if df_type == 'missing links':
        # take the indices of the missing links
        missing_link_mask = missing_link_matrix > 0
        i, j = np.where(missing_link_mask)
        return i, j
    
    # if instead we are building the train dataset
    # find the number of missing link candidates
    num_link_candidates = np.sum(missing_link_matrix > 0)

    # calculate dataset length as the number of missing link candidates * 5
    dataset_length = int(max(num_link_candidates * 5, 10e5))

    # set the seed
    np.random.seed(1)

    # sample dataset_length random indices i, j
    n = np.shape(missing_link_matrix)[0]
    i = np.random.choice(n, dataset_length)
    j = np.random.choice(n, dataset_length)

    # iterate until the random indices do not correspond to missing links
    while True:
        mask = missing_link_matrix[i, j] > 0
        
        # remove the indices that correspond to missing links
        i = i[~mask]
        j = j[~mask]

        if len(i) < num_link_candidates:
            i = np.append(i, np.random.choice(n, dataset_length - len(i)))
            j = np.append(j, np.random.choice(n, dataset_length - len(j)))
        else:
            break

    return i, j

def build_dataset(adjacency_matrix, similarity_matrix, missing_link_matrix, 
                        common_neighbors_matrix, total_neighbors_matrix, 
                        graph, cluster_labels, categories_dict, df_type, filtered_categories_dict=None):
    """
    Build a dataset based on the given matrices and graph.

    Parameters:
    - adjacency_matrix (numpy.ndarray): The adjacency matrix representing the links between nodes.
    - similarity_matrix (numpy.ndarray): The similarity matrix representing the similarity scores between nodes.
    - missing_link_matrix (numpy.ndarray): The matrix representing the missing links between nodes.
    - common_neighbors_matrix (numpy.ndarray): The matrix representing the number of common neighbors between nodes.
    - total_neighbors_matrix (numpy.ndarray): The matrix representing the total number of neighbors for each node pair.
    - graph (networkx.Graph): The graph representing the nodes and their connections.
    - cluster_labels (numpy.ndarray): The cluster labels for each node.
    - categories_dict (dict): A dictionary mapping each node to its categories.
    - df_type (str): The type of dataset being built.
    - filtered_categories_dict (dict, optional): A dictionary storing the filtered categories for each node to avoid recomputation.

    Returns:
    - df (pandas.DataFrame): The built dataset.
    - filtered_categories_dict (dict): The updated filtered categories dictionary.
    """
   
    # extract node names
    node_names = list(graph.nodes)
    
    # initialise the dataframe
    df = pd.DataFrame(columns=columns)
    
    # find the indices of the links
    i, j = find_indices_of_links(df_type, missing_link_matrix)
    
    # insert the similarity scores
    df['similarity'] = similarity_matrix[i, j]

    # get the node names and the degrees of the links
    indices = zip(i, j)
    node_names_of_indices = [(node_names[node_i], node_names[node_j]) for node_i, node_j in indices]
    degrees = list(dict(graph.degree()).values())

    # initialise the filtered categories dictionary to store the filtered categories of each node, avoiding recomputation
    if filtered_categories_dict is None:
        filtered_categories_dict = {}

    # Add a progress bar to the for loop
    for idx, nodes in tqdm(enumerate(node_names_of_indices), total=len(node_names_of_indices), desc=f"Building {df_type} dataset"):
        node_i, node_j = nodes

        # Check if the nodes are in the filtered categories dictionary
        if node_i not in filtered_categories_dict:
            categories_i = categories_dict[node_i]
            filtered_categories_dict[node_i] = filter_sentences(categories_i, substring_to_remove_categories)
        categories_i = filtered_categories_dict[node_i]

        if node_j not in filtered_categories_dict:
            categories_j = categories_dict[node_j]
            filtered_categories_dict[node_j] = filter_sentences(categories_j, substring_to_remove_categories)
        categories_j = filtered_categories_dict[node_j]

        # Insert node names
        df.loc[idx, 'node_1'] = node_i
        df.loc[idx, 'node_2'] = node_j

        # Insert categories features in the dataframe
        df.loc[idx, 'common_categories'] = len(categories_i.intersection(categories_j)) # Number of common categories
        df.loc[idx, 'total_categories'] = len(categories_i.union(categories_j)) # Total number of categories
        df.loc[idx, 'n_categories_node_1'] = len(categories_i) # Number of categories of node 1
        df.loc[idx, 'n_categories_node_2'] = len(categories_j) # Number of categories of node 2

        # Insert cluster features
        df.loc[idx, 'cluster_node_1'] = cluster_labels[i[idx]] # Cluster of node 1
        df.loc[idx, 'cluster_node_2'] = cluster_labels[j[idx]] # Cluster of node 2

        # Insert common neighbors features
        df.loc[idx, 'common_neighbors'] = common_neighbors_matrix[i[idx], j[idx]] # Number of common neighbors
        df.loc[idx, 'total_neighbors'] = total_neighbors_matrix[i[idx], j[idx]] # Total number of neighbors

        # Insert degree features
        df.loc[idx, 'degree_node_1'] = degrees[i[idx]] # Degree of node 1
        df.loc[idx, 'degree_node_2'] = degrees[j[idx]] # Degree of node 2

        # Insert link feature (target variable)
        df.loc[idx, 'link'] = adjacency_matrix[i[idx], j[idx]] # Link between node 1 and node 2

    # reorder the columns of the dataframe
    df = df[columns]

    return df, filtered_categories_dict

def process_node_pair(idx, nodes, filtered_categories_dict, categories_dict, substring_to_remove_categories, cluster_labels, i, j, common_neighbors_matrix, total_neighbors_matrix, degrees, adjacency_matrix):
    """
    Process a pair of nodes and calculate various metrics based on their categories, cluster labels, and adjacency matrix.

    Args:
        idx (int): The index of the node pair.
        nodes (tuple): A tuple containing the two nodes.
        filtered_categories_dict (dict): A dictionary containing pre-filtered categories for each node.
        categories_dict (dict): A dictionary containing all categories for each node.
        substring_to_remove_categories (str): A substring to be removed from the categories.
        cluster_labels (list): A list of cluster labels for each node.
        i (list): A list of indices for the first node in each pair.
        j (list): A list of indices for the second node in each pair.
        common_neighbors_matrix (numpy.ndarray): A matrix containing the number of common neighbors between nodes.
        total_neighbors_matrix (numpy.ndarray): A matrix containing the total number of neighbors for each node.
        degrees (numpy.ndarray): An array containing the degrees of each node.
        adjacency_matrix (numpy.ndarray): An adjacency matrix representing the links between nodes.

    Returns:
        dict: A dictionary containing various metrics calculated for the node pair, including the index, node names,
              common categories, total categories, number of categories for each node, cluster labels for each node,
              number of common neighbors, total number of neighbors, degrees of each node, and the link between the nodes.
    """
    node_i, node_j = nodes
    
    if node_i not in filtered_categories_dict:
        categories_i = categories_dict[node_i]
        filtered_categories_dict[node_i] = filter_sentences(categories_i, substring_to_remove_categories)
    categories_i = filtered_categories_dict[node_i]

    if node_j not in filtered_categories_dict:
        categories_j = categories_dict[node_j]
        filtered_categories_dict[node_j] = filter_sentences(categories_j, substring_to_remove_categories)
    categories_j = filtered_categories_dict[node_j]

    return {
        'idx': idx,
        'node_1': node_i,
        'node_2': node_j,
        'common_categories': len(categories_i.intersection(categories_j)),
        'total_categories': len(categories_i.union(categories_j)),
        'n_categories_node_1': len(categories_i),
        'n_categories_node_2': len(categories_j),
        'cluster_node_1': cluster_labels[i[idx]],
        'cluster_node_2': cluster_labels[j[idx]],
        'common_neighbors': common_neighbors_matrix[i[idx], j[idx]],
        'total_neighbors': total_neighbors_matrix[i[idx], j[idx]],
        'degree_node_1': degrees[i[idx]],
        'degree_node_2': degrees[j[idx]],
        'link': adjacency_matrix[i[idx], j[idx]]
    }

import concurrent.futures

def build_dataset_multi_thread(adjacency_matrix, similarity_matrix, missing_link_matrix, 
                        common_neighbors_matrix, total_neighbors_matrix, 
                        graph, cluster_labels, categories_dict, df_type, filtered_categories_dict=None):

    # extract node names
    node_names = list(graph.nodes)
    
    # initialise the dataframe
    df = pd.DataFrame(columns=columns)
    
    # find the indices of the links
    i, j = find_indices_of_links(df_type, missing_link_matrix)
    
    # get the node names and the degrees of the links
    indices = zip(i, j)
    node_names_of_indices = [(node_names[node_i], node_names[node_j]) for node_i, node_j in indices]
    degrees = list(dict(graph.degree()).values())

    # initialise the filtered categories dictionary to store the filtered categories of each node, avoiding recomputation
    if filtered_categories_dict is None:
        filtered_categories_dict = {}

    # initialise the results list to store the results of the threads (dictionaries with the features of each pair of nodes)
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_node_pair,
                idx, nodes, filtered_categories_dict, categories_dict, substring_to_remove_categories, cluster_labels, i, j, common_neighbors_matrix, total_neighbors_matrix, degrees, adjacency_matrix
            )
            for idx, nodes in enumerate(node_names_of_indices)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    df = pd.DataFrame(results).sort_values(by='idx').drop(columns='idx').reset_index(drop=True)
    
    # insert the similarity scores
    df['similarity'] = similarity_matrix[i, j]

    # assert there are no missing values in the similarity column
    assert df['similarity'].isna().sum() == 0

    # reorder the columns of the dataframe
    columns_to_keep = columns
    if df_type == 'missing links':
        columns_to_keep.remove('link')
    df = df[columns_to_keep]
    
    return df, filtered_categories_dict

    


    
            



    
