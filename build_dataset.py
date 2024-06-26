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

    if df_type == 'missing links':
        # take the indices of the missing links
        missing_link_mask = missing_link_matrix > 0
        i, j = np.where(missing_link_mask)
        return i, j
    
    # if instead we are building the train dataset
    # find the number of missing link candidates
    num_link_candidates = np.sum(missing_link_matrix > 0)

    # calculate dataset lengh as the number of missing link candidates * 5
    dataset_length = int(min(num_link_candidates, 2*10e4))

    # sample dataset_lenght random indices i, j
    n = np.shape(missing_link_matrix)[0]
    i = np.random.choice(n, dataset_length)
    j = np.random.choice(n, dataset_length)

    # iterate until the random indices do not correspond to missing links
    while True:
        mask = missing_link_matrix[i, j] > 0
        
        # remove the indices that correspond to missing links
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

    return df, filtered_categories_dict


def build_missing_link_dataset(adjacency_matrix, similarity_matrix, missing_link_matrix, common_neighbors_matrix, total_neighbors_matrix, graph, cluster_labels, categories_dict):
    
    # extract node names
    node_names = list(graph.nodes)

    # find the number of missing link candidates
    num_link_candidates = np.sum(missing_link_matrix > 0)

    # initialise the dataframe
    df = pd.DataFrame(columns=['node 1', 'node 2', 'Common Neighbors', 'Total Neighbors','Similarity', 'Common Categories', 'Total Categories', 
                               'n_categories node 1', 'n_categories node 2', 
                               'cluster node 1', 'cluster node 2'])
    
    missing_link_mask = missing_link_matrix > 0

    # take the indices of the missing links
    i, j = np.where(missing_link_mask)

    # insert the similarity scores
    df['Similarity'] = similarity_matrix[i, j]

    # get the node names of the missing links
    indices = zip(i, j)
    node_names_of_indices = [(node_names[node_i], node_names[node_j]) for node_i, node_j in indices]
    
    # initialise the features
    common_categories = np.zeros(len(i))
    total_categories = np.zeros(len(i))
    n_categories_node_1 = np.zeros(len(i))
    n_categories_node_2 = np.zeros(len(i))
    cluster_node_1 = np.zeros(len(i))
    cluster_node_2 = np.zeros(len(i))
    common_neighbors = np.zeros(len(i))
    total_neighbors = np.zeros(len(i))

    node1 = []
    node2 = []

    idx = 0
    for idx, nodes in enumerate(node_names_of_indices):
        node_i, node_j = nodes
        node1.append(node_i)
        node2.append(node_j)
        categories_i = categories_dict[node_i]
        categories_j = categories_dict[node_j]
        common_categories[idx] = len(categories_i.intersection(categories_j))
        total_categories[idx] = len(categories_i.union(categories_j))
        n_categories_node_1[idx] = len(categories_i)
        n_categories_node_2[idx] = len(categories_j)
        cluster_node_1[idx] = cluster_labels[i[idx]]
        cluster_node_2[idx] = cluster_labels[j[idx]]
        common_neighbors[idx] = common_neighbors_matrix[i[idx], j[idx]]
        total_neighbors[idx] = total_neighbors_matrix[i[idx], j[idx]]

    df['Common Categories'] = common_categories
    df['Total Categories'] = total_categories
    df['n_categories node 1'] = n_categories_node_1
    df['n_categories node 2'] = n_categories_node_2
    df['cluster node 1'] = cluster_node_1
    df['cluster node 2'] = cluster_node_2
    df['node 1'] = node1
    df['node 2'] = node2
    df['Common Neighbors'] = common_neighbors
    df['Total Neighbors'] = total_neighbors

    print(node1)

    return df


    


    
            



    
