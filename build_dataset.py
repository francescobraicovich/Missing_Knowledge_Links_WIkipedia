import pandas as pd
import numpy as np

# Define the blacklist substrings
substring_to_remove_links = ['wikipedia', 'category', 'identifier', 'help', 'template', ':', 'wayback', 'isbn', 'jstor']
substring_to_remove_categories = ['wiki', 'cs1', 'articles', 'pages', 'redirects', 'template', 'disputes', 'iso', 'dmy', 'short', 's1']

def build_train_dataset(adjacency_matrix, similarity_matrix, missing_link_matrix, common_neighbors_matrix, total_neighbors_matrix, graph, cluster_labels, categories_dict):

    # extract node names
    node_names = list(graph.nodes)

    # find the number of missing link candidates
    num_link_candidates = np.sum(missing_link_matrix > 0)

    # calculate dataset lengh as the number of missing link candidates * 5
    dataset_length = num_link_candidates
    dataset_length = max(dataset_length, 10e4)
    dataset_length = int(dataset_length)

    # initialise the dataframe
    df = pd.DataFrame(columns=['node 1', 'node 2', 'Degree node 1', 'Degree node 2', 'Common Neighbors', 
                               'Total Neighbors', 'Similarity', 'Common Categories', 'Total Categories', 
                               'n_categories node 1', 'n_categories node 2', 
                               'cluster node 1', 'cluster node 2', 'Link'])


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

    df['Similarity'] = similarity_matrix[i, j]
    indices = zip(i, j)
    node_names_of_indices = [(node_names[node_i], node_names[node_j]) for node_i, node_j in indices]
    degrees = list(dict(graph.degree()).values())
    print(degrees)
    print(len(degrees))
    
    common_categories = np.zeros(len(i))
    total_categories = np.zeros(len(i))
    n_categories_node_1 = np.zeros(len(i))
    n_categories_node_2 = np.zeros(len(i))
    cluster_node_1 = np.zeros(len(i))
    cluster_node_2 = np.zeros(len(i))
    common_neighbors = np.zeros(len(i))
    total_neighbors = np.zeros(len(i))
    degree_node_1 = np.zeros(len(i))
    degree_node_2 = np.zeros(len(i))

    link = np.zeros(len(i))

    pos = 0
    
    filtered_categories_dict = {}

    for pos, nodes in enumerate(node_names_of_indices):
        node_i, node_j = nodes
        
        if node_i in filtered_categories_dict:
            categories_i = set(filtered_categories_dict[node_i])
        else:
            categories_i = categories_dict[node_i]
            mask = np.zeros(len(categories_dict[node_i]), dtype=bool)
            
            for substring in substring_to_remove_categories:
                mask |= np.char.find(list(categories_i), substring) != -1

            categories_i = set(categories_i[~mask])
            filtered_categories_dict[node_i] = categories_i
        
        if node_j in filtered_categories_dict:
            categories_j = set(filtered_categories_dict[node_j])
        else:
            categories_j = categories_dict[node_j]
            mask = np.zeros(len(categories_dict[node_j]), dtype=bool)
            
            for substring in substring_to_remove_categories:
                mask |= np.char.find(list(categories_j), substring) != -1

            categories_j = set(categories_j[~mask])
            filtered_categories_dict[node_j] = categories_j

        

        common_categories[pos] = len(categories_i.intersection(categories_j))
        total_categories[pos] = len(categories_i.union(categories_j))
        n_categories_node_1[pos] = len(categories_i)
        n_categories_node_2[pos] = len(categories_j)
        cluster_node_1[pos] = cluster_labels[i[pos]]
        cluster_node_2[pos] = cluster_labels[j[pos]]
        common_neighbors[pos] = common_neighbors_matrix[i[pos], j[pos]]
        total_neighbors[pos] = total_neighbors_matrix[i[pos], j[pos]]
        degree_node_1[pos] = degrees[i[pos]]
        degree_node_2[pos] = degrees[j[pos]]
        link[pos] = adjacency_matrix[i[pos], j[pos]]

    df['Common Categories'] = common_categories
    df['Total Categories'] = total_categories
    df['n_categories node 1'] = n_categories_node_1
    df['n_categories node 2'] = n_categories_node_2
    df['cluster node 1'] = cluster_node_1
    df['cluster node 2'] = cluster_node_2
    df['Common Neighbors'] = common_neighbors
    df['Total Neighbors'] = total_neighbors
    df['Link'] = link

    return df


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
    dataset_length = np.sum(missing_link_mask)

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

    pos = 0
    for pos, nodes in enumerate(node_names_of_indices):
        node_i, node_j = nodes
        node1.append(node_i)
        node2.append(node_j)
        categories_i = categories_dict[node_i]
        categories_j = categories_dict[node_j]
        common_categories[pos] = len(categories_i.intersection(categories_j))
        total_categories[pos] = len(categories_i.union(categories_j))
        n_categories_node_1[pos] = len(categories_i)
        n_categories_node_2[pos] = len(categories_j)
        cluster_node_1[pos] = cluster_labels[i[pos]]
        cluster_node_2[pos] = cluster_labels[j[pos]]
        common_neighbors[pos] = common_neighbors_matrix[i[pos], j[pos]]
        total_neighbors[pos] = total_neighbors_matrix[i[pos], j[pos]]

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


    


    
            



    
