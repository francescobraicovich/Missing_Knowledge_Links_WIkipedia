import networkx as nx
import numpy as np

def find_missing_links(graph, similarity_matrix, cluster_labels, weak_threshold, 
                       strong_threshold, specific_cluster=None, specific_node=None):

    if specific_node is not None:
        # find the cluster of the specific node
        specific_cluster = cluster_labels[list(graph.nodes).index(specific_node)]

    adj_matrix = nx.to_numpy_array(graph)
    node_names = list(graph.nodes)

    # initialise a missing link dictionary
    missing_links = {}

    # find the number of clusters
    num_clusters = len(np.unique(cluster_labels))
    min_range = 0

    if specific_cluster is not None:
        min_range = specific_cluster
        num_clusters = specific_cluster + 1

    # iterate over the clusters
    for c in range(num_clusters):

        # get the indices of the nodes in the cluster
        cluster_indices = np.where(cluster_labels == c)[0]

        for i in range(len(cluster_indices)):
            index_i = cluster_indices[i]
            node_i = node_names[index_i]

            if specific_node is not None and node_i != specific_node:
                continue

            else:
                for j in range(i+1, len(cluster_indices)):

                    
                    index_j = cluster_indices[j]                
                    node_j = node_names[index_j]

                    # get the similarity score
                    similarity_score = similarity_matrix[index_i, index_j]

                    # check if the similarity score is below the threshold
                    if similarity_score >= weak_threshold and adj_matrix[index_i, index_j] == 0:
                        missing_links[(node_i, node_j)] = similarity_score

        # take the upper triangular part of the similarity matrix to avoid duplicates
        similarity_matrix = np.triu(similarity_matrix, k=1)

        # mask nodes with high similarity scores using the strong threshold
        strong_indices = np.where(similarity_matrix < strong_threshold)

        # save the links that are not in the graph
        for i, j in zip(strong_indices[0], strong_indices[1]):
            if adj_matrix[i, j] == 1:
                continue
            
            node_i = node_names[i]
            node_j = node_names[j]

            if (node_i, node_j) in missing_links or (node_j, node_i) in missing_links:
                pass
            
            else:
                print(f'Adding missing link between {node_i} and {node_j}, similarity score: {similarity_matrix[i, j]}')
                missing_links[(node_i, node_j)] = similarity_matrix[i, j]

    # order the missing links by similarity score
    missing_links = dict(sorted(missing_links.items(), key=lambda x: x[1], reverse=True))

    return missing_links

def print_missing_links(missing_links, cluster_labels, start_node, n=10):
    

    print(f'Top {n} missing links:')
    for i, (link, score) in enumerate(missing_links.items()):
        print(f'{i}: {link[0]} <-- {score} --> {link[1]}')
        if i == n:
            break

    print('')