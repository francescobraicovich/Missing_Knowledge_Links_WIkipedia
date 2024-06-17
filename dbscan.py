import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize

def dbscan_from_similarity(similarity_matrix, min_samples=2):
    
    # find the maximum similarity score and normalise the similarity matrix
    max_similarity = np.max(similarity_matrix)
    similarity_matrix = similarity_matrix / max_similarity
    
    # Convert the similarity matrix to a distance matrix
    distance_matrix = 1 - similarity_matrix

    # find min and max cluster sizes
    n = np.shape(similarity_matrix)[0]
    min_cluster_size = int(np.ceil(n/100))
    max_cluster_size = int(np.ceil(n/10))

    # define a set of values for cluster_selection_epsilon
    cluster_selection_epsilon_values = np.linspace(0.1, 0.5, 20)

    # save the best silhouette score
    best_silhouette = 0
    best_labels = None
    times_silhouette_not_improved = 0
    
    for eps in cluster_selection_epsilon_values:
        
        # run HDBSCAN
        dbscan = HDBSCAN(min_samples=min_samples, metric='precomputed', cluster_selection_epsilon=eps, 
                         min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, n_jobs=-1)
        labels = dbscan.fit_predict(distance_matrix)
        
        # check if the number of clusters is within the range
        num_clusters = len(np.unique(labels))

        # calculate the silhouette score
        silhouette = silhouette_score(distance_matrix, labels)

        print(f'Cluster Selection Epsilon: {eps}, Number of Clusters: {num_clusters}, Silhouette Score: {silhouette}')

        # update the best silhouette score
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            times_silhouette_not_improved = 0
            best_labels = labels
        else:
            times_silhouette_not_improved += 1

        if times_silhouette_not_improved > 2:
            break
  
    # print the silhouette score
    silhouette = silhouette_score(distance_matrix, best_labels)

    print(f'Silhouette Score: {silhouette}')
    
    return best_labels