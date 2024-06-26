import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score



def dbscan_from_similarity(similarity_matrix, noise_threshold=None, subcluster_noise=True, verbose=True):
    """
    Perform DBSCAN clustering based on a similarity matrix.

    Parameters:
    similarity_matrix (numpy.ndarray): The similarity matrix.
    noise_threshold (int, optional): The threshold for noise cluster size. Defaults to None.
    subcluster_noise (bool, optional): Whether to subcluster the noise cluster. Defaults to True.
    verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
    numpy.ndarray: The cluster labels.
    """

    # find the maximum similarity score and normalise the similarity matrix
    max_similarity = np.max(similarity_matrix)
    similarity_matrix = similarity_matrix / max_similarity
    
    # Convert the similarity matrix to a distance matrix
    distance_matrix = 1 - similarity_matrix

    # find min and max cluster sizes
    n = np.shape(similarity_matrix)[0]

    # define a set of values for cluster_selection_epsilon
    cluster_selection_epsilon_values = np.linspace(0.01, 0.9, 30)

    # save the best silhouette score
    best_silhouette = -1
    best_labels = None
    times_silhouette_not_improved = 0
    
    for eps in cluster_selection_epsilon_values:
        
        # run HDBSCAN
        dbscan = HDBSCAN(metric='precomputed', cluster_selection_epsilon=eps, min_samples=None, 
                         cluster_selection_method='eom', n_jobs=-1, max_cluster_size=300)
        
        labels = dbscan.fit_predict(distance_matrix)
        
        # check if the number of clusters is within the range
        num_clusters = len(np.unique(labels))

        if num_clusters < 2:
            continue

        # calculate the silhouette score
        silhouette = silhouette_score(1-similarity_matrix, labels)

        if verbose:
            print(f'Epsilon: {np.round(eps, 3)}, N Clusters: {np.round(num_clusters, 3)}, Silhouette Score: {np.round(silhouette, 3)}')

        # update the best silhouette score
        if silhouette >= best_silhouette:
            best_silhouette = silhouette
            times_silhouette_not_improved = 0
            best_labels = labels
        else:
            times_silhouette_not_improved += 1

        if times_silhouette_not_improved > 3 and best_silhouette > 0:
            break

    # check the size of the -1 cluster
    noise_size = np.sum(best_labels == -1)

    if noise_threshold is None:
        noise_threshold = int(0.05 * n)

    if noise_size > noise_threshold and subcluster_noise and noise_size > 50:
        print('')
        print(f'Warning: The noise cluster is too large, Noise size: {noise_size}, Threshold: {noise_threshold}')
        print('Subclustering the noise cluster')

        # find the indices of the noise cluster
        noise_indices = np.where(best_labels == -1)[0]

        # run HDBSCAN on the noise cluster
        noise_similarity_matrix = similarity_matrix[noise_indices, :][:, noise_indices]
        noise_labels = dbscan_from_similarity(noise_similarity_matrix, noise_threshold=noise_threshold, subcluster_noise=True, verbose=False)

        # find where the noise cluster has been assigned
        noise_indices = noise_indices[noise_labels != -1]

        # update the labels
        best_labels[noise_indices] = noise_labels[noise_labels != -1] + np.max(best_labels) + 1
  
    # print the silhouette score
    silhouette = silhouette_score(1-similarity_matrix, best_labels)

    if verbose:
        print('-'*50)
        print(f'Final Silhouette Score: {silhouette}')
        print('-'*50)
    
    return best_labels