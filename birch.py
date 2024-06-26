
import numpy as np
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN

def birch_from_similarity(similarity_matrix):

    # find the maximum similarity score and normalise the similarity matrix
    max_similarity = np.max(similarity_matrix)
    similarity_matrix = similarity_matrix / max_similarity
    
    # Convert the similarity matrix to a distance matrix
    distance_matrix = 1 - similarity_matrix

    # find the branching factor
    branching_factor = 50

    # find the threshold
    threshold = 0.5

    # set minimum and maximum number of clusters
    min_samples_in_cluster = 5
    max_samples_in_cluster = 400

    try: 
        # run Birch
        birch_affinity = Birch(branching_factor=branching_factor, threshold=threshold, n_clusters=AffinityPropagation(damping=0.5))

        birch_affinity_labels = birch_affinity.fit_predict(distance_matrix)

        cluster_sizes = np.bincount(birch_affinity_labels)
        max_cluster_size = np.max(cluster_sizes)
        min_cluster_size = np.min(cluster_sizes)

        assert min_cluster_size >= min_samples_in_cluster
        assert max_cluster_size <= max_samples_in_cluster

    except AssertionError:

        # save best score and best labels
        best_score = -1
        best_labels = None

        # min and max number of clusters
        min_clusters = np.shape(similarity_matrix)[0] // 200
        max_clusters = np.shape(similarity_matrix)[0] // 20

        # iterate over the number of clusters
        cluster_range = np.linspace(min_clusters, max_clusters, 5, dtype=int)
        for n_clusters in cluster_range:                
                # run Birch
                birch = Birch(branching_factor=branching_factor, threshold=threshold, n_clusters=n_clusters)
    
                labels = birch.fit_predict(distance_matrix)

                # calculate cluster sizes
                cluster_sizes = np.bincount(labels)
                max_cluster_size = np.max(cluster_sizes)
                min_cluster_size = np.min(cluster_sizes)
    
                # calculate the silhouette score
                silhouette = silhouette_score(distance_matrix, labels)
    
                if min_cluster_size >= min_samples_in_cluster and max_cluster_size <= max_samples_in_cluster and silhouette > best_score:
                    best_score = silhouette
                    best_labels = labels
                    print(f'Best silhouette score: {best_score} with {n_clusters} clusters. Min cluster size: {min_cluster_size}, Max cluster size: {max_cluster_size}  ')

    if best_labels is None:
        best_labels = birch_affinity_labels

    # calculate the silhouette score
    silhouette = silhouette_score(distance_matrix, best_labels)

    print(f'Silhouette Score: {silhouette}')

    return best_labels
