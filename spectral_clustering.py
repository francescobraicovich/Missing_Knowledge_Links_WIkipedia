import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

def spectral_clustering_optimal_clusters(similarity_matrix):
    # Function to find the optimal number of clusters using eigengap heuristic
    def find_optimal_clusters(eigenvalues):
        eigengap = np.diff(eigenvalues)
        optimal_clusters = np.argmax(eigengap) + 1
        return optimal_clusters

    # Compute the Laplacian matrix
    laplacian = np.diag(np.sum(similarity_matrix, axis=1)) - similarity_matrix
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    
    # Find the optimal number of clusters
    optimal_clusters = find_optimal_clusters(eigenvalues)
    
    # Perform Spectral Clustering
    spectral = SpectralClustering(n_clusters=optimal_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(similarity_matrix)

    # print the silhouette score
    silhouette = silhouette_score(similarity_matrix, labels)
    print(f'Silhouette Score: {silhouette}')
    
    return labels, optimal_clusters