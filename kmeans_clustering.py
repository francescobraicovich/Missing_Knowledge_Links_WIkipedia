from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score

def kmeans_from_similarity(similarity_matrix):

    n = similarity_matrix.shape[0]
    min_clusters = int(similarity_matrix.shape[0] / 500)
    max_clusters = int(similarity_matrix.shape[0] / 50)

    # normalize the similarity matrix
    max_val = np.max(similarity_matrix)
    similarity_matrix = similarity_matrix / max_val

    # Convert the similarity matrix to a distance matrix
    distance_matrix = 1 - similarity_matrix

    print('distance computed')
    
    # Apply MDS to convert the distance matrix into feature vectors
    mds = MDS(n_components=4, dissimilarity="precomputed", n_jobs=-1)
    feature_vectors = mds.fit_transform(distance_matrix)

    print('MDS applied')
    
    # Determine the optimal number of clusters using silhouette score
    best_score = -1
    best_k = min_clusters
    best_labels = None
    
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(feature_vectors)
        
        # calculate the score from the original similarity matrix
        score = silhouette_score(similarity_matrix, labels)
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # print the silhouette score
    print(f'Silhouette Score: {best_score}')
    
    return best_labels
