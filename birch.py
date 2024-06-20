
import numpy as np
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation

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

    # run Birch
    birch = Birch(branching_factor=branching_factor, threshold=threshold, n_clusters=AffinityPropagation(damping=0.9))

    labels = birch.fit_predict(distance_matrix)

    # calculate the silhouette score
    silhouette = silhouette_score(distance_matrix, labels)

    print(f'Silhouette Score: {silhouette}')

    return labels
