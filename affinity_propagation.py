import numpy as np
from sklearn.cluster import AffinityPropagation
from io import StringIO
import warnings
from sklearn.metrics import silhouette_score
import joblib
import sys

def cluster_with_convergence_bool(similarity_arr, damping):
    # Redirect stdout to a StringIO buffer
    original_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Call the script function
        clustering  = AffinityPropagation(affinity='precomputed', damping=damping, verbose=True).fit_predict(similarity_arr)

        # Get the printed output from the buffer
        printed_output = sys.stdout.getvalue()
        convergence = True
        if 'Did not converge' in printed_output:
            convergence = False
        return clustering, convergence

    finally:
        # Restore the original stdout
        sys.stdout = original_stdout

def cluster(similarity_arr, max_reps=5, damping_start=0.9):
    # Run the clustering function multiple times to check for convergence
    damping_step = (1 - damping_start) / (max_reps)

    for i in range(max_reps):
        damping = damping_start + i*damping_step
        clustering, convergence = cluster_with_convergence_bool(similarity_arr, damping)
        if convergence:

            # Print the silhouette score
            silhouette = silhouette_score(similarity_arr, clustering)
            print(f'Silhouette Score: {silhouette}')
            
            return clustering

    # If the function did not converge, raise a warning
    warnings.warn(f"The clustering algorithm did not converge {max_reps} multiple attempts.")
    return clustering