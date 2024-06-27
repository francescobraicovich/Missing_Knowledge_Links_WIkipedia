# Missing Knowledge Links WIkipedia

## Overview
This Missing Links Analysis project aims to identify missing links between Wikipedia pages that should logically be connected. The process involves building a graph of Wikipedia pages, computing similarity scores, clustering the graph, and using machine learning to predict potential missing links.

## Project Structure
The project consists of several key components, each defined in different files. Here's an overview of the files and their purposes:

1. `Missing_links_analysis.ipynb`: The main Jupyter notebook that orchestrates the entire analysis process.
2. `build_graph.py`: Contains functions to build a graph of Wikipedia pages.
3. `neighbors.py`: Provides functionality to compute Jaccard similarity between pairs of nodes.
4. `dbscan.py`: Implements the HDBSCAN clustering algorithm on the graph.
5. `build_dataset.py`: Constructs the training and missing links datasets.
6. `tune_model.py`: Contains functions to train and tune an XGBoost model.
7. `missing_links.py`: Uses the trained model to predict missing links.
8. `requirements.txt`: Lists all dependencies required for the project.

## Examples of Missing Links:
- `start_page`: **Pythagorean theorem**
- `depth`: 2 (1 is too low, 3 requires too much time)
- Top 5 missing links:

- Top 5 missing links in the same cluster as the **Pythagorean theorem**:

- Top 5 missing links between nodes in different clusters:
  

## Detailed Description of Each File

### 1. `Missing_links_analysis.ipynb`
This Jupyter notebook is the entry point of the project. It integrates all the functions from the other files to perform the complete analysis. The main steps in the notebook are:
- Building a graph of Wikipedia pages.
- Computing pairwise Jaccard similarity.
- Clustering the graph using DBSCAN.
- Identifying missing link candidates based on similarity thresholds.
- Building and training an XGBoost model that is able to predict weather two nodes should be connected
- Predicting missing links:
    - Missing links in the graph.
    - Missing links in the same cluster as the starting page
    - Missing links that include the starting page.
    - Missing links between nodes that are in different clusters.

### 2. `build_graph.py`

This script is responsible for constructing a graph where nodes represent Wikipedia pages and edges represent links between them. The graph is built using data from Wikipedia and applying various filtering and processing techniques to ensure the relevance and quality of the links. Given a certain depth and a starting page, the graph is built by including all wikipedia pages that are at distance equal to the depth from the starting page. After this first step, the graph is completed by adding all links between nodes that are already in the graph. This is the lengthiest part of the process given the elevated number of api requests.

#### Dependencies
- `networkx`: For creating and managing the graph.
- `requests`: For making HTTP requests to fetch data.
- `pandas` and `numpy`: For data manipulation and processing.
- `matplotlib`: For plotting and visualization.
- `urllib.parse`: For URL parsing.
- `wikipediaapi`: For accessing Wikipedia data.
- `os` and `pickle`: For file operations and data serialization.
- `concurrent.futures`: For parallel processing.

#### Key features

1. **Blacklists for Filtering**:
    - `substring_to_remove_links` and `substring_to_remove_categories`: Lists of substrings used to filter out irrelevant links and categories from Wikipedia pages.

2. **Wikipedia API Initialization**:
    - `wiki_wiki`: Initializes a Wikipedia API object for fetching page data.

3. **Filter Functions**:
    - `filter_sentences_vectorised`: Filters out sentences that contain any of the specified words. This function processes the sentences to ensure they are of equal length, splits them into words, and removes sentences containing the specified words.
    - `filter_sentences`: A non-vectorized version of the filtering function for comparison.

4. **Graph Construction**:
    - The script includes a series of functions to build and filter the graph:
    - `fetch_wikipedia_page`: Fetches a Wikipedia page and its links.
    - `build_graph`: Main function to build the graph. It initializes a NetworkX graph, fetches pages and their links, applies filters, and adds nodes and edges to the graph.
    - `filter_links`: Applies the link filter to remove irrelevant links.
    - `filter_categories`: Applies the category filter to remove irrelevant categories.

5. **Serialization and Deserialization**:
    - `save_graph`: Saves the graph to a file using pickle. This step makes it possible to reuse the graph without constructing it again a second time.
    - `load_graph`: Loads a graph from a file using pickle (only if it was saved before).

6. **Parallel Processing**:
    - Utilizes `concurrent.futures` for fetching Wikipedia pages in parallel, speeding up the graph construction process.

### 3. `neighbors.py`
This script is responsible for calculating the Jaccard similarity between nodes in a graph. The Jaccard similarity is a measure of similarity between two sets, defined as the size of the intersection divided by the size of the union of the sets. In this context, it is used to measure the similarity between pairs of nodes based on their common neighbors. The `neighbors.py` script provides functions to compute the common neighbors and total neighbors between nodes in a graph, and then uses these values to calculate the Jaccard similarity coefficient. The adjacency matrix of the graph is processed to determine the common neighbors, which is then used along with the total number of neighbors to compute the Jaccard similarity. This measure helps identify nodes that are likely to be connected based on their shared connections with other nodes.

#### Dependencies
- `numpy`: For numerical operations and array manipulations.
- `scipy.sparse`: For handling sparse matrices, which are efficient for storing and processing large adjacency matrices.
- `threadpoolctl`: For controlling the number of threads used in computations, optimizing performance.

#### Functions
- `get_common_neighbors`: This function computes the number of common neighbors between nodes in a given adjacency matrix. It converts the adjacency matrix into sparse matrix formats (`csr_matrix` and `csc_matrix`) for efficient computation. The product of these matrices gives the number of common neighbors between each pair of nodes, which is then converted back to a dense matrix.
- `get_total_neighbors`: This function calculates the total number of neighbors for each node. It sums the columns of the adjacency matrix to get the number of neighbors for each node and then computes the outer sum to get the total neighbors between pairs of nodes. The common neighbors and direct connections are subtracted to get the total neighbors matrix, which is adjusted to avoid division by zero.
- `get_jaccard_coefficient`: This function calculates the Jaccard coefficient using the common neighbors and total neighbors matrices. The Jaccard coefficient is computed as the ratio of the number of common neighbors to the total number of neighbors minus the direct connection.

### 4. `dbscan.py`
This script is responsible for performing DBSCAN clustering based on a similarity matrix. DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are closely packed together, marking points that are in low-density regions as outliers. The `dbscan.py` script uses the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) variant, which improves upon DBSCAN by handling varying densities. This script takes a similarity matrix as input, converts it into a distance matrix, and applies the HDBSCAN algorithm to cluster the nodes. It iterates over a range of parameters to find the best clustering solution based on the silhouette score, a measure of how similar an object is to its own cluster compared to other clusters.

#### Dependencies
The script relies on the following libraries:
- `numpy`: For numerical operations and array manipulations.
- `sklearn.cluster.HDBSCAN`: For performing the HDBSCAN clustering algorithm.
- `sklearn.metrics`: For calculating the silhouette score to evaluate clustering quality.

#### Functions

- `dbscan_from_similarity`. This function performs DBSCAN clustering based on a similarity matrix. It normalizes the similarity matrix, converts it to a distance matrix, and applies HDBSCAN with a range of `cluster_selection_epsilon` values to find the best clustering solution. The best solution is determined based on the silhouette score, ensuring the clusters are well-defined and distinct. Moreover the function avoids having noise clusters exessively big by reclustering noise into new clusters when this happens.

### 5. `missing_links.py`
This code identifies potential missing links in a given graph based on a similarity matrix, cluster labels and two similarity thresholds. It helps to predict which connections could exist but are not currently present in the graph. Each couple of nodes that is in the same cluster and has a similarity score higher than the weak similarity threshold is saved as a missing link candidate. The same for couple of nodes that have similarity score higher than the strong threshold, even if not in the same cluster.

#### Dependencies
- `networkx`
- `numpy`
- `concurrent.futures`

#### Functions

- `find_missing_link_candidates`: This function identifies potential missing links within a graph by analyzing the similarity scores between nodes and their cluster labels. It uses given thresholds to determine which links are missing and generates a dictionary and matrix indicating these potential links. The function can also focus on specific clusters or nodes if specified.

- `print_missing_links_dict`: This function prints the top N missing links and their similarity scores from a dictionary of potential missing links. By default, it prints the top 10 missing links, providing a quick overview of the most significant missing connections in the graph.

### 6. `build_dataset.py`
This code constructs datasets from a given graph computing various features for each pair of nodes. The features included in the dataset are: node degrees, common neighbors, total neighbors, similarity scores, common categories, total categories, number of categories for each node, and cluster labels. These features are obtained through parallel processing, using thread pools to efficiently compute the required attributes for each node pair. The datasets created are either for training (`train`) or for predicting missing links (`missing links`). The `train` dataset includes randomly chosen couples of nodes making sure they do not include the missing link candidates, whilst the `missing links` dataset is made up of the missing link candidates. The datasets are then fed to an XGBoost model that predicts weather two nodes should be linked or not.

## Requirements
- `pandas`
- `numpy`
- `tqdm`
- `concurrent.futures` (part of the standard library)

## Functions

- `filter_sentences`: This function filters out sentences that contain any of the specified words from a list. It converts all sentences to lowercase, creates a mask for sentences containing the unwanted words, and returns a set of filtered sentences. This is to remove unwanted categories such as: 'Short aritcles' or 'Articles that need verification'.

- `find_indices_of_links`: This function finds the indices of links based on the given DataFrame type and a missing link matrix. It returns the row and column indices of the links for either 'missing links' or 'train dataset'.

- `process_node_pair`: This function processes a pair of nodes to compute various features such as degrees, common neighbors, and categories. It filters categories using a predefined blacklist and returns a dictionary with the computed features for the node pair.

- `build_dataset_multi_thread`: This function builds a dataset by extracting node names, finding link indices, and computing features for each pair of nodes using a thread pool for parallel processing. It constructs a DataFrame with the computed features and returns it along with an updated filtered categories dictionary.

### 7. `tune_model.py`
This module contains functions to train and tune an XGBoost model to predict missing links. Key functions include:
- `tune_model`: Tunes the hyperparameters of a ML model.

### 8. `requirements.txt`
This file lists all the dependencies required for the project to run. The main dependencies are:
- NetworkX
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Wikipedia-API
