# Missing Knowledg Links WIkipedia

## Overview
The Missing Links Analysis project aims to identify missing links between Wikipedia pages that should logically be connected. The process involves building a graph of Wikipedia pages, computing similarity scores, clustering the graph, and using machine learning to predict potential missing links.

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

## Detailed Description of Each File

### 1. `Missing_links_analysis.ipynb`
This Jupyter notebook is the entry point of the project. It integrates all the functions from the other files to perform the complete analysis. The main steps in the notebook are:
- Building a graph of Wikipedia pages.
- Computing pairwise Jaccard similarity.
- Clustering the graph using DBSCAN.
- Identifying missing link candidates based on similarity thresholds.
- Building and training an XGBoost model.
- Predicting missing links.

### 2. `build_graph.py`

This script is responsible for constructing a graph where nodes represent Wikipedia pages and edges represent links between them. The graph is built using data from Wikipedia and applying various filtering and processing techniques to ensure the relevance and quality of the links. Given a certain depth and a starting page, the graph is built by including all wikipedia pages that are at distance equal to the depth from the starting page. After this first step, the graph is completed by adding all links between nodes that are already in the graph. This is the lengthiest part of the process given the elevated number of api requests.

#### Dependencies
The script uses the following libraries:
- `networkx`: For creating and managing the graph.
- `requests`: For making HTTP requests to fetch data.
- `pandas` and `numpy`: For data manipulation and processing.
- `matplotlib`: For plotting and visualization.
- `urllib.parse`: For URL parsing.
- `wikipediaapi`: For accessing Wikipedia data.
- `os` and `pickle`: For file operations and data serialization.
- `concurrent.futures`: For parallel processing.

#### Key Components

1. **Blacklists for Filtering**:
    - `substring_to_remove_links` and `substring_to_remove_categories`: Lists of substrings used to filter out irrelevant links and categories from Wikipedia pages.

2. **Wikipedia API Initialization**:
    - `wiki_wiki`: Initializes a Wikipedia API object for fetching page data.

3. **Filter Functions**:
    - `filter_sentences_vectorised(sentences, words_to_filter)`: Filters out sentences that contain any of the specified words. This function processes the sentences to ensure they are of equal length, splits them into words, and removes sentences containing the specified words.
    - `filter_sentences(sentences, words_to_filter)`: A non-vectorized version of the filtering function for comparison.

4. **Graph Construction**:
    - The script includes a series of functions to build and filter the graph:
      - `fetch_wikipedia_page(title)`: Fetches a Wikipedia page and its links.
      - `build_graph(pages)`: Main function to build the graph. It initializes a NetworkX graph, fetches pages and their links, applies filters, and adds nodes and edges to the graph.
      - `filter_links(links)`: Applies the link filter to remove irrelevant links.
      - `filter_categories(categories)`: Applies the category filter to remove irrelevant categories.

5. **Serialization and Deserialization**:
    - `save_graph(graph, filename)`: Saves the graph to a file using pickle. This step makes it possible to reuse the graph without constructing it again a second time.
    - `load_graph(filename)`: Loads a graph from a file using pickle (only if it was saved before).

6. **Parallel Processing**:
    - Utilizes `concurrent.futures` for fetching Wikipedia pages in parallel, speeding up the graph construction process.

### 3. `neighbors.py`
This script is responsible for calculating the Jaccard similarity between nodes in a graph. The Jaccard similarity is a measure of similarity between two sets, defined as the size of the intersection divided by the size of the union of the sets. In this context, it is used to measure the similarity between pairs of nodes based on their common neighbors. The `neighbors.py` script provides functions to compute the common neighbors and total neighbors between nodes in a graph, and then uses these values to calculate the Jaccard similarity coefficient. The adjacency matrix of the graph is processed to determine the common neighbors, which is then used along with the total number of neighbors to compute the Jaccard similarity. This measure helps identify nodes that are likely to be connected based on their shared connections with other nodes.

#### Dependencies
The script relies on the following libraries:
- `numpy`: For numerical operations and array manipulations.
- `scipy.sparse`: For handling sparse matrices, which are efficient for storing and processing large adjacency matrices.
- `threadpoolctl`: For controlling the number of threads used in computations, optimizing performance.

#### Key Components

1. **Common Neighbors Calculation**:
    - `get_common_neighbors(adj_matrix)`: This function computes the number of common neighbors between nodes in a given adjacency matrix. It converts the adjacency matrix into sparse matrix formats (`csr_matrix` and `csc_matrix`) for efficient computation. The product of these matrices gives the number of common neighbors between each pair of nodes, which is then converted back to a dense matrix.

2. **Total Neighbors Calculation**:
    - `get_total_neighbors(adj_matrix, common_neighbors)`: This function calculates the total number of neighbors for each node. It sums the columns of the adjacency matrix to get the number of neighbors for each node and then computes the outer sum to get the total neighbors between pairs of nodes. The common neighbors and direct connections are subtracted to get the total neighbors matrix, which is adjusted to avoid division by zero.

3. **Jaccard Coefficient Calculation**:
    - `get_jaccard_coefficient(common_neighbors, total_neighbors)`: This function calculates the Jaccard coefficient using the common neighbors and total neighbors matrices. The Jaccard coefficient is computed as the ratio of the number of common neighbors to the total number of neighbors minus the direct connection.

### 4. `dbscan.py`
This file implements the DBSCAN clustering algorithm to group similar nodes together. Main functions are:
- `perform_dbscan`: Applies DBSCAN clustering on the graph based on the Jaccard similarity scores.

### 5. `build_dataset.py`
This script is responsible for creating the training and missing links datasets. Important functions are:
- `create_train_dataset`: Constructs a training dataset with features and labels.
- `create_missing_links_dataset`: Creates a dataset of missing link candidates.

### 6. `tune_model.py`
This module contains functions to train and tune an XGBoost model to predict missing links. Key functions include:
- `train_xgboost_model`: Trains the XGBoost model on the training dataset.
- `tune_model`: Tunes the hyperparameters of the XGBoost model.

### 7. `missing_links.py`
This file uses the trained XGBoost model to predict missing links between Wikipedia pages. Main functions are:
- `predict_missing_links`: Predicts whether there should be a link between pairs of nodes based on the model.

### 8. `requirements.txt`
This file lists all the dependencies required for the project to run. The main dependencies are:
- NetworkX
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Wikipedia-API

## Installation and Setup
To set up the project, follow these steps:

1. Clone the repository to your local machine.
2. Create a virtual environment:
   ```bash
   python -m venv venv
