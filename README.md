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

This script is responsible for constructing a graph where nodes represent Wikipedia pages and edges represent links between them. The graph is built using data from Wikipedia and applying various filtering and processing techniques to ensure the relevance and quality of the links.

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
    - `save_graph(graph, filename)`: Saves the graph to a file using pickle.
    - `load_graph(filename)`: Loads a graph from a file using pickle.

6. **Parallel Processing**:
    - Utilizes `concurrent.futures` for fetching Wikipedia pages in parallel, speeding up the graph construction process.

#### How it Works

1. **Initialization**:
   - The script starts by initializing the Wikipedia API and defining blacklists for filtering irrelevant links and categories.

2. **Fetching and Filtering Data**:
   - `fetch_wikipedia_page` is used to retrieve a Wikipedia page and extract its links. The links are then filtered using `filter_links` to remove any that contain blacklisted substrings.
   - Categories are similarly filtered using `filter_categories`.

3. **Building the Graph**:
   - The `build_graph` function constructs the graph by iterating through a list of pages, fetching their data, filtering links and categories, and adding the relevant nodes and edges to the NetworkX graph.

4. **Optimization and Parallel Processing**:
   - To handle large volumes of data efficiently, the script uses `concurrent.futures` to fetch page data in parallel, reducing the overall time required to build the graph.

5. **Saving and Loading the Graph**:
   - Once the graph is built, it can be saved to a file using `save_graph` for later use. The graph can also be loaded from a file using `load_graph`, enabling reuse without rebuilding.

#### Reasoning Behind the Code

- **Filtering**: By removing irrelevant links and categories, the script ensures that the graph only contains meaningful connections, improving the quality of subsequent analyses.
- **Parallel Processing**: Leveraging parallel processing significantly speeds up data fetching, making the script more efficient and scalable.
- **Modularity**: Functions are designed to be modular, allowing for easy testing, maintenance, and reuse in different parts of the project.

Overall, this script lays the foundation for building a high-quality graph of Wikipedia pages, which is essential for identifying missing links and conducting further analyses.


### 3. `neighbors.py`
This module provides the functionality to compute the Jaccard similarity between pairs of nodes in the graph. Key functions include:
- `compute_jaccard_similarity`: Computes the Jaccard similarity score for all pairs of nodes in the graph.

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
