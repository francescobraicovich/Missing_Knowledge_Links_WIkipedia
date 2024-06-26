# Missing Knowledg Links WIkipedia

## Overview
The Missing Links Analysis project aims to identify missing links between Wikipedia pages that should logically be connected. The process involves building a graph of Wikipedia pages, computing similarity scores, clustering the graph, and using machine learning to predict potential missing links.

## Project Structure
The project consists of several key components, each defined in different files. Here's an overview of the files and their purposes:

1. `Missing_links_analysis.ipynb`: The main Jupyter notebook that orchestrates the entire analysis process.
2. `build_graph.py`: Contains functions to build a graph of Wikipedia pages.
3. `neighbors.py`: Provides functionality to compute Jaccard similarity between pairs of nodes.
4. `dbscan.py`: Implements the DBSCAN clustering algorithm on the graph.
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
This script contains functions to construct a graph where nodes represent Wikipedia pages and edges represent links between them. The main function in this file is:
- `build_graph`: Reads Wikipedia data and constructs a NetworkX graph.

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
