"""
Main orchestrator class for the Missing Links Analyzer pipeline.
"""
import logging
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List
import numpy as np
import pandas as pd
import networkx as nx
from xgboost import XGBClassifier # For type hinting
import pickle
from sklearn.model_selection import train_test_split


# Project modules
from .config import settings, AppConfig
from .graph_builder import build_graph
from .similarity_calculator import calculate_jaccard_similarity
from .clustering import cluster_nodes_hdbscan
# Note: DEFAULT_COLUMNS was imported from dataset_builder in previous attempt, ensure it's available or defined here if needed for predict_df feature selection
from .dataset_builder import build_feature_dataset, DEFAULT_MAX_TRAINING_SAMPLES, DEFAULT_COLUMNS 
from .model_trainer import train_and_tune_model
from .link_predictor import identify_candidate_links_by_similarity

logger = logging.getLogger(__name__)

class MissingLinksAnalyzer:
    """
    Orchestrates the entire pipeline for identifying missing links in a graph.
    """

    def __init__(self, config: AppConfig = settings):
        """
        Initializes the MissingLinksAnalyzer with a configuration.

        Args:
            config (AppConfig, optional): Configuration object. Defaults to global settings.
        """
        self.config: AppConfig = config
        
        # Data attributes - initialized to None
        self.graph: Optional[nx.Graph] = None
        self.links_dict: Optional[Dict[str, np.ndarray]] = None
        self.categories_dict: Optional[Dict[str, np.ndarray]] = None
        
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.common_neighbors_matrix: Optional[np.ndarray] = None
        self.total_neighbors_matrix: Optional[np.ndarray] = None # Union of neighbors
        self.jaccard_similarity_matrix: Optional[np.ndarray] = None
        
        self.cluster_labels: Optional[np.ndarray] = None
        
        self.weak_similarity_threshold: Optional[float] = None
        self.strong_similarity_threshold: Optional[float] = None
        
        self.candidate_links_dict: Optional[Dict[Tuple[str, str], float]] = None
        self.candidate_links_matrix: Optional[np.ndarray] = None # Matrix form of candidates
        
        self.train_df: Optional[pd.DataFrame] = None
        self.predict_df: Optional[pd.DataFrame] = None # For features of candidate links
        
        self.trained_model: Optional[XGBClassifier] = None
        self.model_params: Optional[Dict[str, Any]] = None # Stores best hyperparams including 'optimal_threshold'
        self.model_test_metrics: Optional[Dict[str, Any]] = None
        
        self.final_missing_links_df: Optional[pd.DataFrame] = None
        
        self._filtered_categories_cache: Dict[str, set[str]] = {} # Internal cache for category filtering

        logger.info(f"MissingLinksAnalyzer initialized with config: {self.config.title}")

    def _ensure_data_loaded(self, required_attrs: List[str]) -> None:
        """
        Checks if prerequisite data attributes are loaded before running a step.

        Args:
            required_attrs (List[str]): A list of attribute names that must not be None.

        Raises:
            RuntimeError: If any of the required attributes is None.
        """
        for attr_name in required_attrs:
            if getattr(self, attr_name, None) is None: # Added default to getattr for safety
                error_msg = (
                    f"Prerequisite data '{attr_name}' not loaded or generated. "
                    f"Please run the appropriate preceding steps in the pipeline."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        logger.debug(f"All required attributes {required_attrs} are loaded.")

    def load_or_build_graph(self) -> None:
        """
        Loads an existing graph or builds a new one based on configuration.
        Stores graph, links_dict, and categories_dict.
        """
        logger.info("Step 1: Loading or Building Graph...")
        graph_data = build_graph(
            start_page=self.config.graph_params.start_page,
            depth=self.config.graph_params.depth,
            min_links_completion=self.config.graph_params.min_links_for_completion
        )
        if graph_data[0] is not None: 
            self.graph, self.links_dict, self.categories_dict = graph_data
            logger.info(f"Graph for '{self.config.graph_params.start_page}' (Depth {self.config.graph_params.depth}) "
                        f"loaded/built successfully: {self.graph.number_of_nodes()} nodes, "
                        f"{self.graph.number_of_edges()} edges.")
        else:
            # Raise an error to halt pipeline if graph is essential and fails to load/build
            error_msg = "Graph building/loading failed. Check graph_builder logs. Cannot proceed."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def calculate_similarity(self) -> None:
        """
        Calculates Jaccard similarity and related matrices for the graph.
        Requires self.graph to be loaded.
        """
        logger.info("Step 2: Calculating Similarity Matrices...")
        self._ensure_data_loaded(["graph"])
        assert self.graph is not None 
        
        (self.adjacency_matrix, 
         self.common_neighbors_matrix, 
         self.total_neighbors_matrix, 
         self.jaccard_similarity_matrix) = calculate_jaccard_similarity(self.graph)
        
        logger.info("Jaccard similarity and related matrices calculated successfully.")
        logger.debug(f"Adjacency matrix shape: {self.adjacency_matrix.shape if self.adjacency_matrix is not None else 'N/A'}")
        logger.debug(f"Jaccard similarity matrix shape: {self.jaccard_similarity_matrix.shape if self.jaccard_similarity_matrix is not None else 'N/A'}")

    def perform_clustering(self) -> None:
        """
        Performs node clustering based on similarity and adjacency.
        Requires jaccard_similarity_matrix and adjacency_matrix.
        """
        logger.info("Step 3: Performing Node Clustering...")
        self._ensure_data_loaded(["jaccard_similarity_matrix", "adjacency_matrix"])
        assert self.jaccard_similarity_matrix is not None and self.adjacency_matrix is not None

        clustering_input_matrix = self.jaccard_similarity_matrix + self.adjacency_matrix
        np.fill_diagonal(clustering_input_matrix, np.max(clustering_input_matrix)) 
        
        logger.info("Using HDBSCAN for clustering.")
        self.cluster_labels = cluster_nodes_hdbscan(
            similarity_matrix=clustering_input_matrix,
            cluster_selection_epsilon_values=self.config.clustering_params.hdbscan_epsilon_values,
            noise_threshold_ratio=self.config.clustering_params.noise_threshold_ratio,
            subcluster_noise_enabled=self.config.clustering_params.subcluster_noise_enabled
            # min_cluster_size and min_samples will use defaults in cluster_nodes_hdbscan
        )
        num_unique_labels = len(np.unique(self.cluster_labels[self.cluster_labels != -1])) if self.cluster_labels is not None else 0
        logger.info(f"Clustering complete. Found {num_unique_labels} clusters "
                    f"(excluding noise). Total nodes processed: {len(self.cluster_labels) if self.cluster_labels is not None else 0}.")

    def determine_similarity_thresholds(self) -> None:
        """
        Determines weak and strong similarity thresholds based on existing link similarities.
        Requires jaccard_similarity_matrix and adjacency_matrix.
        """
        logger.info("Step 4: Determining Similarity Thresholds...")
        self._ensure_data_loaded(["jaccard_similarity_matrix", "adjacency_matrix"])
        assert self.jaccard_similarity_matrix is not None and self.adjacency_matrix is not None

        boolean_adj_matrix = self.adjacency_matrix > 0
        existing_link_similarities = self.jaccard_similarity_matrix[boolean_adj_matrix]
        
        if len(existing_link_similarities) == 0:
            logger.warning("No existing links found. Using fallback thresholds: weak=0.1, strong=0.2")
            self.weak_similarity_threshold = 0.1 
            self.strong_similarity_threshold = 0.2
        else:
            self.weak_similarity_threshold = float(np.quantile(
                existing_link_similarities, self.config.similarity_params.weak_threshold_quantile))
            self.strong_similarity_threshold = float(np.quantile(
                existing_link_similarities, self.config.similarity_params.strong_threshold_quantile))

        logger.info(f"Weak similarity threshold: {self.weak_similarity_threshold:.4f}")
        logger.info(f"Strong similarity threshold: {self.strong_similarity_threshold:.4f}")

    def identify_initial_candidates(self) -> None:
        """
        Identifies initial candidate missing links using similarity and clustering.
        """
        logger.info("Step 5: Identifying Initial Candidate Links...")
        required = ["graph", "jaccard_similarity_matrix", "cluster_labels", 
                    "weak_similarity_threshold", "strong_similarity_threshold"]
        self._ensure_data_loaded(required)
        assert self.graph is not None and \
               self.jaccard_similarity_matrix is not None and \
               self.cluster_labels is not None and \
               self.weak_similarity_threshold is not None and \
               self.strong_similarity_threshold is not None
        
        self.candidate_links_dict, self.candidate_links_matrix = identify_candidate_links_by_similarity(
            graph=self.graph,
            similarity_matrix=self.jaccard_similarity_matrix,
            cluster_labels=self.cluster_labels,
            weak_threshold=self.weak_similarity_threshold,
            strong_threshold=self.strong_similarity_threshold
        )
        logger.info(f"Identified {len(self.candidate_links_dict)} initial candidate links.")

    def build_ml_datasets(self) -> None:
        """ Builds training and prediction datasets for the machine learning model. """
        logger.info("Step 6: Building Machine Learning Datasets...")
        required = ["graph", "adjacency_matrix", "jaccard_similarity_matrix", 
                    "common_neighbors_matrix", "total_neighbors_matrix", 
                    "cluster_labels", "categories_dict", "candidate_links_matrix"]
        self._ensure_data_loaded(required)
        # Assertions for mypy
        assert self.graph is not None and self.adjacency_matrix is not None and \
               self.jaccard_similarity_matrix is not None and self.common_neighbors_matrix is not None and \
               self.total_neighbors_matrix is not None and self.cluster_labels is not None and \
               self.categories_dict is not None and self.candidate_links_matrix is not None

        logger.info("Building 'train' dataset...")
        self.train_df, self._filtered_categories_cache = build_feature_dataset(
            graph=self.graph, adjacency_matrix=self.adjacency_matrix,
            similarity_matrix=self.jaccard_similarity_matrix,
            missing_link_candidates_matrix=self.adjacency_matrix, # Exclude existing links from negative sampling
            common_neighbors_matrix=self.common_neighbors_matrix,
            total_neighbors_matrix=self.total_neighbors_matrix,
            cluster_labels=self.cluster_labels, categories_dict=self.categories_dict,
            dataset_type='train', filtered_categories_cache=self._filtered_categories_cache,
            # Using DEFAULT_MAX_TRAINING_SAMPLES; can be made configurable via self.config if needed
            max_training_samples=DEFAULT_MAX_TRAINING_SAMPLES 
        )
        logger.info(f"Built 'train' dataset with {len(self.train_df) if self.train_df is not None else 0} samples.")

        logger.info("Building 'predict' dataset for candidate links...")
        self.predict_df, self._filtered_categories_cache = build_feature_dataset(
            graph=self.graph, adjacency_matrix=self.adjacency_matrix,
            similarity_matrix=self.jaccard_similarity_matrix,
            missing_link_candidates_matrix=self.candidate_links_matrix, # Use identified candidates
            common_neighbors_matrix=self.common_neighbors_matrix,
            total_neighbors_matrix=self.total_neighbors_matrix,
            cluster_labels=self.cluster_labels, categories_dict=self.categories_dict,
            dataset_type='predict', filtered_categories_cache=self._filtered_categories_cache
        )
        logger.info(f"Built 'predict' dataset with {len(self.predict_df) if self.predict_df is not None else 0} candidate links.")

    def train_link_model(self) -> None:
        """ Trains and tunes the link prediction model. """
        logger.info("Step 7: Training Link Prediction Model...")
        self._ensure_data_loaded(["train_df"])
        assert self.train_df is not None

        if self.train_df.empty or 'link' not in self.train_df.columns or len(self.train_df['link'].unique()) < 2:
            logger.error("Training data is unsuitable. Skipping model training.")
            return

        features = [col for col in DEFAULT_COLUMNS if col not in ['node_1', 'node_2', 'link']]
        X = self.train_df[features]
        y = self.train_df['link']

        X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
            X, y, test_size=self.config.model_params.test_split_ratio, 
            random_state=self.config.model_params.random_state, stratify=y)
        
        scale_pos_weight = (y_train_ml.value_counts().get(0,0) / y_train_ml.value_counts().get(1,1)) if y_train_ml.value_counts().get(1,1) > 0 else 1
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        model_base_params = {'objective': 'binary:logistic', 'scale_pos_weight': scale_pos_weight,
                             'random_state': self.config.model_params.random_state,
                             'use_label_encoder': False, 'eval_metric': 'logloss'}
        
        hyperparam_search_space = {
            'n_estimators': [self.config.model_params.xgboost_n_estimators], 
            'max_depth': [self.config.model_params.xgboost_max_depth],
            'alpha': [self.config.model_params.xgboost_alpha]
        }

        self.trained_model, self.model_params, self.model_test_metrics = train_and_tune_model(
            X_train=X_train_ml, y_train=y_train_ml, X_test=X_test_ml, y_test=y_test_ml,
            model_base_params=model_base_params, hyperparam_search_space=hyperparam_search_space)
        
        logger.info(f"Model training complete. Best hyperparams: {self.model_params}")
        logger.info(f"Model test metrics: {self.model_test_metrics}")

        if self.trained_model:
            model_filename = f"{self.config.graph_params.start_page.replace(' ', '_')}_depth{self.config.graph_params.depth}_xgb_model.pkl"
            model_path = self.config.paths.model_output_dir / model_filename
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_path, 'wb') as f: 
                    pickle.dump(self.trained_model, f)
                logger.info(f"Trained model saved to: {model_path}")
            except (IOError, pickle.PicklingError) as e: 
                logger.exception(f"Error saving trained model to {model_path}: {e}")
            except Exception as e: # Catch any other unexpected errors during saving
                 logger.exception(f"Unexpected error saving trained model to {model_path}: {e}")


    def predict_final_missing_links(self) -> None:
        """ Predicts probabilities for candidate links and applies threshold. """
        logger.info("Step 8: Predicting Final Missing Links...")
        required = ["predict_df", "trained_model", "model_params"]
        self._ensure_data_loaded(required)
        assert self.predict_df is not None and self.trained_model is not None and self.model_params is not None
        
        if self.predict_df.empty:
            logger.warning("Prediction dataset is empty. No links to predict.")
            self.final_missing_links_df = pd.DataFrame(columns=['node_1', 'node_2', 'probability', 'prediction'])
            return

        train_features = [col for col in DEFAULT_COLUMNS if col not in ['node_1', 'node_2', 'link']]
        missing_features = [f for f in train_features if f not in self.predict_df.columns]
        if missing_features:
            logger.error(f"Predict_df missing features: {missing_features}. Cannot predict.")
            self.final_missing_links_df = pd.DataFrame(columns=['node_1', 'node_2', 'probability', 'prediction'])
            return
            
        X_predict = self.predict_df[train_features]
        link_probabilities = self.trained_model.predict_proba(X_predict)[:, 1]
        optimal_threshold = self.model_params.get('optimal_threshold', 0.5)
        
        binary_predictions = (link_probabilities >= optimal_threshold).astype(int)
        
        self.final_missing_links_df = pd.DataFrame({
            'node_1': self.predict_df['node_1'], 'node_2': self.predict_df['node_2'],
            'probability': link_probabilities, 'prediction': binary_predictions
        }).sort_values(by='probability', ascending=False)
        
        logger.info(f"Generated final predictions. Top 5:\n{self.final_missing_links_df.head()}")

    def run_full_pipeline(self, steps: Optional[List[str]] = None) -> None:
        """ Orchestrates the full pipeline. """
        logger.info("Starting full missing links analysis pipeline...")
        if steps: logger.warning("Selective step execution not fully implemented; running all steps.")

        pipeline_methods = [
            self.load_or_build_graph, self.calculate_similarity, self.perform_clustering,
            self.determine_similarity_thresholds, self.identify_initial_candidates,
            self.build_ml_datasets, self.train_link_model, self.predict_final_missing_links
        ]
        for method in pipeline_methods:
            try:
                method()
            except RuntimeError as e:
                logger.error(f"Pipeline halted at '{method.__name__}': {e}")
                return
            except Exception as e:
                logger.error(f"Unexpected error at '{method.__name__}': {e}", exc_info=True)
                return
        logger.info("Full pipeline completed successfully.")