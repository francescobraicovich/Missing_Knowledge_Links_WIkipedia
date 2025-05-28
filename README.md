# Missing Links Analyzer

## Summary
The Missing Links Analyzer is a Python project designed to identify potential missing links (undiscovered or unacknowledged connections) within a network of Wikipedia pages. It achieves this by building a graph from a specified start page, analyzing page similarities and clustering, and then training a machine learning model to predict and rank these potential missing links.

## Features and Functionality
*   **Graph Construction:** Builds a directed graph of Wikipedia pages starting from a user-defined page and crawling to a specified depth.
*   **Similarity Calculation:** Computes Jaccard similarity coefficients between all pairs of pages in the graph based on their common neighbors.
*   **Node Clustering:** Applies the HDBSCAN algorithm to group similar pages together, helping to identify communities or related topics.
*   **Candidate Link Identification:** Identifies potential missing links using two main strategies:
    *   Pairs of pages within the same cluster that have a similarity score above a "weak" threshold.
    *   Any pair of pages across the graph with a similarity score above a "strong" threshold.
*   **Feature Engineering:** Constructs a rich feature dataset for each candidate link, including graph-based metrics (degrees, common neighbors), similarity scores, category-based features (common/total categories), and cluster information.
*   **Machine Learning Model Training:** Trains an XGBoost classifier to predict the likelihood of a missing link existing between two pages. Includes hyperparameter tuning (currently placeholder, uses fixed params from config) and evaluation.
*   **Prediction & Ranking:** Uses the trained model to predict the probability of each candidate link being a true missing link and ranks them accordingly.
*   **Configuration:** The entire pipeline is configurable through a central `settings.toml` file, allowing users to customize paths, graph parameters, similarity thresholds, clustering settings, and model parameters.
*   **Command-Line Interface:** Provides a CLI script (`scripts/run_analysis.py`) to execute the full analysis pipeline with default or custom configurations.

## Directory Structure
*   **`src/missing_links_analyzer/`**: Contains the core Python source code for all modules of the analyzer (graph building, similarity calculation, clustering, dataset creation, model training, link prediction, configuration management, and the main analyzer orchestrator).
*   **`tests/`**: Includes unit tests for various components of the project, ensuring functionality and correctness.
*   **`scripts/`**: Contains command-line interface scripts, primarily `run_analysis.py` to execute the pipeline.
*   **`config/`**: Holds configuration files, with `settings.toml` being the primary file for pipeline parameters.
*   **`data/`**: Default directory for storing output data such as generated graphs, datasets, trained models, and logs. (Note: This directory is in `.gitignore`).

## Installation Instructions

### Prerequisites
*   Python 3.11 or higher.
*   Git (for cloning the repository).

### Steps
1.  **Clone the Repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd missing-links-analyzer # Or your repository's root folder name
    ```

2.  **Set Up a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    ```
    Activate the environment:
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    Ensure your virtual environment is activated, then run:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration
The primary configuration for the Missing Links Analyzer is managed through the `config/settings.toml` file. This file uses the TOML format for easy readability and modification.

Key configurable parameters include:
*   **Paths:** Output directories for graphs, datasets, models, and logs (`paths.graph_output_dir`, `paths.log_file_path`, etc.).
*   **Graph Parameters:** The starting Wikipedia page for graph construction (`graph_params.start_page`), crawling depth (`graph_params.depth`), and minimum links for graph completion (`graph_params.min_links_for_completion`).
*   **Similarity Parameters:** Quantiles for determining weak and strong similarity thresholds (`similarity_params.weak_threshold_quantile`, `similarity_params.strong_threshold_quantile`).
*   **Clustering Parameters:** HDBSCAN settings like `noise_threshold_ratio`, `subcluster_noise_enabled`, and `hdbscan_epsilon_values`.
*   **Model Parameters:** XGBoost model settings such as `xgboost_n_estimators`, `xgboost_max_depth`, `xgboost_alpha`, as well as `test_split_ratio` and `random_state` for training.
*   **Logging:** Log level for console and file output (`logging.log_level`).

To use a custom configuration:
1.  Copy the default `config/settings.toml` to a new file (e.g., `config/my_custom_settings.toml`).
2.  Modify the parameters in your new file as needed.
3.  Run the analysis script using the `--config` flag (see Usage section).

## Usage

### Command-Line Interface
The primary way to run the analysis is via the `run_analysis.py` script located in the `scripts/` directory.

1.  **Run with Default Configuration:**
    Ensure your virtual environment is activated and you are in the project's root directory.
    ```bash
    python scripts/run_analysis.py
    ```
    This will use the parameters defined in `config/settings.toml`.

2.  **Run with a Custom Configuration:**
    ```bash
    python scripts/run_analysis.py --config path/to/your_custom_settings.toml
    ```
    Replace `path/to/your_custom_settings.toml` with the actual path to your custom configuration file.

### Programmatic Usage (High-Level Example)
The `MissingLinksAnalyzer` class can also be used programmatically within your own Python scripts:
```python
# Ensure the project root is in PYTHONPATH or use appropriate relative imports
# from src.missing_links_analyzer.analyzer import MissingLinksAnalyzer
# from src.missing_links_analyzer.config import load_config, AppConfig
#
# # Load default configuration
# # analyzer_default = MissingLinksAnalyzer() 
# # analyzer_default.run_full_pipeline()
# # default_results = analyzer_default.final_missing_links_df
# # if default_results is not None:
# #    print("Top missing links (default config):")
# #    print(default_results.head())
#
# # Load a custom configuration (if you have one)
# try:
#     custom_config_path = "config/settings.toml" # Replace with your custom config path if different
#     custom_config = load_config(custom_config_path)
#     analyzer_custom = MissingLinksAnalyzer(config=custom_config)
#     analyzer_custom.run_full_pipeline()
#     custom_results = analyzer_custom.final_missing_links_df
#     if custom_results is not None:
#         print("\\nTop missing links (custom config):")
#         print(custom_results.head())
# except FileNotFoundError:
#     print(f"Custom config file not found at {custom_config_path}. Skipping programmatic custom run example.")
# except Exception as e:
#     print(f"An error occurred during programmatic execution: {e}")
```
*(Note: The programmatic example is illustrative. You might need to adjust paths or ensure the `src` directory is correctly recognized by Python's import system, which is handled by `sys.path` manipulation in `scripts/run_analysis.py`)*

## How to Run Tests
Unit tests are located in the `tests/` directory and can be run using Python's `unittest` module.

1.  **Discover and Run All Tests:**
    From the project's root directory:
    ```bash
    python -m unittest discover tests
    ```

2.  **Run a Specific Test File:**
    For example, to run tests only in `test_utils.py`:
    ```bash
    python -m unittest tests.test_utils # Note: use module dot notation
    ```
    Or, if in the `tests` directory:
    ```bash
    python -m unittest test_utils.py
    ```

## License
MIT License (Placeholder)
*(This project is currently provided without a formal license. Users should assume all rights are reserved unless a license file (e.g., LICENSE.md) is added to the repository.)*

## Acknowledgements
(Acknowledgements to be added if any)

```
