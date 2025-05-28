import tomllib # Use import toml if Python < 3.11 and toml is in requirements.txt
from pathlib import Path
import logging

CONFIG_FILE_PATH = Path(__file__).parent.parent.parent / "config" / "settings.toml"

class AppConfig:
    def __init__(self, data):
        self.paths = self._Paths(data.get("paths", {}))
        self.graph_params = self._GraphParams(data.get("graph_params", {}))
        self.similarity_params = self._SimilarityParams(data.get("similarity_params", {}))
        self.clustering_params = self._ClusteringParams(data.get("clustering_params", {}))
        self.model_params = self._ModelParams(data.get("model_params", {}))
        self.logging_params = self._LoggingParams(data.get("logging", {}))
        self.title = data.get("title", "Missing Links Analyzer")

    class _Paths:
        def __init__(self, data):
            self.graph_output_dir = Path(data.get("graph_output_dir", "data/graphs/"))
            self.dataset_output_dir = Path(data.get("dataset_output_dir", "data/datasets/"))
            self.model_output_dir = Path(data.get("model_output_dir", "data/models/"))
            self.log_file_path = Path(data.get("log_file_path", "data/logs/analyzer.log"))

    class _GraphParams:
        def __init__(self, data):
            self.start_page = data.get("start_page", "Wikipedia")
            self.depth = int(data.get("depth", 2))
            self.min_links_for_completion = int(data.get("min_links_for_completion", 15))

    class _SimilarityParams:
        def __init__(self, data):
            self.weak_threshold_quantile = float(data.get("weak_threshold_quantile", 0.75))
            self.strong_threshold_quantile = float(data.get("strong_threshold_quantile", 0.95))

    class _ClusteringParams:
        def __init__(self, data):
            self.noise_threshold_ratio = float(data.get("noise_threshold_ratio", 0.05))
            self.subcluster_noise_enabled = bool(data.get("subcluster_noise_enabled", True))
            self.hdbscan_epsilon_values = [float(x) for x in data.get("hdbscan_epsilon_values", [])]


    class _ModelParams:
        def __init__(self, data):
            self.test_split_ratio = float(data.get("test_split_ratio", 0.3))
            self.random_state = int(data.get("random_state", 42))
            self.xgboost_n_estimators = int(data.get("xgboost_n_estimators", 100))
            self.xgboost_max_depth = int(data.get("xgboost_max_depth", 5))
            self.xgboost_alpha = float(data.get("xgboost_alpha", 0.1))

    class _LoggingParams:
         def __init__(self, data):
             self.log_level = data.get("log_level", "INFO").upper()
             
def load_config(config_path: Path = CONFIG_FILE_PATH) -> AppConfig:
    """Loads configuration from the TOML file."""
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f) # Use toml.load(f) if using the toml package
        return AppConfig(data)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        # Fallback to default config if file not found or provide a way to initialize
        return AppConfig({}) # Returns a config with all defaults
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return AppConfig({}) # Returns a config with all defaults

# Load config on module import
settings = load_config()

# Ensure output directories exist
settings.paths.graph_output_dir.mkdir(parents=True, exist_ok=True)
settings.paths.dataset_output_dir.mkdir(parents=True, exist_ok=True)
settings.paths.model_output_dir.mkdir(parents=True, exist_ok=True)
if settings.paths.log_file_path.parent:
     settings.paths.log_file_path.parent.mkdir(parents=True, exist_ok=True)

# Basic logging setup (can be expanded in a dedicated logging utility)
logging.basicConfig(
    level=settings.logging_params.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(), # Log to console
        logging.FileHandler(settings.paths.log_file_path) # Log to file
    ]
)

if __name__ == "__main__":
    # Example of how to access settings
    print(f"Project Title: {settings.title}")
    print(f"Graph output directory: {settings.paths.graph_output_dir}")
    print(f"Start page: {settings.graph_params.start_page}")
    print(f"Log level: {settings.logging_params.log_level}")
    # Test logging
    logging.info("Configuration loaded successfully.")
    logging.debug("This is a debug message.") # Wont show if level is INFO
    logging.warning("This is a test warning.")
