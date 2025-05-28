import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
import tomllib # Standard library, but need to patch its use within config.py

# Adjust import path based on your project structure if tests are run from a different root
from src.missing_links_analyzer.config import load_config, AppConfig, CONFIG_FILE_PATH, settings as default_settings_instance

# A dummy TOML content string for successful load testing if needed, or for mocking.
DUMMY_TOML_CONTENT_VALID = """
title = "Dummy Test Configuration"

[paths]
graph_output_dir = "data/test_graphs/"
dataset_output_dir = "data/test_datasets/"
model_output_dir = "data/test_models/"
log_file_path = "data/logs/test_analyzer.log"

[graph_params]
start_page = "TestPage"
depth = 1
min_links_for_completion = 5

[similarity_params]
weak_threshold_quantile = 0.7
strong_threshold_quantile = 0.9

[clustering_params]
noise_threshold_ratio = 0.1
subcluster_noise_enabled = false
hdbscan_epsilon_values = [0.2, 0.4]

[model_params]
test_split_ratio = 0.25
random_state = 100
xgboost_n_estimators = 50
xgboost_max_depth = 3
xgboost_alpha = 0.05

[logging]
log_level = "DEBUG"
"""

class TestConfigLoading(unittest.TestCase):

    def test_load_default_config_exists_and_parses(self):
        """ Test that the default CONFIG_FILE_PATH exists and loads correctly. """
        self.assertTrue(CONFIG_FILE_PATH.is_file(), f"Default config file {CONFIG_FILE_PATH} should exist.")
        
        # default_settings_instance is loaded when config.py is imported.
        config = default_settings_instance 
        self.assertIsInstance(config, AppConfig)
        
        # Check a sample value from your actual default settings.toml
        # This value needs to match what's in config/settings.toml
        self.assertEqual(config.graph_params.depth, 2) 
        self.assertEqual(config.title, "Missing Links Analyzer Configuration") 
        
        # Verify directory creation side-effect (these directories are created on module import of config.py)
        self.assertTrue(Path(config.paths.graph_output_dir).exists())
        self.assertTrue(Path(config.paths.dataset_output_dir).exists())
        self.assertTrue(Path(config.paths.model_output_dir).exists())
        # Check log directory parent if log_file_path is not in current dir
        if config.paths.log_file_path.parent != Path("."):
             self.assertTrue(config.paths.log_file_path.parent.exists())


    @patch('builtins.open', side_effect=FileNotFoundError("Mocked FileNotFoundError by open"))
    def test_load_config_file_not_found_returns_default(self, mock_file_open):
        """ Test that load_config returns a default AppConfig if the file is not found. """
        # The 'open' builtin is patched to raise FileNotFoundError when called by load_config
        config = load_config(Path("non_existent_config.toml"))
        
        self.assertIsInstance(config, AppConfig)
        # Check a default value that AppConfig sets when data is empty or missing
        self.assertEqual(config.graph_params.start_page, "Wikipedia") # Default from AppConfig._GraphParams
        self.assertEqual(config.graph_params.depth, 2) # Default from AppConfig._GraphParams
        self.assertEqual(config.paths.log_file_path, Path("data/logs/analyzer.log")) # Default from AppConfig._Paths

    @patch('src.missing_links_analyzer.config.tomllib.load', side_effect=tomllib.TOMLDecodeError("Mocked TOML parse error"))
    @patch('builtins.open', new_callable=mock_open, read_data="invalid toml content") # Mock open with some data
    def test_load_config_parse_error_returns_default(self, mock_file_open, mock_tomllib_load):
        """ Test that load_config returns a default AppConfig if there's a TOML parsing error. """
        config = load_config(Path("dummy_config_for_parse_error.toml"))
        
        self.assertIsInstance(config, AppConfig)
        # Check a default value
        self.assertEqual(config.graph_params.depth, 2) # Default from AppConfig._GraphParams
        self.assertEqual(config.logging_params.log_level, "INFO") # Default from AppConfig._LoggingParams

    def test_path_attributes_are_path_objects(self):
        """ Test that path attributes in a loaded config are pathlib.Path objects. """
        # Using a mocked successful load with dummy content
        # tomllib.loads is used here because we are providing content directly, not a file object.
        # The patch for tomllib.load is to ensure it uses this parsed data.
        parsed_dummy_toml = tomllib.loads(DUMMY_TOML_CONTENT_VALID)
        with patch('builtins.open', mock_open(read_data=DUMMY_TOML_CONTENT_VALID)): # Mocks open for load_config
            with patch('src.missing_links_analyzer.config.tomllib.load', return_value=parsed_dummy_toml): # Mocks tomllib.load call
                config = load_config(Path("dummy_valid_config.toml"))

        self.assertIsInstance(config.paths.graph_output_dir, Path)
        self.assertEqual(config.paths.graph_output_dir, Path("data/test_graphs/"))
        self.assertIsInstance(config.paths.dataset_output_dir, Path)
        self.assertEqual(config.paths.dataset_output_dir, Path("data/test_datasets/"))
        self.assertIsInstance(config.paths.model_output_dir, Path)
        self.assertEqual(config.paths.model_output_dir, Path("data/test_models/"))
        self.assertIsInstance(config.paths.log_file_path, Path)
        self.assertEqual(config.paths.log_file_path, Path("data/logs/test_analyzer.log"))
        
        # Check a non-path attribute for good measure
        self.assertEqual(config.graph_params.start_page, "TestPage")
        self.assertEqual(config.logging_params.log_level, "DEBUG")

    def test_app_config_with_empty_data(self):
        """ Test AppConfig initialization with completely empty data. """
        config = AppConfig({}) # Pass empty dict to simulate no data from TOML
        self.assertEqual(config.title, "Missing Links Analyzer") # Default title
        self.assertEqual(config.paths.graph_output_dir, Path("data/graphs/")) # Default path
        self.assertEqual(config.graph_params.depth, 2) # Default graph param
        self.assertEqual(config.similarity_params.weak_threshold_quantile, 0.75) # Default sim param
        self.assertEqual(config.clustering_params.noise_threshold_ratio, 0.05) # Default clustering param
        self.assertEqual(config.model_params.random_state, 42) # Default model param
        self.assertEqual(config.logging_params.log_level, "INFO") # Default logging param

    def test_app_config_with_partial_data(self):
        """ Test AppConfig initialization with partially missing data. """
        partial_data = {
            "title": "Partial Config",
            "graph_params": {
                "depth": 3 # Overridden
            },
            "logging": {
                "log_level": "WARNING" # Overridden
            }
            # Other sections like 'paths', 'similarity_params', etc., are missing
        }
        config = AppConfig(partial_data)
        self.assertEqual(config.title, "Partial Config")
        # Paths should take defaults
        self.assertEqual(config.paths.graph_output_dir, Path("data/graphs/"))
        # Graph params
        self.assertEqual(config.graph_params.depth, 3) # Overridden
        self.assertEqual(config.graph_params.start_page, "Wikipedia") # Default
        # Logging params
        self.assertEqual(config.logging_params.log_level, "WARNING") # Overridden
        # Model params should take defaults
        self.assertEqual(config.model_params.test_split_ratio, 0.3)


if __name__ == '__main__':
    unittest.main()
