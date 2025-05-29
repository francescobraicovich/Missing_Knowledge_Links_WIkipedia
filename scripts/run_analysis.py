"""
Command-line interface for running the Missing Links Analysis pipeline.

This script allows users to run the full analysis pipeline, optionally providing
a custom configuration file. If no custom configuration is specified, the default
settings (from /settings.toml via src.missing_links_analyzer.config) are used.

Example usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --config path/to/your/custom_settings.toml
"""
import argparse
import logging
from pathlib import Path
import sys

# Add project root to sys.path to allow direct import of missing_links_analyzer
# This assumes the script is run from the project root (e.g., `python scripts/run_analysis.py`)
# or from the scripts directory itself.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(PROJECT_ROOT))
    print(f'Path added to sys.path: {PROJECT_ROOT}', file=sys.stderr)
    
    from src.missing_links_analyzer.analyzer import MissingLinksAnalyzer
    from src.missing_links_analyzer.config import load_config, settings as default_settings, AppConfig
except ImportError as e:
    # Fallback if __file__ is not defined (e.g. in some interactive environments)
    # or if the typical project structure is not as expected.
    print(f"Error during initial imports, possibly due to sys.path issues: {e}", file=sys.stderr)
    print("Ensure you are running this script from the project root or 'scripts/' directory.", file=sys.stderr)
    sys.exit(1)


# Get a logger for this script.
# The actual logging configuration (handlers, level, format) is expected to be set up
# by the config.py module when it's imported. This script's logger will inherit that.
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Missing Links Analysis pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a custom settings.toml file. If not provided, default settings are used."
    )
    args = parser.parse_args()

    app_config: AppConfig
    if args.config:
        if not args.config.is_file():
            # Use print for critical errors before logging might be fully configured from custom file
            print(f"Error: Custom configuration file not found at {args.config}", file=sys.stderr)
            # Log it as well, in case basic logging is already working
            logger.error(f"Custom configuration file not found at {args.config}")
            sys.exit(1)
        
        # Log this attempt before trying to load, as loading might reconfigure logging.
        logger.info(f"Attempting to load custom configuration from: {args.config}")
        try:
            app_config = load_config(args.config) # This returns a new AppConfig instance
            logger.info(f"Successfully loaded custom configuration: {args.config.name}")
            
            # Update the root logger's level based on the custom config.
            # Handlers and formatters set up by config.py's initial import will remain,
            # but this ensures their effective level respects the custom config.
            logging.getLogger().setLevel(app_config.logging_params.log_level.upper())
            logger.info(f"Root logger level updated to: {app_config.logging_params.log_level.upper()} from custom config.")

        except Exception as e:
            logger.error(f"Failed to load custom configuration from {args.config}: {e}", exc_info=True)
            print(f"Error loading custom configuration: {e}", file=sys.stderr)
            sys.exit(1)
            
    else:
        logger.info("Using default configuration.")
        app_config = default_settings
        # Ensure root logger level matches default settings if it wasn't set by custom config
        logging.getLogger().setLevel(app_config.logging_params.log_level.upper())


    # Fallback: if config.py's logging setup somehow didn't add any handlers
    # (e.g., due to a file path error during initial setup),
    # at least log critical messages from this script to stdout.
    if not logging.getLogger().handlers:
         logging.basicConfig(stream=sys.stdout, 
                             level=app_config.logging_params.log_level.upper(), # Use level from loaded config
                             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
         logger.info("Fallback basic logging configured for console output as no root handlers were found.")
    
    logger.info(f"Starting Missing Links Analysis pipeline for start page: '{app_config.graph_params.start_page}'")
    # Log the effective level to confirm what level messages will actually be processed at by the root logger.
    logger.info(f"Effective root logger level: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")
    logger.info(f"Log file path specified in config: {app_config.paths.log_file_path}")


    try:
        analyzer = MissingLinksAnalyzer(config=app_config)
        analyzer.run_full_pipeline()
        logger.info("Missing Links Analysis pipeline completed successfully.")
        # Provide a user-friendly message to stdout as well
        print("\nPipeline completed successfully. Check logs for details.")
        sys.exit(0) # Explicitly exit with success code
    except RuntimeError as e: # Catch errors from _ensure_data_loaded specifically
        logger.error(f"Pipeline execution halted due to a prerequisite data issue: {e}", exc_info=True)
        print(f"\nPipeline Error: {e}. Check logs for details.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
        print(f"\nAn unexpected critical error occurred: {e}. Check logs for details.", file=sys.stderr)
        sys.exit(1)