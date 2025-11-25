"""
Application Configuration
Centralized configuration for catalog and schema names.

Update these values to change the default catalog and schema used across all notebooks.
"""

# Unity Catalog Configuration
DEFAULT_CATALOG_NAME = "demos"
DEFAULT_SCHEMA_NAME = "career_path_temp"

# Alternative: Use these if you want to use a different catalog/schema
# DEFAULT_CATALOG_NAME = "zivile"
# DEFAULT_SCHEMA_NAME = "sap_demo"

# MLflow Experiment Configuration
MLFLOW_EXPERIMENT_PATH = "/Shared/career_path_intelligence_engine/career_intelligence_models_updated"

