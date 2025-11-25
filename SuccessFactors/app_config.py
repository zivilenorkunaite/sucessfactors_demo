"""
Application Configuration
Centralized configuration for catalog and schema names.

Update these values to change the default catalog and schema used across all notebooks.
"""

# Unity Catalog Configuration
DEFAULT_CATALOG_NAME = "demos"
DEFAULT_SCHEMA_NAME = "career_path_temp"


# MLflow Experiment Configuration
MLFLOW_EXPERIMENT_PATH = "/Shared/career_path_intelligence_engine/career_intelligence_models_updated"

# Data Product Table Names (SAP SuccessFactors Data Products via Delta Sharing)
EMPLOYEES_DATA_PRODUCT_TABLE = "core_workforce_data_dp.coreworkforcedata.coreworkforce_standardfields"
PERFORMANCE_DATA_PRODUCT_TABLE = "performance_data_dp.performancedata.performancedata"
LEARNING_DATA_PRODUCT_TABLE = "learning_history_dp.learninghistory.learningcompletion"

