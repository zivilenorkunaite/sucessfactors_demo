# SAP SuccessFactors Career Path Intelligence Engine

A comprehensive Databricks-based solution for analyzing employee career paths using SAP SuccessFactors data products and machine learning.

## Overview

This project provides an end-to-end solution for:
- **Data Generation**: Synthetic SAP SuccessFactors data generation with fallback to SAP BDC data products
- **ML Model Training**: Career success prediction, retention risk, high potential identification, and promotion readiness models
- **Career Intelligence**: AI-powered insights and recommendations for employee career development

## Repository Structure

```
SuccessFactors/
├── 01_data_generation.py              # Data generation notebook with SAP BDC integration
├── 02_career_intelligence_ml_models.py # ML model training notebook
├── 03_career_path_intelligence_engine_full.py # Main career intelligence engine
├── career_intelligence_helpers.py      # Helper functions for ML models and predictions
├── setup_config.py                    # Unity Catalog configuration
└── app_config.py                      # Application configuration (catalog/schema defaults)
```

## Features

### Data Generation (`01_data_generation.py`)
- **SAP BDC Data Products Integration**: Loads data from SAP SuccessFactors via Delta Sharing
- **Fallback Mechanism**: Generates synthetic data when data products are unavailable
- **Schema Consistency**: Ensures identical schemas between data product and generated data
- **Unity Catalog Lineage**: Preserves data lineage for governance
- **Configurable Thresholds**: All business rules and thresholds are centralized as constants

### ML Models (`02_career_intelligence_ml_models.py`)
- Career Success Prediction
- Retention Risk Prediction
- High Potential Identification
- Promotion Readiness Scoring

### Career Intelligence Engine (`03_career_path_intelligence_engine_full.py`)
- AI-powered career path recommendations
- Natural language queries using Databricks AI (`ai_query()`)
- Performance analysis and insights
- Career development suggestions

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to SAP SuccessFactors BDC data products (optional, falls back to generated data)
- Python 3.8+
- Required Python packages (installed via `%pip install` in notebooks):
  - Faker
  - PySpark
  - MLflow
  - NumPy

## Configuration

### Unity Catalog Setup

Default catalog and schema are configured in `app_config.py`:
```python
DEFAULT_CATALOG_NAME = "demos"
DEFAULT_SCHEMA_NAME = "career_path_temp"
```

To customize:
1. Update `app_config.py` with your catalog/schema names
2. Or use Databricks widgets to override at runtime

### Data Product Configuration

The notebooks automatically attempt to load from SAP SuccessFactors data products:
- `core_workforce_data_dp.coreworkforcedata.coreworkforce_standardfields` (Employees)
- `performance_reviews_dp.performancereviews.commentfeedback` (Performance Reviews)
- `learning_history_dp.learninghistory.learningcompletion` (Learning Records)

If data products are unavailable, the system automatically falls back to generated synthetic data.

## Usage

### 1. Setup and Configuration
```python
# Run setup_config.py to configure Unity Catalog
%run ./setup_config.py
```

### 2. Generate Data
Run `01_data_generation.py` to:
- Load data from SAP BDC data products (if available)
- Generate synthetic data as fallback
- Save to Unity Catalog tables: `employees`, `performance`, `learning`, `goals`, `compensation`

### 3. Train ML Models
Run `02_career_intelligence_ml_models.py` to train and register ML models in Unity Catalog.

### 4. Run Career Intelligence Engine
Run `03_career_path_intelligence_engine_full.py` to:
- Load trained models
- Generate career insights
- Provide AI-powered recommendations

## Key Features

### Centralized Configuration
All thresholds and business rules are defined as constants at the top of `01_data_generation.py`:
- Performance rating thresholds
- Tenure requirements
- Salary increase percentages
- Status mappings
- And more...

### Data Lineage
The solution preserves Unity Catalog lineage by:
- Using Spark-native operations (no `.collect()`)
- Spark Connect compatible UDFs
- Direct DataFrame transformations

### Error Handling
- Safe type casting with `try_cast` for malformed data
- Explicit schema definitions for empty DataFrames
- Comprehensive error logging and fallback mechanisms

## License

This project is provided as-is for demonstration purposes.

## Author

Zivile Norkunaite

## Repository

https://github.com/zivilenorkunaite/sucessfactors_demo

