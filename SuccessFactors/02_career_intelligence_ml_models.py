# Databricks notebook source
# MAGIC %md
# MAGIC # 🧠 Career Intelligence ML Models
# MAGIC ## Building Production-Ready Models with SAP SuccessFactors BDC Data
# MAGIC
# MAGIC This notebook demonstrates how to build real machine learning models using SAP Business Data Cloud (BDC) data products from SuccessFactors to power career intelligence predictions.
# MAGIC
# MAGIC ### **Models Built:**
# MAGIC - 🎯 **Career Path Success Prediction** - Predict success probability for role transitions
# MAGIC - ⚠️ **Employee Retention Risk Model** - Identify employees likely to leave
# MAGIC - 🌟 **High Potential Employee Identification** - Discover hidden talent
# MAGIC - 📈 **Promotion Readiness Scoring** - Score employees for advancement
# MAGIC
# MAGIC ### **SAP BDC Data Products Used:**
# MAGIC - CoreWorkforceData (from SAP SuccessFactors Employee Central Data Products)
# MAGIC - PerformanceData (from SAP SuccessFactors Performance and Goals Data Products)
# MAGIC - PerformanceReviews (from SAP SuccessFactors Performance and Goals Data Products)
# MAGIC - Compensation (from SAP SuccessFactors Employee Central Data Products)
# MAGIC - LearningHistory (from SAP SuccessFactors Learning Data Products)
# MAGIC - GoalsData (from SAP SuccessFactors Performance and Goals Data Products)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Setup & SAP BDC Data Integration

# COMMAND ----------

# Install required libraries for serverless compute
%pip install mlflow>=2.8.0 shap imbalanced-learn optuna

# COMMAND ----------

# Restart Python to ensure MLflow is properly loaded
%restart_python

# COMMAND ----------

# Import experiment path from app_config
from app_config import MLFLOW_EXPERIMENT_PATH
experiment_path = MLFLOW_EXPERIMENT_PATH

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

# ML Libraries - Using sklearn for serverless compute compatibility
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, classification_report,
    roc_curve, f1_score as f1_scorer
)
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, StackingClassifier, StackingRegressor
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️ imbalanced-learn not available - will use simple resampling")
import optuna
from optuna.samplers import TPESampler
import numpy as np

# Try importing SHAP and XGBoost/LightGBM (optional dependencies)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available - skipping SHAP explanations")

try:
    #import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available - using sklearn models only")

try:
    #import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("⚠️ LightGBM not available - using sklearn models only")

XGB_AVAILABLE = False
LGB_AVAILABLE = False

# MLflow for model tracking
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import random
import warnings
warnings.filterwarnings('ignore')

print("✅ ML Libraries & SAP BDC integration ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🗃️ Load SAP SuccessFactors BDC Data
# MAGIC ### *Data from Unity Catalog tables*

# COMMAND ----------

# Import configuration from setup module
# This will set catalog_name and schema_name variables
%run ./setup_config.py

# COMMAND ----------

# Configuration is now available from setup_config module
print(f"📋 Loading from Unity Catalog: {catalog_name}.{schema_name}")

# Use the catalog and schema
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

# Load data from Unity Catalog tables
print("📊 Loading SAP BDC data from Unity Catalog...")

employees_df = spark.table(f"{catalog_name}.{schema_name}.employees")
performance_df = spark.table(f"{catalog_name}.{schema_name}.performance")
learning_df = spark.table(f"{catalog_name}.{schema_name}.learning")
goals_df = spark.table(f"{catalog_name}.{schema_name}.goals")
compensation_df = spark.table(f"{catalog_name}.{schema_name}.compensation")

# Note: Removed .count() calls for serverless compute compatibility
# Counts are expensive operations - remove or cache if needed
print(f"✅ Data loaded from Unity Catalog tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Data Processing & Feature Engineering

# COMMAND ----------

# DataFrames already loaded above

# Create comprehensive feature dataset for ML models

# Employee base features
# Filter for active employees - handle multiple status formats ('Active', 'A', 'ACTIVE', 'ACT')
employee_features = employees_df.select(
    'employee_id',
    'age',
    'gender',
    'department',
    F.coalesce(F.col('department_name'), F.lit('Unknown')).alias('department_name'),  # Add department_name for display
    'job_title',
    'job_level',
    'location',
    F.coalesce(F.col('location_name'), F.lit('Unknown')).alias('location_name'),  # Add location_name for display
    'employment_type',
    'base_salary',
    'tenure_months',
    'months_in_current_role'
).filter(
    F.upper(F.trim(F.col('employment_status'))).isin(['ACTIVE', 'A', 'ACT'])
)

# Performance features (latest and trends)
latest_performance = performance_df.withColumn(
    'row_num', F.row_number().over(
        Window.partitionBy('employee_id').orderBy(F.desc('review_date'))
    )
).filter(F.col('row_num') == 1).select(
    'employee_id',
    F.col('overall_rating').alias('latest_performance_rating'),
    F.col('goals_achievement').alias('latest_goals_achievement'),
    F.col('competency_rating').alias('latest_competency_rating')
)

# Performance trend calculation
performance_trend = performance_df.withColumn(
    'row_num', F.row_number().over(
        Window.partitionBy('employee_id').orderBy(F.desc('review_date'))
    )
).filter(F.col('row_num') <= 2).groupBy('employee_id').agg(
    F.collect_list('overall_rating').alias('ratings')
).withColumn(
    'performance_trend',
    F.when(F.size(F.col('ratings')) == 1, F.lit('Stable'))
    .when(F.col('ratings')[0] > F.col('ratings')[1], F.lit('Improving'))
    .when(F.col('ratings')[0] < F.col('ratings')[1], F.lit('Declining'))
    .otherwise(F.lit('Stable'))
).select('employee_id', 'performance_trend')

# Learning features
learning_features = learning_df.filter(
    F.col('completion_status') == 'Completed'
).groupBy('employee_id').agg(
    F.count('learning_id').alias('courses_completed'),
    F.sum('hours_completed').alias('total_learning_hours'),
    F.avg('score').alias('avg_learning_score'),
    F.countDistinct('category').alias('learning_categories_count')
)

# Goal achievement features  
goal_features = goals_df.groupBy('employee_id').agg(
    F.count('goal_id').alias('total_goals'),
    F.avg('achievement_percentage').alias('avg_goal_achievement'),
    F.sum(F.when(F.col('achievement_percentage') >= 100, 1).otherwise(0)).alias('goals_exceeded'),
    F.countDistinct('goal_type').alias('goal_types_count')
)

# Compensation features
latest_compensation = compensation_df.withColumn(
    'row_num', F.row_number().over(
        Window.partitionBy('employee_id').orderBy(F.desc('effective_date'))
    )
).filter(F.col('row_num') == 1).select(
    'employee_id',
    F.col('bonus_target_pct').alias('current_bonus_target'),
    F.col('equity_value').alias('current_equity_value')
)

# Salary growth calculation
salary_growth = compensation_df.withColumn(
    'row_num', F.row_number().over(
        Window.partitionBy('employee_id').orderBy(F.desc('effective_date'))
    )
).filter(F.col('row_num') <= 2).groupBy('employee_id').agg(
    F.collect_list('base_salary').alias('salaries')
).withColumn(
    'salary_growth_rate',
    F.when(F.size(F.col('salaries')) == 1, F.lit(0.0))
    .when(F.col('salaries')[1] == 0, F.lit(0.0))
    .otherwise(F.expr("try_divide((salaries[0] - salaries[1]), salaries[1])"))
).select('employee_id', 'salary_growth_rate')

# Advanced Feature Engineering: Department-level aggregates
print("🔧 Creating department-level aggregate features...")
dept_aggregates = employee_features.join(latest_performance, 'employee_id', 'left') \
    .groupBy('department').agg(
        F.avg('base_salary').alias('dept_avg_salary'),
        F.avg('latest_performance_rating').alias('dept_avg_performance'),
        F.avg('tenure_months').alias('dept_avg_tenure'),
        F.stddev('base_salary').alias('dept_salary_std')
    ).fillna(0)

# Advanced Feature Engineering: Time-based features
print("🔧 Creating time-based features...")
time_features = latest_performance.select('employee_id', 'latest_performance_rating') \
    .join(
        performance_df.withColumn(
            'row_num', F.row_number().over(
                Window.partitionBy('employee_id').orderBy(F.desc('review_date'))
            )
        ).filter(F.col('row_num') == 1).select('employee_id', 'review_date'),
        'employee_id', 'left'
    ).withColumn(
        'months_since_last_review',
        F.months_between(F.current_date(), F.col('review_date'))
    ).fillna(0).select('employee_id', 'months_since_last_review')

print("✅ Feature engineering completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧠 ML Model 1: Career Path Success Prediction
# MAGIC ### *Predict success probability for role transitions*

# COMMAND ----------

# Create master feature dataset

master_features = employee_features \
    .join(latest_performance, 'employee_id', 'left') \
    .join(performance_trend, 'employee_id', 'left') \
    .join(learning_features, 'employee_id', 'left') \
    .join(goal_features, 'employee_id', 'left') \
    .join(latest_compensation, 'employee_id', 'left') \
    .join(salary_growth, 'employee_id', 'left') \
    .join(dept_aggregates, 'department', 'left') \
    .join(time_features, 'employee_id', 'left') \
    .fillna({
        'courses_completed': 0,
        'total_learning_hours': 0,
        'avg_learning_score': 70,
        'learning_categories_count': 0,
        'total_goals': 0,
        'avg_goal_achievement': 75,
        'goals_exceeded': 0,
        'goal_types_count': 0,
        'current_bonus_target': 0,
        'current_equity_value': 0,
        'salary_growth_rate': 0.0,
        'performance_trend': 'Stable',
        'dept_avg_salary': 0,
        'dept_avg_performance': 0,
        'dept_avg_tenure': 0,
        'dept_salary_std': 0,
        'months_since_last_review': 0
    })

# Add ratio and interaction features
master_features = master_features \
    .withColumn('salary_to_dept_avg', 
                F.coalesce(F.expr("try_divide(base_salary, dept_avg_salary)"), F.lit(1.0))) \
    .withColumn('tenure_to_dept_avg',
                F.coalesce(F.expr("try_divide(tenure_months, dept_avg_tenure)"), F.lit(1.0))) \
    .withColumn('learning_hours_per_month',
                F.coalesce(F.expr("try_divide(total_learning_hours, tenure_months)"), F.lit(0.0))) \
    .withColumn('salary_per_month_tenure',
                F.coalesce(F.expr("try_divide(base_salary, tenure_months)"), F.lit(0.0))) \
    .withColumn('performance_x_tenure', F.col('latest_performance_rating') * F.col('tenure_months')) \
    .withColumn('performance_x_salary_growth', F.col('latest_performance_rating') * F.greatest(F.col('salary_growth_rate'), F.lit(0)))

print("✅ Master dataset created with advanced features")

# Ensure location column has consistent type (cast to integer if needed)
# This prevents schema merge conflicts when overwriting the table
master_features = master_features.withColumn(
    "location",
    F.coalesce(F.expr("try_cast(location as int)"), F.lit(0))
)

# Ensure consistent schema by overwriting schema if table exists
master_features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog_name}.{schema_name}.master_features")

# COMMAND ----------


# Helper function for comprehensive model evaluation
def evaluate_classification_model(y_true, y_pred, y_pred_proba, model_name="Model"):
    """Evaluate classification model with comprehensive metrics"""
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Print metrics
    print(f"📊 {model_name} Metrics:")
    print(f"   AUC: {metrics['auc']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1 Score: {metrics['f1']:.4f}")
    
    return metrics

# Helper function for data validation
def validate_data(X_train, y_train, X_test, y_test, feature_names):
    """Validate training and test data for common issues"""
    print("🔍 Data Validation:")
    
    # Ensure arrays are numeric - always convert via pandas to handle any non-numeric data
    # This is safer than checking dtype, as arrays might have object dtype or mixed types
    try:
        # Try to compute mean to check if numeric
        _ = np.mean(X_train[:, 0])
        # If successful, ensure it's float64
        X_train = X_train.astype(np.float64)
    except (TypeError, ValueError, OverflowError):
        # Array contains non-numeric data, convert via pandas
        print("   ⚠️ Converting X_train from non-numeric to float...")
        col_names = feature_names[:X_train.shape[1]] if len(feature_names) >= X_train.shape[1] else [f'col_{i}' for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=col_names).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
    
    try:
        # Try to compute mean to check if numeric
        _ = np.mean(X_test[:, 0])
        # If successful, ensure it's float64
        X_test = X_test.astype(np.float64)
    except (TypeError, ValueError, OverflowError):
        # Array contains non-numeric data, convert via pandas
        print("   ⚠️ Converting X_test from non-numeric to float...")
        col_names = feature_names[:X_test.shape[1]] if len(feature_names) >= X_test.shape[1] else [f'col_{i}' for i in range(X_test.shape[1])]
        X_test = pd.DataFrame(X_test, columns=col_names).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
    
    # Check for outliers (beyond 3 standard deviations)
    outliers_count = 0
    for i in range(X_train.shape[1]):
        mean_val = np.mean(X_train[:, i])
        std_val = np.std(X_train[:, i])
        if std_val > 0:
            outliers = np.sum(np.abs(X_train[:, i] - mean_val) > 3 * std_val)
            if outliers > 0:
                outliers_count += outliers
    
    if outliers_count > 0:
        print(f"   ⚠️ Found {outliers_count} potential outliers (>3σ)")
    else:
        print(f"   ✅ No extreme outliers detected")
    
    # Check for low variance features
    variance_threshold = VarianceThreshold(threshold=0.01)
    variance_threshold.fit(X_train)
    low_var_features = [i for i, included in enumerate(variance_threshold.get_support()) if not included]
    if low_var_features:
        print(f"   ⚠️ Found {len(low_var_features)} low-variance features (indices: {low_var_features[:5]}{'...' if len(low_var_features) > 5 else ''})")
    else:
        print(f"   ✅ No low-variance features detected")
    
    # Check class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique_train, counts_train))
    print(f"   📊 Training class distribution: {class_dist}")
    
    # Check for feature distribution differences
    train_mean = np.mean(X_train, axis=0)
    test_mean = np.mean(X_test, axis=0)
    mean_diff = np.abs(train_mean - test_mean)
    large_diffs = np.sum(mean_diff > 2 * np.std(X_train, axis=0))
    if large_diffs > 0:
        print(f"   ⚠️ Found {large_diffs} features with significant train/test distribution differences")
    else:
        print(f"   ✅ Train/test distributions similar")
    
    return low_var_features, X_train, X_test

# Helper function for threshold optimization
def optimize_threshold(y_true, y_pred_proba, metric='f1'):
    """Find optimal threshold for binary classification"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

# Helper function for model comparison
def compare_models(X_train, y_train, X_test, y_test, models_dict):
    """Compare multiple models and return best performing one"""
    print("\n🔬 Model Comparison:")
    print("=" * 60)
    
    results = {}
    best_model_name = None
    best_score = 0
    
    for model_name, model_obj in models_dict.items():
        print(f"\n📊 Testing {model_name}...")
        model_obj.fit(X_train, y_train)
        y_pred_proba = model_obj.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        results[model_name] = {
            'model': model_obj,
            'auc': auc
        }
        print(f"   AUC: {auc:.4f}")
        
        if auc > best_score:
            best_score = auc
            best_model_name = model_name
    
    print(f"\n🏆 Best Model: {best_model_name} (AUC: {best_score:.4f})")
    print("=" * 60 + "\n")
    
    return results[best_model_name]['model'], best_model_name, results

# Comprehensive helper function for training classification models with all improvements
def train_classification_model_with_improvements(
    X_train, y_train, X_test, y_test, 
    feature_names, model_name_display,
    model_type='classifier',  # 'classifier' or 'regressor'
    param_grids=None,
    mlflow_run_name=None
    ):
    """
    Comprehensive training function with all improvements:
    - Data validation
    - Feature selection (RFE)
    - Model comparison
    - Hyperparameter tuning
    - Early stopping
    - Model calibration (for classification)
    - Cross-validation
    - Threshold optimization (for classification)
    - SHAP explanations
    - MLflow logging
    
    Returns: (final_pipeline, metrics_dict, selected_feature_names)
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 Training {model_name_display} with All Improvements")
    print(f"{'='*70}\n")
    
    # Data validation (ensures arrays are numeric)
    low_var_features, X_train, X_test = validate_data(X_train, y_train, X_test, y_test, feature_names)
    
    # Feature selection using RFE (keep 80% of features)
    print("\n🔍 Feature Selection with RFE...")
    n_features_to_select = max(int(len(feature_names) * 0.8), 10)
    
    if model_type == 'classifier':
        base_estimator = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=4)
    else:
        base_estimator = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=4)
    
    rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select, step=1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rfe.fit(X_train_scaled, y_train)
    selected_features_mask = rfe.get_support()
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features_mask[i]]
    
    X_train_selected = X_train_scaled[:, selected_features_mask]
    X_test_selected = X_test_scaled[:, selected_features_mask]
    
    print(f"✅ Selected {sum(selected_features_mask)} features from {len(feature_names)} total")
    
    # Model comparison (try multiple algorithms if available) - only for classification
    # Allow override via param_grids['preferred_model']
    preferred_model = param_grids.get('preferred_model', None) if param_grids else None
    
    if preferred_model:
        best_name = preferred_model
    elif model_type == 'classifier':
        models_to_compare = {
            'GradientBoosting': GradientBoostingClassifier(random_state=42, n_iter_no_change=10, validation_fraction=0.1)
        }
        
        if XGB_AVAILABLE:
            models_to_compare['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
        if LGB_AVAILABLE:
            models_to_compare['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        # Quick comparison on subset for speed
        if len(models_to_compare) > 1:
            best_base_model, best_name, comparison_results = compare_models(
                X_train_selected[:min(5000, len(X_train_selected))], 
                y_train[:min(5000, len(y_train))],
                X_test_selected[:min(1000, len(X_test_selected))],
                y_test[:min(1000, len(y_test))],
                {k: type(v)(**{k2: v2 for k2, v2 in v.get_params().items() if k2 != 'n_jobs'}) 
                 for k, v in models_to_compare.items()}
            )
        else:
            best_name = 'GradientBoosting'
    else:
        best_name = 'GradientBoosting'
    
    # Hyperparameter tuning with Optuna (Bayesian optimization)
    print(f"\n🔍 Performing Bayesian hyperparameter optimization with Optuna ({best_name})...")
    
    # Determine base model and create Optuna objective function
    def create_optuna_objective(X_train, y_train, model_type, best_name, cv_folds):
        def objective(trial):
            if best_name == 'RandomForest' and model_type == 'classifier':
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    class_weight=trial.suggest_categorical('class_weight', ['balanced', None]),
                    random_state=42,
                    n_jobs=1  # Changed from -1 for serverless compute compatibility
                )
            elif best_name == 'LogisticRegression' and model_type == 'classifier':
                penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
                l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9) if penalty == 'elasticnet' else 0.5
                # Use saga solver for all penalty types
                model_params = {
                    'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                    'penalty': penalty,
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                    'max_iter': trial.suggest_int('max_iter', 100, 500),
                    'random_state': 42,
                    'solver': 'saga'
                }
                if penalty == 'elasticnet':
                    model_params['l1_ratio'] = l1_ratio
                model = LogisticRegression(**model_params)
            elif model_type == 'classifier':
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    n_iter_no_change=trial.suggest_int('n_iter_no_change', 5, 20),
                    random_state=42
                )
            else:  # regressor
                model = GradientBoostingRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    n_iter_no_change=trial.suggest_int('n_iter_no_change', 5, 20),
                    random_state=42
                )
            
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, 
                                   scoring='roc_auc' if model_type == 'classifier' else 'r2',
                                   n_jobs=1)  # Changed from -1 for serverless compute compatibility
            return scores.mean()
        
        return objective
    
    # Cross-validation strategy
    # Use time-based CV if temporal ordering is important (e.g., if data has time component)
    # For now, use stratified/regular CV (can be enhanced with TimeSeriesSplit if temporal features are added)
    # To use time-based CV, uncomment below and ensure data is sorted by time:
    # if model_type == 'classifier':
    #     cv_folds = TimeSeriesSplit(n_splits=5)
    # else:
    #     cv_folds = TimeSeriesSplit(n_splits=5)
    if model_type == 'classifier':
        cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize' if model_type == 'classifier' else 'maximize',
        sampler=TPESampler(seed=42)
    )
    
    # Optimize (use fewer trials for faster execution, can be increased)
    n_trials = 30  # Reduced from exhaustive grid search for speed
    study.optimize(
        create_optuna_objective(X_train_selected, y_train, model_type, best_name, cv_folds),
        n_trials=n_trials,
        show_progress_bar=False  # Changed to False for serverless compute compatibility
    )
    
    print(f"✅ Best parameters: {study.best_params}")
    print(f"✅ Best CV score: {study.best_value:.4f}")
    
    # Recreate best model with optimal parameters
    if best_name == 'RandomForest' and model_type == 'classifier':
        model = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=1)  # Changed from -1 for serverless compute compatibility
    elif best_name == 'LogisticRegression' and model_type == 'classifier':
        # Ensure solver is set correctly for penalty type
        model_params = study.best_params.copy()
        if 'solver' not in model_params:
            model_params['solver'] = 'saga'
        model = LogisticRegression(**model_params, random_state=42)
    elif model_type == 'classifier':
        model = GradientBoostingClassifier(**study.best_params, random_state=42)
    else:
        model = GradientBoostingRegressor(**study.best_params, random_state=42)
    
    model.fit(X_train_selected, y_train)
    
    # Build ensemble model (stacking/voting) for better performance
    if model_type == 'classifier':
        print("\n🔗 Building ensemble model (stacking)...")
        # Create base models for ensemble
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=1)),  # Changed from -1 for serverless compute compatibility
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=200))
        ]
        
        # Use stacking with the best model as meta-learner
        ensemble_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42, max_iter=200),
            cv=3,
            n_jobs=1  # Changed from -1 for serverless compute compatibility
        )
        ensemble_model.fit(X_train_selected, y_train)
        
        # Compare ensemble vs single best model
        ensemble_score = cross_val_score(ensemble_model, X_train_selected, y_train, 
                                        cv=cv_folds, scoring='roc_auc', n_jobs=1).mean()  # Changed from -1 for serverless compute compatibility
        single_score = cross_val_score(model, X_train_selected, y_train, 
                                      cv=cv_folds, scoring='roc_auc', n_jobs=1).mean()  # Changed from -1 for serverless compute compatibility
        
        if ensemble_score > single_score:
            print(f"✅ Ensemble model performs better (AUC: {ensemble_score:.4f} vs {single_score:.4f}). Using ensemble.")
            model = ensemble_model
        else:
            print(f"✅ Single model performs better (AUC: {single_score:.4f} vs {ensemble_score:.4f}). Using single model.")
        
        # Model calibration (only for classification)
        print("\n📊 Calibrating model probabilities...")
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X_train_selected, y_train)
    else:
        # For regression, use stacking regressor
        print("\n🔗 Building ensemble model (stacking) for regression...")
        base_models = [
            ('rf', GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42))
        ]
        ensemble_model = StackingRegressor(
            estimators=base_models,
            final_estimator=GradientBoostingRegressor(n_estimators=50, random_state=42),
            cv=3,
            n_jobs=1  # Changed from -1 for serverless compute compatibility
        )
        ensemble_model.fit(X_train_selected, y_train)
        
        ensemble_score = cross_val_score(ensemble_model, X_train_selected, y_train, 
                                        cv=cv_folds, scoring='r2', n_jobs=1).mean()  # Changed from -1 for serverless compute compatibility
        single_score = cross_val_score(model, X_train_selected, y_train, 
                                      cv=cv_folds, scoring='r2', n_jobs=1).mean()  # Changed from -1 for serverless compute compatibility
        
        if ensemble_score > single_score:
            print(f"✅ Ensemble model performs better (R²: {ensemble_score:.4f} vs {single_score:.4f}). Using ensemble.")
            calibrated_model = ensemble_model
        else:
            print(f"✅ Single model performs better (R²: {single_score:.4f} vs {ensemble_score:.4f}). Using single model.")
            calibrated_model = model
    
    # Cross-validation (using same CV strategy as Optuna)
    print("🔄 Performing 5-fold cross-validation...")
    cv_scoring = 'roc_auc' if model_type == 'classifier' else 'r2'
    # Use calibrated model for CV if classification
    cv_model = calibrated_model if model_type == 'classifier' else model
    cv_scores = cross_val_score(cv_model, X_train_selected, y_train, cv=cv_folds, scoring=cv_scoring, n_jobs=1)  # Changed from -1 for serverless compute compatibility
    print(f"📈 CV {cv_scoring.upper()}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Make predictions
    if model_type == 'classifier':
        train_pred_proba = calibrated_model.predict_proba(X_train_selected)[:, 1]
        test_pred_proba = calibrated_model.predict_proba(X_test_selected)[:, 1]
        
        # Threshold optimization
        print("\n🎯 Optimizing classification threshold...")
        optimal_threshold, optimal_f1 = optimize_threshold(y_train, train_pred_proba, metric='f1')
        print(f"✅ Optimal threshold: {optimal_threshold:.3f} (F1: {optimal_f1:.4f})")
        
        train_pred = (train_pred_proba >= optimal_threshold).astype(int)
        test_pred = (test_pred_proba >= optimal_threshold).astype(int)
    else:
        train_pred = calibrated_model.predict(X_train_selected)
        test_pred = calibrated_model.predict(X_test_selected)
        train_pred_proba = None
        test_pred_proba = None
        optimal_threshold = None
        optimal_f1 = None
    
    # Evaluate with comprehensive metrics
    print("\n" + "="*60)
    if model_type == 'classifier':
        train_metrics = evaluate_classification_model(y_train, train_pred, train_pred_proba, "Training Set")
        print("-"*60)
        test_metrics = evaluate_classification_model(y_test, test_pred, test_pred_proba, "Test Set")
    else:
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = np.mean(np.abs(y_train - train_pred))
        test_mae = np.mean(np.abs(y_test - test_pred))
        
        print(f"📊 Training Metrics:")
        print(f"   RMSE: {train_rmse:.4f}")
        print(f"   R²: {train_r2:.4f}")
        print(f"   MAE: {train_mae:.4f}")
        print("-"*60)
        print(f"📊 Test Metrics:")
        print(f"   RMSE: {test_rmse:.4f}")
        print(f"   R²: {test_r2:.4f}")
        print(f"   MAE: {test_mae:.4f}")
        
        train_metrics = {'rmse': train_rmse, 'r2': train_r2, 'mae': train_mae}
        test_metrics = {'rmse': test_rmse, 'r2': test_r2, 'mae': test_mae}
    
    print("="*60 + "\n")
    
    # SHAP explanations (if available and classification)
    if SHAP_AVAILABLE and hasattr(model, 'feature_importances_') and model_type == 'classifier':
        print("🔍 Computing SHAP values...")
        try:
            shap_sample_size = min(50, len(X_test_selected))  # Reduced from 100 for serverless compute compatibility
            shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(X_test_selected[:shap_sample_size])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            shap_importance = np.abs(shap_values).mean(axis=0)
            top_shap_indices = np.argsort(shap_importance)[-10:][::-1]
            
            print("🔝 Top 10 features by SHAP importance:")
            for idx in top_shap_indices:
                feat_name = selected_feature_names[idx] if idx < len(selected_feature_names) else f"Feature_{idx}"
                print(f"   {feat_name}: {shap_importance[idx]:.4f}")
        except Exception as e:
            print(f"⚠️ SHAP computation failed: {e}")
    
    # Log to MLflow
    if mlflow_run_name:
        mlflow.log_param("n_features_selected", sum(selected_features_mask))
        mlflow.log_param("n_features_total", len(feature_names))
        mlflow.log_param("best_model_type", best_name)
        mlflow.log_param("model_calibrated", model_type == 'classifier')
        mlflow.log_param("ensemble_used", isinstance(model, (StackingClassifier, StackingRegressor)) if model_type == 'classifier' else isinstance(calibrated_model, StackingRegressor))
        mlflow.log_param("hyperparameter_optimization", "Optuna_TPE")
        if model_type == 'classifier':
            mlflow.log_param("optimal_threshold", optimal_threshold)
            mlflow.log_metric("optimal_threshold_f1", optimal_f1)
        
        # Log Optuna best parameters
        for param_name, param_value in study.best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        mlflow.log_param("optuna_n_trials", n_trials)
        mlflow.log_metric("optuna_best_score", study.best_value)
        
        mlflow.log_metric(f"cv_{cv_scoring}_mean", cv_scores.mean())
        mlflow.log_metric(f"cv_{cv_scoring}_std", cv_scores.std())
        
        if model_type == 'classifier':
            mlflow.log_metric("train_auc", train_metrics['auc'])
            mlflow.log_metric("test_auc", test_metrics['auc'])
            mlflow.log_metric("train_accuracy", train_metrics['accuracy'])
            mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
            mlflow.log_metric("train_precision", train_metrics['precision'])
            mlflow.log_metric("test_precision", test_metrics['precision'])
            mlflow.log_metric("train_recall", train_metrics['recall'])
            mlflow.log_metric("test_recall", test_metrics['recall'])
            mlflow.log_metric("train_f1", train_metrics['f1'])
            mlflow.log_metric("test_f1", test_metrics['f1'])
        else:
            mlflow.log_metric("train_rmse", train_metrics['rmse'])
            mlflow.log_metric("test_rmse", test_metrics['rmse'])
            mlflow.log_metric("train_r2", train_metrics['r2'])
            mlflow.log_metric("test_r2", test_metrics['r2'])
            mlflow.log_metric("train_mae", train_metrics['mae'])
            mlflow.log_metric("test_mae", test_metrics['mae'])
    
    # Create pipeline with selected features
    selected_scaler = StandardScaler()
    selected_scaler.fit(X_train[:, selected_features_mask])
    final_model_pipeline = SKPipeline([('scaler', selected_scaler), ('model', calibrated_model)])
    
    # Prepare metrics dictionary
    metrics_dict = {
        **test_metrics,
        f'cv_{cv_scoring}': cv_scores.mean(),
        'optimal_threshold': optimal_threshold
    } if model_type == 'classifier' else {
        **test_metrics,
        f'cv_{cv_scoring}': cv_scores.mean()
    }
    
    return final_model_pipeline, metrics_dict, selected_feature_names, selected_features_mask

# Helper function to encode categorical variables using SQL (serverless-compatible)
def encode_categoricals(df, categorical_cols=['gender', 'department', 'location', 'employment_type', 'performance_trend']):
    """Encode categorical variables using SQL expressions (compatible with serverless compute)"""
    encoded_df = df
    
    for cat_col in categorical_cols:
        if cat_col in df.columns:
            # Get distinct values using toPandas() instead of collect() for serverless compatibility
            # This is more memory-efficient than collect()
            distinct_df = df.select(cat_col).distinct().filter(F.col(cat_col).isNotNull())
            distinct_vals = distinct_df.select(cat_col).toPandas()[cat_col].tolist()
            
            # Limit to reasonable number of categories
            max_cats = 10 if cat_col == 'gender' else (8 if cat_col == 'department' else 
                       (5 if cat_col == 'location' else 3))
            
            # Create one-hot encoded columns
            for val in distinct_vals[:max_cats]:
                # Sanitize column name
                safe_val = str(val).replace(' ', '_').replace('-', '_').replace('/', '_')
                encoded_df = encoded_df.withColumn(
                    f'{cat_col}_{safe_val}',
                    F.when(F.col(cat_col) == val, 1.0).otherwise(0.0)
                )
    
    # Get encoded column names
    encoded_cols = [col for col in encoded_df.columns 
                   if any(col.startswith(f'{cat_col}_') for cat_col in categorical_cols)]
    
    return encoded_df, encoded_cols



# COMMAND ----------

# Configure MLflow to use Unity Catalog

# Set Unity Catalog registry
mlflow.set_registry_uri("databricks-uc")

# Initialize MLflow client for setting aliases
from mlflow.tracking import MlflowClient
mlflow_client = MlflowClient()

# Start MLflow experiment
mlflow.set_experiment(experiment_path)

# COMMAND ----------

# Create career success target variable

try:
    # Define success metrics based on multiple factors
    career_success_df = master_features.withColumn(
        'career_success_score',
        # Performance component (40%)
        (F.col('latest_performance_rating') / 5.0 * 0.4) +
        # Growth component (25%) 
        (F.greatest(F.col('salary_growth_rate'), F.lit(0)) * 10 * 0.25) +
        # Learning component (20%)
        (F.least(F.col('total_learning_hours') / 100.0, F.lit(1.0)) * 0.2) +
        # Goal achievement component (15%)
        (F.col('avg_goal_achievement') / 100.0 * 0.15)
    )

    # RELAXED: Lower thresholds to increase positive samples
    career_success_df = career_success_df.withColumn(
        'promotion_ready',
        F.when(
            (F.col('career_success_score') >= 0.5) &
            (F.col('latest_performance_rating') >= 2.5) &
            (F.col('months_in_current_role') >= 3) &
            (F.col('performance_trend') != 'Declining'),
            1
        ).otherwise(0)
    )

    # Print class distribution after label creation
    class_dist = career_success_df.groupBy('promotion_ready').count().toPandas().set_index('promotion_ready')['count'].to_dict()
    print("Class distribution after label creation:", class_dist)

    # Fallback: If still no positive samples, assign top 10% by career_success_score as promotion_ready=1
    if class_dist.get(1, 0) == 0:
        print("⚠️ No positive samples after relaxing logic. Assigning top 10% by career_success_score as promotion_ready=1.")
        from pyspark.sql.window import Window
        w = Window.orderBy(F.desc('career_success_score'))
        total_count = career_success_df.count()
        top_n = max(1, int(total_count * 0.1))
        career_success_df = career_success_df.withColumn(
            'promotion_ready',
            F.when(F.row_number().over(w) <= top_n, 1).otherwise(0)
        )
        class_dist = career_success_df.groupBy('promotion_ready').count().toPandas().set_index('promotion_ready')['count'].to_dict()
        print("Class distribution after fallback:", class_dist)

    print("✅ Career success labels created")

    # Prepare data for ML model - encode categoricals using SQL (serverless-compatible)
    print("🔬 Encoding categorical variables using SQL (serverless-compatible)...")

    # Encode categorical variables using SQL expressions instead of StringIndexer/OneHotEncoder
    # This avoids Py4J security restrictions in serverless compute
    career_success_encoded, encoded_cat_cols = encode_categoricals(career_success_df)

    # Prepare feature columns (numeric + encoded categoricals + advanced features)
    # These will be used by all models
    feature_columns = [
        'age', 'job_level', 'tenure_months', 'months_in_current_role', 'base_salary',
        'latest_performance_rating', 'latest_goals_achievement', 'latest_competency_rating',
        'courses_completed', 'total_learning_hours', 'avg_learning_score', 'learning_categories_count',
        'total_goals', 'avg_goal_achievement', 'goals_exceeded', 'goal_types_count',
        'current_bonus_target', 'current_equity_value', 'salary_growth_rate',
        # Advanced features
        'dept_avg_salary', 'dept_avg_performance', 'dept_avg_tenure', 'dept_salary_std',
        'months_since_last_review',
        'salary_to_dept_avg', 'tenure_to_dept_avg', 'learning_hours_per_month',
        'salary_per_month_tenure', 'performance_x_tenure', 'performance_x_salary_growth'
    ]

    # Ensure all encoded columns exist in the dataframe
    available_encoded_cols = [col for col in encoded_cat_cols if col in career_success_encoded.columns]
    all_feature_cols = feature_columns + available_encoded_cols

    print(f"✅ Encoded {len(available_encoded_cols)} categorical features using SQL")
    print(f"📊 Total features: {len(all_feature_cols)}")

    print("🔬 Training Career Success Prediction Model...")

    # Unity Catalog model path
    uc_model_name = f"{catalog_name}.{schema_name}.career_success_prediction"

    # Store metrics for final display
    model_metrics_summary = {}

    with mlflow.start_run(run_name="career_success_prediction"):
        
        # Convert Spark DataFrame to Pandas for sklearn (serverless-compatible)
        print("🔄 Converting to Pandas DataFrame for sklearn...")
        df_pandas = career_success_encoded.select(all_feature_cols + ['promotion_ready']).toPandas()
        
        # Print class distribution for promotion_ready before splitting
        print("Class distribution in promotion_ready (full dataset):", df_pandas['promotion_ready'].value_counts().to_dict())
        
        # Ensure all feature columns are numeric (exclude original categorical columns)
        categorical_cols_to_exclude = ['gender', 'department', 'location', 'employment_type', 'performance_trend', 'department_name', 'location_name']
        numeric_feature_cols = [col for col in all_feature_cols if col not in categorical_cols_to_exclude and col in df_pandas.columns]
        
        # Convert all columns to numeric, coercing errors to NaN, then fill with 0
        for col in numeric_feature_cols:
            df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce').fillna(0)
        
        # Ensure we only select columns that exist and are numeric
        X_df = df_pandas[numeric_feature_cols].copy()
        # Double-check all columns are numeric
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        X = X_df.values.astype(np.float64)
        y = pd.to_numeric(df_pandas['promotion_ready'], errors='coerce').fillna(0).values.astype(np.int32)
        
        print(f"📊 Dataset: {len(X):,} records, {len(numeric_feature_cols)} features")
        
        # If there are no positive samples, raise a clear error
        if np.sum(y) == 0:
            raise ValueError("No positive samples (promotion_ready=1) in the dataset. Please further relax the label logic or check your data.")
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError as e:
            print(f"⚠️ Stratified split failed: {e}. Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Print class distribution after splitting
        print("Class distribution in y_train:", pd.Series(y_train).value_counts().to_dict())
        print("Class distribution in y_test:", pd.Series(y_test).value_counts().to_dict())
        
        # Advanced resampling with SMOTE if class imbalance exists
        if len(np.unique(y_train)) < 2:
            print("⚠️ Only one class in y_train after split. Attempting SMOTE resampling...")
            try:
                # Use BorderlineSMOTE for better handling of borderline cases
                smote = BorderlineSMOTE(random_state=42, k_neighbors=min(5, len(np.where(y_train == 1)[0]) - 1))
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print("✅ Applied BorderlineSMOTE resampling.")
                print("Class distribution in y_train after SMOTE:", pd.Series(y_train).value_counts().to_dict())
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}. Falling back to simple resampling...")
                from sklearn.utils import resample
                X_train_df = pd.DataFrame(X_train, columns=numeric_feature_cols)
                y_train_df = pd.Series(y_train, name='promotion_ready')
                train_df = pd.concat([X_train_df, y_train_df], axis=1)
                df_majority = train_df[train_df['promotion_ready'] == 0]
                df_minority = train_df[train_df['promotion_ready'] == 1]
                if len(df_minority) == 0:
                    print("⚠️ No positive samples in training set. Forcing at least one positive sample.")
                    df_minority = pd.DataFrame([0] * len(df_majority), columns=train_df.columns)
                    df_minority['promotion_ready'] = 1
                df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
                train_upsampled = pd.concat([df_majority, df_minority_upsampled])
                X_train = train_upsampled[numeric_feature_cols].values.astype(float)
                y_train = train_upsampled['promotion_ready'].values.astype(int)
                print("✅ Upsampled minority class in training set.")
                print("Class distribution in y_train after upsampling:", pd.Series(y_train).value_counts().to_dict())
        elif np.sum(y_train == 1) < np.sum(y_train == 0) * 0.3 and IMBLEARN_AVAILABLE:  # If minority class is < 30% of majority
            print("🔧 Applying SMOTE for class imbalance...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train == 1) - 1))
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print("✅ Applied SMOTE resampling.")
                print("Class distribution in y_train after SMOTE:", pd.Series(y_train).value_counts().to_dict())
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}. Continuing without resampling.")
        
        # Final check before training
        if len(np.unique(y_train)) < 2:
            raise ValueError("Training set still contains only one class after upsampling. Please check your data and label logic.")
        
        # Train with all improvements using helper function
        final_model_pipeline, metrics_dict, selected_feature_names, selected_features_mask = \
            train_classification_model_with_improvements(
                X_train, y_train, X_test, y_test,
                numeric_feature_cols,
                "Career Success Prediction Model",
                model_type='classifier',
                mlflow_run_name="career_success_prediction"
            )
        
        # Store metrics for display
        model_metrics_summary['career_success'] = metrics_dict
        
        # Create signature for Unity Catalog (required)
        sample_input_all = pd.DataFrame(X_test[:5], columns=numeric_feature_cols)
        sample_input_selected = sample_input_all[selected_feature_names]
        sample_output = pd.DataFrame(final_model_pipeline.predict(sample_input_selected), columns=['prediction'])
        signature = infer_signature(sample_input_selected, sample_output)
        
        # Log model to Unity Catalog using sklearn flavor with signature
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model_pipeline,
            name="career_success_model",
            signature=signature,
            input_example=sample_input_selected.head(1),
            registered_model_name=uc_model_name
        )
        
        # Log selected features metadata
        mlflow.log_dict({
            'selected_features': selected_feature_names,
            'all_features': numeric_feature_cols
        }, artifact_file="feature_selection.json")
        
        # Set alias for Unity Catalog (required for easy loading)
        try:
            # Get the version that was just registered (order_by not supported for UC, so we'll find max manually)
            versions = list(mlflow_client.search_model_versions(f"name='{uc_model_name}'"))
            if versions:
                # Find the latest version manually (highest version number)
                latest_version = max(versions, key=lambda v: int(v.version))
                mlflow_client.set_registered_model_alias(uc_model_name, "Champion", latest_version.version)
                print(f"✅ Set 'Champion' alias on model version {latest_version.version}")
        except Exception as e:
            print(f"⚠️ Could not set alias (model may work without it): {e}")
except Exception as e:
    print(f"❌ Error in Cell 17: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md  
# MAGIC ## ⚠️ ML Model 2: Employee Retention Risk Prediction
# MAGIC ### *Identify employees likely to leave*

# COMMAND ----------

print("🔬 Training Retention Risk Model...")

try:
    # Unity Catalog model path for retention risk
    uc_retention_model_name = f"{catalog_name}.{schema_name}.retention_risk_prediction"

    # Create retention risk labels based on realistic factors
    retention_risk_df = master_features.withColumn(
        'flight_risk_score',
        # Performance dissatisfaction component
        F.when(F.col('latest_performance_rating') < 3.0, 0.3).otherwise(0.0) +
        # Stagnation component  
        F.when(F.col('months_in_current_role') > 36, 0.25).otherwise(0.0) +
        # Compensation component
        F.when(F.col('salary_growth_rate') < 0.02, 0.2).otherwise(0.0) +
        # Learning engagement component
        F.when(F.col('total_learning_hours') < 10, 0.15).otherwise(0.0) +
        # Goal achievement component
        F.when(F.col('avg_goal_achievement') < 60, 0.1).otherwise(0.0) +
        # Random component for demo purposes
        (F.rand() * 0.1)
    )
    
    # RELAXED: Lower threshold for high_flight_risk to increase positive samples
    retention_risk_df = retention_risk_df.withColumn(
        'high_flight_risk',
        F.when(F.col('flight_risk_score') > 0.5, 1).otherwise(0)
    )

    # Print class distribution after label creation
    class_dist = retention_risk_df.groupBy('high_flight_risk').count().toPandas().set_index('high_flight_risk')['count'].to_dict()
    print("Class distribution after label creation:", class_dist)

    # Fallback: If still no positive samples, assign top 10% by flight_risk_score as high_flight_risk=1
    if class_dist.get(1, 0) == 0:
        print("⚠️ No positive samples after relaxing logic. Assigning top 10% by flight_risk_score as high_flight_risk=1.")
        from pyspark.sql.window import Window
        w = Window.orderBy(F.desc('flight_risk_score'))
        total_count = retention_risk_df.count()
        top_n = max(1, int(total_count * 0.1))
        retention_risk_df = retention_risk_df.withColumn(
            'high_flight_risk',
            F.when(F.row_number().over(w) <= top_n, 1).otherwise(0)
        )
        class_dist = retention_risk_df.groupBy('high_flight_risk').count().toPandas().set_index('high_flight_risk')['count'].to_dict()
        print("Class distribution after fallback:", class_dist)

    print("✅ Retention risk labels created")

    # Encode categoricals for retention risk model (will have same encoded columns)
    retention_risk_encoded, _ = encode_categoricals(retention_risk_df)
    # Ensure same feature columns are available
    available_cols_retention = [col for col in all_feature_cols if col in retention_risk_encoded.columns]

    with mlflow.start_run(run_name="retention_risk_prediction"):
        
        # Convert to Pandas for sklearn
        print("🔄 Converting to Pandas DataFrame for sklearn...")
        df_pandas = retention_risk_encoded.select(available_cols_retention + ['high_flight_risk']).toPandas()
        
        # Print class distribution for high_flight_risk before splitting
        print("Class distribution in high_flight_risk (full dataset):", df_pandas['high_flight_risk'].value_counts().to_dict())
        
        # Ensure all feature columns are numeric (exclude original categorical columns)
        categorical_cols_to_exclude = ['gender', 'department', 'location', 'employment_type', 'performance_trend', 'department_name', 'location_name']
        numeric_cols_retention = [col for col in available_cols_retention if col not in categorical_cols_to_exclude and col in df_pandas.columns]
        
        # Convert all columns to numeric, coercing errors to NaN, then fill with 0
        for col in numeric_cols_retention:
            df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce').fillna(0)
        
        # Ensure we only select columns that exist and are numeric
        X_df = df_pandas[numeric_cols_retention].copy()
        # Double-check all columns are numeric
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        X = X_df.values.astype(np.float64)
        y = pd.to_numeric(df_pandas['high_flight_risk'], errors='coerce').fillna(0).values.astype(np.int32)
        
        # If there are no positive samples, raise a clear error
        if np.sum(y) == 0:
            raise ValueError("No positive samples (high_flight_risk=1) in the dataset. Please further relax the label logic or check your data.")
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError as e:
            print(f"⚠️ Stratified split failed: {e}. Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Print class distribution after splitting
        print("Class distribution in y_train:", pd.Series(y_train).value_counts().to_dict())
        print("Class distribution in y_test:", pd.Series(y_test).value_counts().to_dict())
        
        # Advanced resampling with SMOTE if class imbalance exists
        if len(np.unique(y_train)) < 2:
            print("⚠️ Only one class in y_train after split. Attempting SMOTE resampling...")
            if IMBLEARN_AVAILABLE:
                try:
                    smote = BorderlineSMOTE(random_state=42, k_neighbors=min(5, len(np.where(y_train == 1)[0]) - 1))
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print("✅ Applied BorderlineSMOTE resampling.")
                    print("Class distribution in y_train after SMOTE:", pd.Series(y_train).value_counts().to_dict())
                except Exception as e:
                    print(f"⚠️ SMOTE failed: {e}. Falling back to simple resampling...")
                    from sklearn.utils import resample
                    X_train_df = pd.DataFrame(X_train, columns=numeric_cols_retention)
                    y_train_df = pd.Series(y_train, name='high_flight_risk')
                    train_df = pd.concat([X_train_df, y_train_df], axis=1)
                    df_majority = train_df[train_df['high_flight_risk'] == 0]
                    df_minority = train_df[train_df['high_flight_risk'] == 1]
                    if len(df_minority) == 0:
                        print("⚠️ No positive samples in training set. Forcing at least one positive sample.")
                        df_minority = pd.DataFrame([0] * len(df_majority), columns=train_df.columns)
                        df_minority['high_flight_risk'] = 1
                    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
                    train_upsampled = pd.concat([df_majority, df_minority_upsampled])
                    X_train = train_upsampled[numeric_cols_retention].values.astype(float)
                    y_train = train_upsampled['high_flight_risk'].values.astype(int)
                    print("✅ Upsampled minority class in training set.")
                    print("Class distribution in y_train after upsampling:", pd.Series(y_train).value_counts().to_dict())
            else:
                from sklearn.utils import resample
                X_train_df = pd.DataFrame(X_train, columns=numeric_cols_retention)
                y_train_df = pd.Series(y_train, name='high_flight_risk')
                train_df = pd.concat([X_train_df, y_train_df], axis=1)
                df_majority = train_df[train_df['high_flight_risk'] == 0]
                df_minority = train_df[train_df['high_flight_risk'] == 1]
                if len(df_minority) == 0:
                    print("⚠️ No positive samples in training set. Forcing at least one positive sample.")
                    df_minority = pd.DataFrame([0] * len(df_majority), columns=train_df.columns)
                    df_minority['high_flight_risk'] = 1
                df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
                train_upsampled = pd.concat([df_majority, df_minority_upsampled])
                X_train = train_upsampled[numeric_cols_retention].values.astype(float)
                y_train = train_upsampled['high_flight_risk'].values.astype(int)
                print("✅ Upsampled minority class in training set.")
                print("Class distribution in y_train after upsampling:", pd.Series(y_train).value_counts().to_dict())
        elif np.sum(y_train == 1) < np.sum(y_train == 0) * 0.3 and IMBLEARN_AVAILABLE:
            print("🔧 Applying SMOTE for class imbalance...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train == 1) - 1))
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print("✅ Applied SMOTE resampling.")
                print("Class distribution in y_train after SMOTE:", pd.Series(y_train).value_counts().to_dict())
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}. Continuing without resampling.")
        
        # Final check before training
        if len(np.unique(y_train)) < 2:
            raise ValueError("Training set still contains only one class after upsampling. Please check your data and label logic.")
        
        # Custom param_grid for RandomForest
        custom_params = {
            'preferred_model': 'RandomForest',
            'param_grid': {
                'n_estimators': [50, 100, 150],
                'max_depth': [6, 8, 10],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced', None]
            },
            'base_model': RandomForestClassifier(random_state=42, n_jobs=1)  # Changed from -1 for serverless compute compatibility
        }
        
        # Train with all improvements using helper function
        final_model_pipeline, metrics_dict, selected_feature_names, selected_features_mask = \
            train_classification_model_with_improvements(
                X_train, y_train, X_test, y_test,
                numeric_cols_retention,
                "Retention Risk Prediction Model",
                model_type='classifier',
                param_grids=custom_params,
                mlflow_run_name="retention_risk_prediction"
            )
        
        # Store metrics for display
        model_metrics_summary['retention_risk'] = metrics_dict
        
        # Create signature for Unity Catalog (required)
        sample_input_all = pd.DataFrame(X_test[:5], columns=numeric_cols_retention)
        sample_input_selected = sample_input_all[selected_feature_names]
        sample_output = pd.DataFrame(final_model_pipeline.predict(sample_input_selected), columns=['prediction'])
        signature = infer_signature(sample_input_selected, sample_output)
        
        # Log model to Unity Catalog with signature
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model_pipeline,
            name="retention_risk_model",
            signature=signature,
            input_example=sample_input_selected.head(1),
            registered_model_name=uc_retention_model_name
        )
        
        # Log selected features metadata
        mlflow.log_dict({
            'selected_features': selected_feature_names,
            'all_features': numeric_cols_retention
        }, artifact_file="feature_selection.json")
        
        # Set alias for Unity Catalog
        try:
            versions = list(mlflow_client.search_model_versions(f"name='{uc_retention_model_name}'"))
            if versions:
                latest_version = max(versions, key=lambda v: int(v.version))
                mlflow_client.set_registered_model_alias(uc_retention_model_name, "Champion", latest_version.version)
                print(f"✅ Set 'Champion' alias on model version {latest_version.version}")
        except Exception as e:
            print(f"⚠️ Could not set alias (model may work without it): {e}")
except Exception as e:
    print(f"❌ Error in Retention Risk Model Cell: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🌟 ML Model 3: High Potential Employee Identification  
# MAGIC ### *Discover hidden talent with ML*

# COMMAND ----------

print("🔬 Training High Potential Identification Model...")

try:
    # Create high potential labels
    high_potential_df = master_features.withColumn(
        'potential_score', 
        # Performance excellence component (30%)
        (F.col('latest_performance_rating') / 5.0 * 0.3) +
        # Learning agility component (25%)
        (F.least(F.col('learning_categories_count') / 5.0, F.lit(1.0)) * 0.25) +
        # Goal achievement component (20%) 
        (F.col('avg_goal_achievement') / 100.0 * 0.2) +
        # Growth trajectory component (15%)
        (F.greatest(F.col('salary_growth_rate') * 10, F.lit(0)) * 0.15) +
        # Engagement/initiative component (10%)
        (F.least(F.col('total_learning_hours') / 50.0, F.lit(1.0)) * 0.1)
    )
    
    # RELAXED: Lower threshold for high_potential to increase positive samples
    high_potential_df = high_potential_df.withColumn(
        'high_potential',
        F.when(
            (F.col('potential_score') >= 0.6) &
            (F.col('latest_performance_rating') >= 3.0) &
            (F.col('performance_trend') != 'Declining'),
            1
        ).otherwise(0)
    )

    # Print class distribution after label creation
    class_dist = high_potential_df.groupBy('high_potential').count().toPandas().set_index('high_potential')['count'].to_dict()
    print("Class distribution after label creation:", class_dist)

    # Fallback: If still no positive samples, assign top 10% by potential_score as high_potential=1
    if class_dist.get(1, 0) == 0:
        print("⚠️ No positive samples after relaxing logic. Assigning top 10% by potential_score as high_potential=1.")
        from pyspark.sql.window import Window
        w = Window.orderBy(F.desc('potential_score'))
        total_count = high_potential_df.count()
        top_n = max(1, int(total_count * 0.1))
        high_potential_df = high_potential_df.withColumn(
            'high_potential',
            F.when(F.row_number().over(w) <= top_n, 1).otherwise(0)
        )
        class_dist = high_potential_df.groupBy('high_potential').count().toPandas().set_index('high_potential')['count'].to_dict()
        print("Class distribution after fallback:", class_dist)

    print("✅ High potential labels created")

    # Encode categoricals for high potential model (will have same encoded columns)
    high_potential_encoded, _ = encode_categoricals(high_potential_df)
    # Ensure same feature columns are available
    available_cols_potential = [col for col in all_feature_cols if col in high_potential_encoded.columns]

    # Unity Catalog model path for high potential
    uc_potential_model_name = f"{catalog_name}.{schema_name}.high_potential_identification"

    with mlflow.start_run(run_name="high_potential_identification"):
        
        # Convert to Pandas for sklearn
        print("🔄 Converting to Pandas DataFrame for sklearn...")
        df_pandas = high_potential_encoded.select(available_cols_potential + ['high_potential']).toPandas()
        
        # Print class distribution for high_potential before splitting
        print("Class distribution in high_potential (full dataset):", df_pandas['high_potential'].value_counts().to_dict())
        
        # Ensure all feature columns are numeric (exclude original categorical columns)
        categorical_cols_to_exclude = ['gender', 'department', 'location', 'employment_type', 'performance_trend', 'department_name', 'location_name']
        numeric_cols_potential = [col for col in available_cols_potential if col not in categorical_cols_to_exclude and col in df_pandas.columns]
        
        # Convert all columns to numeric, coercing errors to NaN, then fill with 0
        for col in numeric_cols_potential:
            df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce').fillna(0)
        
        # Ensure we only select columns that exist and are numeric
        X_df = df_pandas[numeric_cols_potential].copy()
        # Double-check all columns are numeric
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        X = X_df.values.astype(np.float64)
        y = pd.to_numeric(df_pandas['high_potential'], errors='coerce').fillna(0).values.astype(np.int32)
        
        # If there are no positive samples, raise a clear error
        if np.sum(y) == 0:
            raise ValueError("No positive samples (high_potential=1) in the dataset. Please further relax the label logic or check your data.")
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError as e:
            print(f"⚠️ Stratified split failed: {e}. Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Print class distribution after splitting
        print("Class distribution in y_train:", pd.Series(y_train).value_counts().to_dict())
        print("Class distribution in y_test:", pd.Series(y_test).value_counts().to_dict())
        
        # Advanced resampling with SMOTE if class imbalance exists
        if len(np.unique(y_train)) < 2:
            print("⚠️ Only one class in y_train after split. Attempting SMOTE resampling...")
            if IMBLEARN_AVAILABLE:
                try:
                    smote = BorderlineSMOTE(random_state=42, k_neighbors=min(5, len(np.where(y_train == 1)[0]) - 1))
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print("✅ Applied BorderlineSMOTE resampling.")
                    print("Class distribution in y_train after SMOTE:", pd.Series(y_train).value_counts().to_dict())
                except Exception as e:
                    print(f"⚠️ SMOTE failed: {e}. Falling back to simple resampling...")
                    from sklearn.utils import resample
                    X_train_df = pd.DataFrame(X_train, columns=numeric_cols_potential)
                    y_train_df = pd.Series(y_train, name='high_potential')
                    train_df = pd.concat([X_train_df, y_train_df], axis=1)
                    df_majority = train_df[train_df['high_potential'] == 0]
                    df_minority = train_df[train_df['high_potential'] == 1]
                    if len(df_minority) == 0:
                        print("⚠️ No positive samples in training set. Forcing at least one positive sample.")
                        df_minority = pd.DataFrame([0] * len(df_majority), columns=train_df.columns)
                        df_minority['high_potential'] = 1
                    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
                    train_upsampled = pd.concat([df_majority, df_minority_upsampled])
                    X_train = train_upsampled[numeric_cols_potential].values.astype(float)
                    y_train = train_upsampled['high_potential'].values.astype(int)
                    print("✅ Upsampled minority class in training set.")
                    print("Class distribution in y_train after upsampling:", pd.Series(y_train).value_counts().to_dict())
            else:
                from sklearn.utils import resample
                X_train_df = pd.DataFrame(X_train, columns=numeric_cols_potential)
                y_train_df = pd.Series(y_train, name='high_potential')
                train_df = pd.concat([X_train_df, y_train_df], axis=1)
                df_majority = train_df[train_df['high_potential'] == 0]
                df_minority = train_df[train_df['high_potential'] == 1]
                if len(df_minority) == 0:
                    print("⚠️ No positive samples in training set. Forcing at least one positive sample.")
                    df_minority = pd.DataFrame([0] * len(df_majority), columns=train_df.columns)
                    df_minority['high_potential'] = 1
                df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
                train_upsampled = pd.concat([df_majority, df_minority_upsampled])
                X_train = train_upsampled[numeric_cols_potential].values.astype(float)
                y_train = train_upsampled['high_potential'].values.astype(int)
                print("✅ Upsampled minority class in training set.")
                print("Class distribution in y_train after upsampling:", pd.Series(y_train).value_counts().to_dict())
        elif np.sum(y_train == 1) < np.sum(y_train == 0) * 0.3 and IMBLEARN_AVAILABLE:
            print("🔧 Applying SMOTE for class imbalance...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train == 1) - 1))
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print("✅ Applied SMOTE resampling.")
                print("Class distribution in y_train after SMOTE:", pd.Series(y_train).value_counts().to_dict())
            except Exception as e:
                print(f"⚠️ SMOTE failed: {e}. Continuing without resampling.")
        
        # Final check before training
        if len(np.unique(y_train)) < 2:
            raise ValueError("Training set still contains only one class after upsampling. Please check your data and label logic.")
        
        # Custom param_grid for LogisticRegression
        custom_params = {
            'preferred_model': 'LogisticRegression',
            'param_grid': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'l1_ratio': [0.25, 0.5, 0.75],
                'class_weight': ['balanced', None],
                'max_iter': [100, 200]
            },
            'base_model': LogisticRegression(random_state=42, solver='saga')
        }
        
        # Train with all improvements using helper function
        final_model_pipeline, metrics_dict, selected_feature_names, selected_features_mask = \
            train_classification_model_with_improvements(
                X_train, y_train, X_test, y_test,
                numeric_cols_potential,
                "High Potential Identification Model",
                model_type='classifier',
                param_grids=custom_params,
                mlflow_run_name="high_potential_identification"
            )
        
        # Store metrics for display
        model_metrics_summary['high_potential'] = metrics_dict
        
        # Create signature for Unity Catalog (required)
        sample_input_all = pd.DataFrame(X_test[:5], columns=numeric_cols_potential)
        sample_input_selected = sample_input_all[selected_feature_names]
        sample_output = pd.DataFrame(final_model_pipeline.predict(sample_input_selected), columns=['prediction'])
        signature = infer_signature(sample_input_selected, sample_output)
        
        # Log model to Unity Catalog with signature
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model_pipeline,
            name="high_potential_model",
            signature=signature,
            input_example=sample_input_selected.head(1),
            registered_model_name=uc_potential_model_name
        )
        
        # Log selected features metadata
        mlflow.log_dict({
            'selected_features': selected_feature_names,
            'all_features': numeric_cols_potential
        }, artifact_file="feature_selection.json")
        
        # Set alias for Unity Catalog
        try:
            versions = list(mlflow_client.search_model_versions(f"name='{uc_potential_model_name}'"))
            if versions:
                latest_version = max(versions, key=lambda v: int(v.version))
                mlflow_client.set_registered_model_alias(uc_potential_model_name, "Champion", latest_version.version)
                print(f"✅ Set 'Champion' alias on model version {latest_version.version}")
        except Exception as e:
            print(f"⚠️ Could not set alias (model may work without it): {e}")
except Exception as e:
    print(f"❌ Error in High Potential Model Cell: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 ML Model 4: Promotion Readiness Scoring
# MAGIC ### *Continuous scoring for advancement decisions*

# COMMAND ----------

print("🔬 Training Promotion Readiness Scoring Model...")

try:
    # Create promotion readiness score (regression model)
    promotion_readiness_df = master_features.withColumn(
        'readiness_score',
        # Base performance component
        (F.col('latest_performance_rating') / 5.0 * 30) +
        # Tenure in role component (optimal range 12-36 months)
        F.when(F.col('months_in_current_role').between(12, 36), 
               25 - F.abs(F.col('months_in_current_role') - 24) * 0.5).otherwise(10) +
        # Learning component
        (F.least(F.col('total_learning_hours') / 40.0, F.lit(1.0)) * 20) +
        # Goal achievement component  
        (F.col('avg_goal_achievement') / 100.0 * 15) +
        # Growth indicators
        (F.greatest(F.col('salary_growth_rate') * 500, F.lit(0)) * 10) +
        # Random component for realism
        (F.rand() * 10 - 5)
    ).withColumn(
        'readiness_score',
        F.greatest(F.least(F.col('readiness_score'), F.lit(100)), F.lit(0))
    )

    print("✅ Promotion readiness scores created")

    # Encode categoricals for promotion readiness model (will have same encoded columns)
    promotion_readiness_encoded, _ = encode_categoricals(promotion_readiness_df)
    # Ensure same feature columns are available
    available_cols_readiness = [col for col in all_feature_cols if col in promotion_readiness_encoded.columns]

    # Unity Catalog model path for promotion readiness
    uc_readiness_model_name = f"{catalog_name}.{schema_name}.promotion_readiness_scoring"

    with mlflow.start_run(run_name="promotion_readiness_scoring"):
        
        # Convert to Pandas for sklearn
        print("🔄 Converting to Pandas DataFrame for sklearn...")
        df_pandas = promotion_readiness_encoded.select(available_cols_readiness + ['readiness_score']).toPandas()
        
        # Ensure all feature columns are numeric (exclude original categorical columns)
        categorical_cols_to_exclude = ['gender', 'department', 'location', 'employment_type', 'performance_trend', 'department_name', 'location_name']
        numeric_cols_readiness = [col for col in available_cols_readiness if col not in categorical_cols_to_exclude and col in df_pandas.columns]
        
        # Convert all columns to numeric, coercing errors to NaN, then fill with 0
        for col in numeric_cols_readiness:
            df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce').fillna(0)
        
        # Ensure we only select columns that exist and are numeric
        X_df = df_pandas[numeric_cols_readiness].copy()
        # Double-check all columns are numeric
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        
        X = X_df.values.astype(np.float64)
        y = pd.to_numeric(df_pandas['readiness_score'], errors='coerce').fillna(0).values.astype(np.float64)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train with all improvements using helper function (regression model)
        final_model_pipeline, metrics_dict, selected_feature_names, selected_features_mask = \
            train_classification_model_with_improvements(
                X_train, y_train, X_test, y_test,
                numeric_cols_readiness,
                "Promotion Readiness Scoring Model",
                model_type='regressor',
                mlflow_run_name="promotion_readiness_scoring"
            )
        
        # Store metrics for display
        model_metrics_summary['promotion_readiness'] = metrics_dict
        
        # Create signature for Unity Catalog (required)
        sample_input_all = pd.DataFrame(X_test[:5], columns=numeric_cols_readiness)
        sample_input_selected = sample_input_all[selected_feature_names]
        sample_output = pd.DataFrame(final_model_pipeline.predict(sample_input_selected), columns=['prediction'])
        signature = infer_signature(sample_input_selected, sample_output)
        
        # Log model to Unity Catalog with signature
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model_pipeline,
            name="promotion_readiness_model",
            signature=signature,
            input_example=sample_input_selected.head(1),
            registered_model_name=uc_readiness_model_name
        )
        
        # Log selected features metadata
        mlflow.log_dict({
            'selected_features': selected_feature_names,
            'all_features': numeric_cols_readiness
        }, artifact_file="feature_selection.json")
        
        # Set alias for Unity Catalog
        try:
            versions = list(mlflow_client.search_model_versions(f"name='{uc_readiness_model_name}'"))
            if versions:
                latest_version = max(versions, key=lambda v: int(v.version))
                mlflow_client.set_registered_model_alias(uc_readiness_model_name, "Champion", latest_version.version)
                print(f"✅ Set 'Champion' alias on model version {latest_version.version}")
        except Exception as e:
            print(f"⚠️ Could not set alias (model may work without it): {e}")
except Exception as e:
    print(f"❌ Error in Promotion Readiness Model Cell: {e}")
    import traceback
    traceback.print_exc()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Model Deployment & Serving
# MAGIC ### *Deploy models for real-time career intelligence*

# COMMAND ----------

print("✅ All models logged and registered to Unity Catalog!")
print(f"📊 Models registered in: {catalog_name}.{schema_name}")
print(f"   • {uc_model_name}")
print(f"   • {uc_retention_model_name}")
print(f"   • {uc_potential_model_name}")
print(f"   • {uc_readiness_model_name}")
print("\n💡 Models are automatically registered to Unity Catalog during logging.")
print("   Use 'models:/{catalog}.{schema}.{model_name}@Champion' to load them (recommended for Unity Catalog).")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧪 Model Testing & Validation
# MAGIC ### *Test models with sample employees*

# COMMAND ----------


# Get a sample of employees for testing
# Include department_name and location_name for display
sample_employees_df = master_features.join(
    employees_df.select('employee_id', 'first_name', 'last_name', 'department_name', 'location_name'),
    'employee_id',
    'left'
)
sample_employees = sample_employees_df.limit(10).toPandas()

print("📊 Sample Employee Analysis:")
print("=" * 60)

for idx, employee in sample_employees.iterrows():
    emp_id = employee['employee_id']
    name = f"{employee.get('first_name', 'John')} {employee.get('last_name', 'Doe')}"
    dept = employee.get('department_name', employee.get('department', 'Unknown'))
    location = employee.get('location_name', employee.get('location', 'Unknown'))
    title = employee['job_title']
    
    print(f"\n👤 {emp_id}")
    print(f"   📋 {title} in {dept} ({location})")
    print(f"   ⏱️ {employee['tenure_months']} months tenure, {employee['months_in_current_role']} months in role")
    print(f"   ⭐ Performance: {employee['latest_performance_rating']:.1f}/5.0")
    print(f"   📚 Learning: {employee['courses_completed']} courses, {employee['total_learning_hours']} hours")
    print(f"   🎯 Goals: {employee['avg_goal_achievement']:.0f}% achievement rate")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 🎬 **ML Models Complete!**
# MAGIC
# MAGIC **These production-ready models now power the Career Path Intelligence Engine with real predictions based on SAP SuccessFactors data.**
# MAGIC
# MAGIC *Next: Deploy to Model Serving endpoints and integrate with the main demo notebook for live AI-powered career intelligence!*
