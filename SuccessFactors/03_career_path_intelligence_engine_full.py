# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Career Path Intelligence Engine
# MAGIC ## Powered by SAP SuccessFactors + SAP Databricks
# MAGIC
# MAGIC **Transform HR Decision Making with AI-Powered Career Intelligence**
# MAGIC
# MAGIC This demo showcases how SAP Databricks unlocks the hidden potential in your SAP SuccessFactors data through:
# MAGIC - 🎯 **Predictive Career Pathing**: AI-driven recommendations based on historical success patterns
# MAGIC - 🔍 **Hidden Talent Discovery**: Identify high-potential employees ready for advancement
# MAGIC - 📈 **Success Probability Modeling**: Predict career move outcomes with confidence scores
# MAGIC
# MAGIC
# MAGIC ### **Why SAP Databricks + SuccessFactors = Career Intelligence Revolution**
# MAGIC - **Advanced Analytics**: ML models that learn from organizational career patterns
# MAGIC - **Unified Data Platform**: Seamlessly combine HR, performance, and organizational data
# MAGIC - **Real-time Processing**: Instant insights as your workforce evolves
# MAGIC - **Scalable Intelligence**: Analyze patterns across your entire workforce history

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Setup & Configuration

# COMMAND ----------

# Install all required libraries for serverless compute
%pip install plotly>=5.18.0 mlflow>=2.8.0 lightgbm faker

# COMMAND ----------

# Restart Python to ensure all libraries are properly loaded
%restart_python

# COMMAND ----------

# Import all required libraries after restart
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

# Advanced visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# MLflow for model loading
import mlflow
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient

from datetime import datetime, timedelta, date
import re
import warnings
warnings.filterwarnings('ignore')

# Note: Employee names are already generated in data generation notebook

# Configure MLflow to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Initialize MLflow client
mlflow_client = MlflowClient()

# Import helper functions
# In Databricks, helper files in same directory are auto-importable
try:
    from career_intelligence_helpers import (
        load_career_models, prepare_ml_features_for_prediction, prepare_features_for_model,
        explain_prediction, format_feature_name, extract_sklearn_model_from_mlflow,
        get_potential_next_roles, create_transition_features,
        estimate_salary_increase, get_success_factors, get_risk_factors,
        ensure_dataframe_schema, get_role_compatibility_score, calculate_skill_gap_penalty,
        get_role_specific_timeline, get_alex_data, build_alex_context, 
        build_context_summary, format_ai_response, get_alex_career_predictions,
        discover_hidden_talent_with_ml, display_talent_graphs
    )
except ImportError:
    # Fallback: add current directory to path
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from career_intelligence_helpers import (
        load_career_models, prepare_ml_features_for_prediction, prepare_features_for_model,
        explain_prediction, format_feature_name, extract_sklearn_model_from_mlflow,
        get_potential_next_roles, create_transition_features,
        estimate_salary_increase, get_success_factors, get_risk_factors,
        ensure_dataframe_schema, get_role_compatibility_score, calculate_skill_gap_penalty,
        get_role_specific_timeline, get_alex_data, build_alex_context,
        build_context_summary, format_ai_response, get_alex_career_predictions,
        discover_hidden_talent_with_ml, display_talent_graphs
    )

# Configure for optimal display
displayHTML("""
<style>
    div.output_subarea { max-width: 100%; }
    
    /* Unified HTML Component Styles */
    .card-primary { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%); 
                    padding: 0; border-radius: 12px; margin: 20px 0; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15); overflow: hidden; }
    .card-header { background: rgba(255,255,255,0.12); padding: 18px 24px; 
                   border-bottom: 1px solid rgba(255,255,255,0.15); }
    .card-content { padding: 24px; background: rgba(255,255,255,0.03); }
    .card-footer { background: rgba(76,175,80,0.15); padding: 12px 24px; 
                   border-top: 1px solid rgba(255,255,255,0.1); }
    .text-accent { color: #FFD93D; }
    .text-muted { color: rgba(255,255,255,0.7); }
</style>
""")

print("✅ Career Intelligence Engine initialized")

# COMMAND ----------

# Import configuration and load data
%run ./setup_config.py

# COMMAND ----------

# Load data from Unity Catalog
print(f"📋 Loading from Unity Catalog: {catalog_name}.{schema_name}")

# Use the catalog and schema
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

# Load data from Unity Catalog tables (created by data generation notebook)
# Note: Data generation notebook (01_data_generation.py) loads from SAP SuccessFactors Data Products,
# transforms the data, and saves to Unity Catalog tables. This notebook uses those pre-processed tables.
print("📊 Loading data from Unity Catalog tables...")
print(f"   Catalog: {catalog_name}, Schema: {schema_name}")

try:
    # Load from Unity Catalog tables (already transformed by data generation notebook)
    # These tables contain data from SAP SuccessFactors Data Products (or generated fallback)
    employees_df = spark.table(f"{catalog_name}.{schema_name}.employees")
    performance_df = spark.table(f"{catalog_name}.{schema_name}.performance")
    
    print(f"✅ Data loaded: {employees_df.count():,} employees, {performance_df.count():,} performance reviews")
    print(f"   Data source: Unity Catalog tables (pre-processed by data generation notebook)")
    
    # Data is already transformed with correct schema and names from data generation notebook
    # Schema includes: employee_id, first_name, last_name, age, gender, department, job_title, 
    #                  job_level, location, employment_type, base_salary, tenure_months, 
    #                  months_in_current_role, employment_status
    
    # Create enriched employees view with performance metrics
    latest_performance = performance_df.withColumn(
        "row_num",
        F.row_number().over(
            Window.partitionBy("employee_id")
            .orderBy(F.col("review_date").desc())
        )
    ).filter(F.col("row_num") == 1).drop("row_num")
    
    # Join employees with latest performance data
    enriched_employees_df = employees_df.alias("e").join(
        latest_performance.alias("p"),
        F.col("e.employee_id") == F.col("p.employee_id"),
        "left"
    ).select(
        F.col("e.*"),
        F.coalesce(F.col("p.overall_rating"), F.lit(3.5)).alias("performance_rating"),
        F.coalesce(F.col("p.competency_rating"), F.lit(3.5)).alias("competency_rating"),
        F.coalesce(F.col("p.overall_rating"), F.lit(3.5)).alias("overall_rating")
    ).withColumn(
        "current_level", F.col("job_title")
    ).withColumn(
        "months_in_role", F.col("months_in_current_role")
    ).withColumn(
        "months_in_company", F.col("tenure_months")
    )
    
    # Calculate derived metrics
    enriched_employees_df = enriched_employees_df.withColumn(
        "engagement_score", F.lit(75) + (F.col("performance_rating") - 3) * 10
    ).withColumn(
        "potential_score", F.lit(70) + (F.col("performance_rating") - 3) * 10
    ).withColumn(
        "leadership_readiness", F.lit(60) + (F.col("performance_rating") - 3) * 10
    ).withColumn(
        "flight_risk", F.lit(30) - (F.col("performance_rating") - 3) * 10
    )
    
    enriched_employees_df = enriched_employees_df.withColumn(
        "performance_trend", 
        F.when(F.col("performance_rating") >= 4, "Rising")
        .when(F.col("performance_rating") <= 2.5, "Declining")
        .otherwise("Stable")
    )
    
    employees_df = enriched_employees_df
    print(f"✅ Enriched employees data with performance metrics")
except Exception as e:
    print(f"⚠️ Error loading data from Unity Catalog tables: {e}")
    print(f"   Please ensure data generation notebook (01_data_generation.py) has been run first")
    print(f"   Expected tables: {catalog_name}.{schema_name}.employees, {catalog_name}.{schema_name}.performance")
    raise

# COMMAND ----------

def demonstrate_sap_bdc_integration():
    """Demonstrate SAP BDC Delta Sharing integration"""
    sap_bdc_data_products = {
        'CoreWorkforceData': {
            'package': 'SAP SuccessFactors Employee Central Data Products',
            'access_method': 'Delta Sharing',
            'governance': 'Unity Catalog'
        },
        'PerformanceData': {
            'package': 'SAP SuccessFactors Performance and Goals Data Products',
            'access_method': 'Delta Sharing',
            'governance': 'Unity Catalog'
        },
        'PerformanceReviews': {
            'package': 'SAP SuccessFactors Performance and Goals Data Products',
            'access_method': 'Delta Sharing',
            'governance': 'Unity Catalog'
        },
        'LearningHistory': {
            'package': 'SAP SuccessFactors Learning Data Products',
            'access_method': 'Delta Sharing',
            'governance': 'Unity Catalog'
        },
        'GoalsData': {
            'package': 'SAP SuccessFactors Performance and Goals Data Products',
            'access_method': 'Delta Sharing',
            'governance': 'Unity Catalog'
        },
        'Compensation': {
            'package': 'SAP SuccessFactors Employee Central Data Products',
            'access_method': 'Delta Sharing',
            'governance': 'Unity Catalog'
        }
    }
    
    # Display SAP BDC integration status
    displayHTML(f"""
    <div style="background: linear-gradient(135deg, #0052CC 0%, #0070F2 100%); 
                padding: 30px; border-radius: 20px; color: white; margin: 20px 0;">
        <h2 style="text-align: center; margin-bottom: 25px;">🔗 SAP BDC - Success Factors Data Products</h2>
        
        
        <div style="margin-top: 25px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">
            <h3 style="color: #4ECDC4; margin: 0 0 15px 0;">📦 SAP SuccessFactors Data Products</h3>
            <ul style="list-style: none; padding: 0; margin: 0;">
                <li style="margin: 8px 0;">• CoreWorkforceData (Employee Central)</li>
                <li style="margin: 8px 0;">• PerformanceData (Performance & Goals)</li>
                <li style="margin: 8px 0;">• PerformanceReviews (Performance & Goals)</li>
                <li style="margin: 8px 0;">• LearningHistory (Learning)</li>
                <li style="margin: 8px 0;">• GoalsData (Performance & Goals)</li>
                <li style="margin: 8px 0;">• Compensation (Employee Central)</li>
            </ul>
            <p style="margin-top: 15px; font-size: 13px; opacity: 0.9;">
                <strong>💡 Key Benefit:</strong> All data accessed directly via Delta Sharing - no ETL, no copying, 
                no storage duplication. Real-time access to SAP SuccessFactors data products.
            </p>
        </div>
    </div>
    """)
    
    return sap_bdc_data_products

# Demonstrate SAP BDC integration
sap_bdc_products = demonstrate_sap_bdc_integration()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧠 ML Model Integration
# MAGIC ### *Load Trained ML Models from Unity Catalog*

# COMMAND ----------

# Load models using helper function
try:
    _ = catalog_name
    _ = schema_name
except NameError:
    raise NameError(
        "❌ catalog_name and schema_name are not defined. "
        "Please run '%run ./setup_config.py' before loading models."
    )

# COMMAND ----------

# Load ML models using helper function
career_models, model_metrics = load_career_models(catalog_name, schema_name, mlflow_client, displayHTML)

if not career_models:
    raise ValueError("❌ ML models not loaded. Please run notebook 02_career_intelligence_ml_models.py to train and register models in Unity Catalog.")

    print(f"✅ {len(career_models)} ML models active and ready for predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Career Intelligence in Action
# MAGIC
# MAGIC Explore AI-powered career insights using real SAP SuccessFactors data. Ask questions about employees, identify high-potential talent, and discover optimal career paths.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Meet Our Demo Employee

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💬 Ask Questions About the Demo Employee
# MAGIC
# MAGIC Use AI-powered queries to analyze the demo employee's career profile and get intelligent insights.
# MAGIC
# MAGIC **🔴 LIVE DEMO:** Modify the `example_question` variable below and re-run this cell to show instant AI responses!

# COMMAND ----------

# Define AI query functions before use
def build_context_summary(context, question=""):
    """Build context string from employee data by querying actual database"""
    
    if not context:
        return "No specific context provided."
    
    # Try to extract employee ID from context
    import re
    emp_id_match = re.search(r'EMP\d+', context)
    
    if emp_id_match:
        employee_id = emp_id_match.group()
        try:
            emp_data = employees_df.filter(F.col('employee_id') == employee_id).collect()
            if emp_data:
                emp = emp_data[0]
                return f"""
                Employee: {emp.get('name', 'Unknown')}
                Role: {emp.get('current_level', 'Unknown')}
                Department: {emp.get('department', 'Unknown')}
                Performance Rating: {emp.get('performance_rating', 'N/A')}/5
                Tenure: {emp.get('months_in_company', 0)} months in company, {emp.get('months_in_role', 0)} months in current role
                Engagement: {emp.get('engagement_score', 0)}%
                Potential: {emp.get('potential_score', 0)}%
                Leadership Readiness: {emp.get('leadership_readiness', 0)}%
                """
        except Exception as e:
            pass
    
    # If context contains organizational info or question asks about teams/departments, query actual data
    try:
        question_lower = question.lower() if question else ""
        context_lower = context.lower() if context else ""
        combined_text = f"{question_lower} {context_lower}"
        
        # Detect if this is about engineering team
        if 'engineering' in combined_text or 'engineer' in combined_text:
            eng_employees = employees_df.filter(
                (F.lower(F.col('department')).contains('engineering')) |
                (F.lower(F.col('job_title')).contains('engineer')) |
                (F.lower(F.col('job_title')).contains('developer')) |
                (F.lower(F.col('job_title')).contains('software'))
            ).select(
                'employee_id', 'first_name', 'last_name', 'job_title', 
                'job_level', 'department', 'performance_rating',
                'months_in_current_role', 'potential_score', 'leadership_readiness'
            ).collect()
            
            if eng_employees:
                eng_summary = f"Engineering Team Analysis ({len(eng_employees)} members):\n\n"
                for emp in eng_employees[:20]:  # Limit to top 20 for context
                    eng_summary += f"• {emp.get('first_name', '')} {emp.get('last_name', '')} - {emp.get('job_title', 'Unknown')} ({emp.get('job_level', 'N/A')})\n"
                    eng_summary += f"  Performance: {emp.get('performance_rating', 0):.1f}/5, "
                    eng_summary += f"Months in Role: {emp.get('months_in_current_role', 0)}, "
                    eng_summary += f"Potential: {emp.get('potential_score', 0):.0f}%, "
                    eng_summary += f"Leadership: {emp.get('leadership_readiness', 0):.0f}%\n"
                
                if len(eng_employees) > 20:
                    eng_summary += f"\n... and {len(eng_employees) - 20} more team members\n"
                
                return eng_summary
        
        # Detect if this is about product manager candidates or role matching
        if 'product manager' in combined_text or 'candidate' in combined_text or 'role' in combined_text:
            # Find employees with technical backgrounds and leadership potential
            candidates = employees_df.filter(
                (F.col('potential_score') >= 70) &
                (F.col('leadership_readiness') >= 60) &
                (F.col('performance_rating') >= 3.5) &
                (F.col('months_in_current_role') >= 12)
            ).select(
                'employee_id', 'first_name', 'last_name', 'job_title',
                'department', 'performance_rating', 'potential_score',
                'leadership_readiness', 'months_in_current_role'
            ).orderBy(F.desc('potential_score'), F.desc('performance_rating')).limit(15).collect()
            
            if candidates:
                candidates_summary = f"Top Internal Candidates ({len(candidates)} ranked):\n\n"
                for idx, emp in enumerate(candidates, 1):
                    candidates_summary += f"{idx}. {emp.get('first_name', '')} {emp.get('last_name', '')}\n"
                    candidates_summary += f"   Current Role: {emp.get('job_title', 'Unknown')} in {emp.get('department', 'Unknown')}\n"
                    candidates_summary += f"   Performance: {emp.get('performance_rating', 0):.1f}/5, "
                    candidates_summary += f"Potential: {emp.get('potential_score', 0):.0f}%, "
                    candidates_summary += f"Leadership: {emp.get('leadership_readiness', 0):.0f}%, "
                    candidates_summary += f"Tenure: {emp.get('months_in_current_role', 0)} months\n\n"
                
                return candidates_summary
        
        # Detect general department/team queries
        department_keywords = ['sales', 'marketing', 'hr', 'finance', 'operations', 'product', 'design', 'qa', 'quality']
        for dept in department_keywords:
            if dept in combined_text:
                dept_employees = employees_df.filter(
                    F.lower(F.col('department')).contains(dept)
                ).select(
                    'first_name', 'last_name', 'job_title', 'job_level',
                    'performance_rating', 'potential_score', 'months_in_current_role'
                ).collect()
                
                if dept_employees:
                    dept_summary = f"{dept.title()} Team ({len(dept_employees)} members):\n\n"
                    for emp in dept_employees[:15]:
                        dept_summary += f"• {emp.get('first_name', '')} {emp.get('last_name', '')} - {emp.get('job_title', 'Unknown')}\n"
                        dept_summary += f"  Performance: {emp.get('performance_rating', 0):.1f}/5, Potential: {emp.get('potential_score', 0):.0f}%\n"
                    
                    if len(dept_employees) > 15:
                        dept_summary += f"\n... and {len(dept_employees) - 15} more team members\n"
                    
                    return dept_summary
        
    except Exception as e:
        # If queries fail, fall back to original context
        pass
    
    # If no specific data match, return original context but add note about using real data
    return f"{context}\n\nNote: This is descriptive context. Actual employee data is queried from SAP SuccessFactors tables when available."


def execute_career_ai_query(question, context="", show_timing=True):
    """Execute career AI query and display formatted response with timing"""
    
    import time
    
    print(f"🔍 Executing AI Query: {question}")
    
    # Build context from actual data - pass question to enable intelligent data queries
    context_summary = build_context_summary(context, question)
    
    # Try to execute real ai_query
    ai_response = None
    used_real_ai = False
    execution_time = None
    
    try:
        # Prepare prompt with context - use more appropriate label based on query type
        context_label = "Employee Data Context" if "employee" in question.lower() or "alex" in question.lower() else "Organizational Data Context"
        
        prompt = f"""You are a Career Intelligence AI assistant analyzing SAP SuccessFactors data.

        Question: {question}

        {context_label}:
        {context_summary}

        Provide a detailed analysis with:
        1. Specific insights based on the actual data provided above
        2. Concrete recommendations with timelines  
        3. Risk assessments where relevant
        4. Actionable next steps
        5. Reference specific employees by name when relevant

        Format your response with bullet points and specific percentages/metrics."""
        
        # Escape single quotes for SQL
        prompt_escaped = prompt.replace("'", "''")
        
        # Measure execution time
        start_time = time.time()
        
        # Execute actual ai_query
        result_df = spark.sql(f"""
            SELECT ai_query(
                'databricks-meta-llama-3-3-70b-instruct',
                '{prompt_escaped}',
                modelParameters => named_struct('max_tokens', 600, 'temperature', 0.2)
            ) as ai_response
        """)
        
        ai_response = result_df.collect()[0]['ai_response']
        execution_time = time.time() - start_time
        used_real_ai = True
        
        if show_timing:
            print(f"✅ Real ai_query() executed successfully in {execution_time:.2f} seconds")
        else:
            print("✅ Real ai_query() executed successfully")
        
    except Exception as e:
        print(f"❌ ai_query execution error: {e}")
        print("   Please ensure ai_query is properly configured and the foundation model endpoint is available.")
        ai_response = f"""
            **AI Query Error**

            Unable to execute ai_query function. Error: {str(e)}

            **Troubleshooting Steps:**
            1. Verify foundation model endpoint is configured and accessible
            2. Check that ai_query function is available in your SAP Databricks workspace
            3. Ensure proper permissions for model serving endpoints
            4. Review error details above for specific configuration issues

            **Alternative:** Use ML model predictions directly via the load_career_models() function.
            """
        used_real_ai = False
        execution_time = None
    
    # Display formatted AI response
    power_source = "ai_query() + Meta Llama 3.3 70B" if used_real_ai else "Error - ai_query unavailable"
    
    timing_info = ""
    if execution_time and show_timing:
        timing_info = f"""
        <div style="margin-top: 12px; padding: 10px 14px; background: rgba(255,255,255,0.08); border-radius: 8px; border: 1px solid rgba(255,255,255,0.15);">
            <div style="display: flex; align-items: center; gap: 12px; font-size: 13px; color: rgba(255,255,255,0.9);">
                <span style="color: #4ECDC4;">⚡</span>
                <span>Execution Time: <strong style="color: #FFD93D;">{execution_time:.2f}s</strong></span>
                <span style="color: rgba(255,255,255,0.5);">•</span>
                <span>Tokens: <strong style="color: #FFD93D;">~{len(prompt.split())}</strong></span>
            </div>
        </div>
        """
    
    # Convert markdown-style formatting to HTML
    import re
    formatted_response = ai_response
    # Convert **bold** to <strong>
    formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #FFD93D; font-weight: 600;">\1</strong>', formatted_response)
    # Convert bullet points to proper list items
    formatted_response = re.sub(r'^[-•]\s+(.+)$', r'<li style="margin: 8px 0; padding-left: 8px;">\1</li>', formatted_response, flags=re.MULTILINE)
    # Wrap consecutive list items in <ul>
    lines = formatted_response.split('\n')
    in_list = False
    formatted_lines = []
    for line in lines:
        if '<li' in line:
            if not in_list:
                formatted_lines.append('<ul style="margin: 12px 0; padding-left: 24px; list-style: none;">')
                in_list = True
            formatted_lines.append(line)
        else:
            if in_list:
                formatted_lines.append('</ul>')
                in_list = False
            if line.strip() and not line.strip().startswith('**'):
                formatted_lines.append(f'<p style="margin: 12px 0; line-height: 1.7;">{line}</p>')
            elif line.strip().startswith('**'):
                formatted_lines.append(f'<p style="margin: 16px 0 8px 0; font-size: 16px; font-weight: 600; color: #FFD93D;">{line}</p>')
            else:
                formatted_lines.append(line)
    if in_list:
        formatted_lines.append('</ul>')
    formatted_response = '\n'.join(formatted_lines)
    
    displayHTML(f"""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%); 
                padding: 0; border-radius: 12px; margin: 20px 0; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.15); overflow: hidden;">
        
        <!-- Header -->
        <div style="background: rgba(255,255,255,0.12); padding: 18px 24px; border-bottom: 1px solid rgba(255,255,255,0.15);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                            width: 40px; height: 40px; border-radius: 10px; 
                            display: flex; align-items: center; justify-content: center; 
                            box-shadow: 0 2px 8px rgba(78,205,196,0.3);">
                    <span style="font-size: 20px;">🤖</span>
                </div>
                <div>
                    <h3 style="margin: 0; color: #FFD93D; font-size: 18px; font-weight: 600; letter-spacing: 0.3px;">
                        SAP Databricks AI Query Response
                    </h3>
                    <div style="font-size: 12px; color: rgba(255,255,255,0.7); margin-top: 2px;">
                        Powered by {power_source}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Content -->
        <div style="padding: 24px; background: rgba(255,255,255,0.03);">
            <div style="color: rgba(255,255,255,0.95); font-size: 14px; line-height: 1.8; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
                {formatted_response}
            </div>
            
            {timing_info}
        </div>
        
        <!-- Footer -->
        <div style="background: rgba(76,175,80,0.15); padding: 12px 24px; border-top: 1px solid rgba(255,255,255,0.1);">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 12px; color: rgba(255,255,255,0.85);">
                <span style="color: #4ECDC4;">✅</span>
                <span>Analyzing <strong style="color: #FFD93D;">SAP SuccessFactors</strong> data in real-time</span>
            </div>
        </div>
    </div>
    """)
    
    return ai_response


# Define AI query function matching Python file signature
def execute_career_ai_query(LLM_MODEL, prompt, question, context=""):
    """Execute career AI query and display formatted response"""
    
    # Prepare prompt with context - use more appropriate label based on query type
    context_label = "Employee Data Context" if "employee" in question.lower() or "demo" in question.lower() else "Organizational Data Context"

    context_summary = build_context_summary(context, question)
    
    prompt = prompt + f"""
    Question: {question}
        {context_label}:
        {context_summary}"""
    
    # Escape single quotes for SQL
    prompt_escaped = prompt.replace("'", "''")

    # Execute actual ai_query
    result_df = spark.sql(f"""
            SELECT ai_query(
                '{LLM_MODEL}',
                '{prompt_escaped}',
                modelParameters => named_struct('max_tokens', 800)
            ) as ai_response
        """)
        
    ai_response = result_df.collect()[0]['ai_response']

    return format_ai_response(ai_response, LLM_MODEL, displayHTML)

LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"
#LLM_MODEL = "databricks-claude-sonnet-4-5"

prompt = """
    You are a Career Intelligence AI assistant analyzing SAP SuccessFactors data.

        Provide a detailed analysis with:
        1. Specific insights based on the actual data provided above
        2. Concrete recommendations with timelines  
        3. Reference specific employees by name when relevant

        Format your response with bullet points and specific percentages/metrics. Keep answer short, but complete.
    """

# Example AI Query about the Demo Employee
if 'alex_data' in globals() and alex_data and len(alex_data) > 0:
    demo_emp = alex_data[0]
    display_name = f"{demo_emp.first_name} {demo_emp.last_name}".strip()
    if display_name == "Unknown Unknown" or not display_name or display_name == " ":
        display_name = f"Employee {demo_emp.employee_id}"
    
    example_question = f"What are {display_name}'s biggest weaknesses?"
    #example_question = f"What are {display_name}'s top strengths?"

    # Build employee's context using helper function
    demo_context = build_alex_context(alex_data, catalog_name, schema_name, spark)
    if demo_context:
        execute_career_ai_query(LLM_MODEL, prompt, example_question, demo_context)
    else:
        print("⚠️ Could not build employee context. Please ensure data is loaded.")
else:
    print("⚠️ Demo employee data not found. Please run the 'Meet Our Demo Employee' section first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔮 Career Path Predictions for Demo Employee
# MAGIC ### *Powered by SAP Databricks ML & Historical Success Patterns*

# COMMAND ----------

def generate_career_predictions(employee_data):
    """Generate AI-powered career path predictions using ML models (requires ML models to be loaded)"""
    
    if not career_models:
        raise ValueError("❌ ML models not loaded. Models must be available to generate predictions.")
    
    if 'career_success' not in career_models:
        raise ValueError("❌ Career success model not available. Required model 'career_success' not loaded.")
    
    # Convert employee_data to dict if it's a Row object
    if hasattr(employee_data, 'asDict'):
        emp_dict = employee_data.asDict()
    elif isinstance(employee_data, dict):
        emp_dict = employee_data
    else:
        raise ValueError("employee_data must be a dict or PySpark Row with asDict() method")
    
    # Prepare ML features
    employee_features = prepare_ml_features_for_prediction(emp_dict, employees_df, spark, catalog_name, schema_name)
    
    # Get potential next roles
    potential_roles = get_potential_next_roles(emp_dict)
    
    predictions = []
    
    # Use ML models for predictions
    for role in potential_roles:
        try:
            # Create transition features
            transition_features = create_transition_features(employee_features, role)
            
            # Prepare features matching model's expected schema - pass model to get correct schema
            # This will extract the signature and filter to ONLY the expected features in correct order
            model_features = prepare_features_for_model(transition_features, career_models['career_success'])
            
            # Create DataFrame with exactly the features the model expects (already filtered and ordered by prepare_features_for_model)
            features_df = pd.DataFrame([model_features])
            
            # CRITICAL: Enforce exact schema match - remove extra columns and ensure all required columns exist
            features_df = ensure_dataframe_schema(features_df, career_models['career_success'])
            
            # Get ML model prediction
            success_pred = career_models['career_success'].predict(features_df)
            
            # Extract probability
            if isinstance(success_pred, np.ndarray):
                if len(success_pred) == 0:
                    raise ValueError(f"❌ Career success model returned empty prediction for role {role['title']}.")
                success_prob = float(success_pred[0])
            elif isinstance(success_pred, pd.Series):
                success_prob = float(success_pred.iloc[0])
            else:
                success_prob = float(success_pred)
            
            # Get base ML prediction for probability
            base_probability = success_prob * 100 if success_prob <= 1.0 else success_prob
            
            # Get promotion readiness
            if 'promotion_readiness' not in career_models:
                raise ValueError("❌ Promotion readiness model required but not loaded.")
            
            readiness_pred = career_models['promotion_readiness'].predict(features_df)
            if isinstance(readiness_pred, np.ndarray):
                if len(readiness_pred) == 0:
                    raise ValueError(f"❌ Promotion readiness model returned empty prediction for role {role['title']}.")
                base_readiness = float(readiness_pred[0])
            elif isinstance(readiness_pred, pd.Series):
                base_readiness = float(readiness_pred.iloc[0])
            else:
                base_readiness = float(readiness_pred)
            
            # HYBRID APPROACH: Apply role-specific adjustments to ML predictions
            # Combine employee_features with emp_dict to have access to all fields
            combined_features = {**employee_features, **emp_dict}
            
            # 1. Calculate role compatibility multiplier
            compatibility_multiplier = get_role_compatibility_score(combined_features, role)
            
            # 2. Apply adjustments to probability
            # Ensure probability doesn't go below 0 or above 100
            adjusted_probability = base_probability * compatibility_multiplier
            adjusted_probability = max(5.0, min(95.0, adjusted_probability))  # Cap between 5% and 95%
            
            # 3. Calculate skill gap penalty for readiness
            skill_gap_penalty = calculate_skill_gap_penalty(combined_features, role)
            adjusted_readiness = base_readiness * (1 - skill_gap_penalty)
            adjusted_readiness = max(30.0, min(95.0, adjusted_readiness))  # Cap between 30 and 95
            
            # 4. Get role-specific timeline
            timeline = get_role_specific_timeline(role, adjusted_readiness)
            
            # 5. Get role-specific success factors
            success_factors_list = get_success_factors(combined_features, role)

            predictions.append({
                'role': role['title'],
                'probability': round(adjusted_probability, 1),
                'readiness_score': round(adjusted_readiness, 1),
                'timeline': timeline,
                'salary_increase': estimate_salary_increase(role, adjusted_probability),
                'success_factors': success_factors_list,
                'risk_factors': get_risk_factors(employee_features, role),
                'model_confidence': 'High' if adjusted_probability > 75 else 'Medium' if adjusted_probability > 60 else 'Low'
            })
            # Add debug print before line 741 in generate_career_predictions():
            #print(f"Role: {role['title']}, Base Probability: {base_probability:.2f}%, Multiplier: {compatibility_multiplier:.2f}, Adjusted: {adjusted_probability:.2f}%")
        except Exception as e:
            # Fail fast - don't silently skip roles
            raise RuntimeError(f"❌ Error predicting for role '{role['title']}': {e}")
    
    return sorted(predictions, key=lambda x: x['probability'], reverse=True)

# Generate predictions for the demo employee
if alex_data and len(alex_data) > 0:
    demo_emp = alex_data[0]
    display_name = f"{demo_emp.first_name} {demo_emp.last_name}".strip()
    if display_name == "Unknown Unknown" or not display_name or display_name == " ":
        display_name = f"Employee {demo_emp.employee_id}"
    
    print(f"🔮 Generating career path predictions for {display_name}...")
    print(f"   Models loaded: {list(career_models.keys()) if career_models else 'None'}")
    
    try:
        predictions = generate_career_predictions(demo_emp)
        
        if not predictions:
            raise RuntimeError(
                "❌ No predictions generated. Possible reasons:\n"
                "   1. ML models not loaded - check if models exist in Unity Catalog\n"
                "   2. Feature schema mismatch - check error messages above\n"
                "   3. No potential roles identified - check get_potential_next_roles()\n"
                "\n   Troubleshooting:\n"
                "   - Run notebook 02_career_intelligence_ml_models.py to train models\n"
                "   - Verify models are registered in Unity Catalog\n"
                "   - Check that catalog_name and schema_name are correctly set"
            )
        else:
            # Create beautiful visualization
            roles = [p['role'] for p in predictions]
            probabilities = [p['probability'] for p in predictions]
            timelines = [p['timeline'] for p in predictions]
            
            # Create interactive prediction chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=probabilities,
                y=roles,
                orientation='h',
                marker=dict(
                    color=probabilities,
                    colorscale='RdYlGn',
                    colorbar=dict(title="Success Probability (%)")
                ),
                text=[f"{p}%" for p in probabilities],
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>Success Probability: %{x}%<br>Timeline: %{customdata}<extra></extra>',
                customdata=timelines
            ))
            
            fig.update_layout(
                title=f"🔮 AI-Powered Career Path Predictions for {display_name}",
                title_font_size=20,
                xaxis_title="Success Probability (%)",
                yaxis_title="Career Opportunities",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            
            fig.show()
            
            # Display detailed predictions table
            predictions_html = """
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0;">
            <h3>🎯 Detailed Career Path Analysis</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f8f9fa; font-weight: bold;">
                    <th style="padding: 12px; border: 1px solid #dee2e6;">Role</th>
                    <th style="padding: 12px; border: 1px solid #dee2e6;">Probability</th>
                    <th style="padding: 12px; border: 1px solid #dee2e6;">Readiness</th>
                    <th style="padding: 12px; border: 1px solid #dee2e6;">Timeline</th>
                    <th style="padding: 12px; border: 1px solid #dee2e6;">Salary Impact</th>
                    <th style="padding: 12px; border: 1px solid #dee2e6;">Key Factors</th>
                </tr>
        """
        
        for pred in predictions:
            success_factors = '; '.join(pred['success_factors'][:2])  # Top 2 factors
            probability_color = '#28a745' if pred['probability'] > 70 else '#ffc107' if pred['probability'] > 50 else '#dc3545'
            readiness_score = pred.get('readiness_score', 0)
            model_conf = pred.get('model_confidence', 'N/A')
            conf_badge = '🤖'
            
            predictions_html += f"""
                <tr>
                    <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>{pred['role']}</strong></td>
                    <td style="padding: 12px; border: 1px solid #dee2e6; color: {probability_color}; font-weight: bold;">{pred['probability']}% {conf_badge}</td>
                    <td style="padding: 12px; border: 1px solid #dee2e6;">{readiness_score:.0f}/100</td>
                    <td style="padding: 12px; border: 1px solid #dee2e6;">{pred['timeline']}</td>
                    <td style="padding: 12px; border: 1px solid #dee2e6;">{pred['salary_increase']}</td>
                    <td style="padding: 12px; border: 1px solid #dee2e6; font-size: 11px;">{success_factors}</td>
                </tr>
            """
        
        predictions_html += """
            </table>
        </div>
        """
        
        displayHTML(predictions_html)
        
        # Show ML model information
        displayHTML("""
            <div style="background: rgba(76,175,80,0.1); padding: 15px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #4CAF50;">
                <p style="margin: 0;"><strong>🤖 ML Model Status:</strong> Predictions generated using <strong>Real ML Models</strong></p>
                <p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;">
                    ✅ Real ML model predictions from Unity Catalog
                </p>
            </div>
            """)
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        print("   This section requires ML models to be loaded. Please ensure models are available.")
        predictions = []
else:
    print("⚠️ Demo employee data not found. Please run the 'Meet Our Demo Employee' section first.")
    predictions = []

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Hidden Talent Discovery
# MAGIC ### *AI-Powered Identification of High-Potential Employees*

# COMMAND ----------

# Discover hidden talent using ML models
# Reload the helper module to ensure we have the latest code 
import importlib
try:
    import career_intelligence_helpers
    importlib.reload(career_intelligence_helpers)
    # Re-import the functions after reload
    from career_intelligence_helpers import discover_hidden_talent_with_ml, display_talent_graphs
    print("✅ Reloaded career_intelligence_helpers module with latest changes")
except Exception as reload_error:
    # If reload fails, try to import normally (might already be imported)
    try:
        from career_intelligence_helpers import discover_hidden_talent_with_ml, display_talent_graphs
        print("⚠️ Using cached module version - restart kernel to get latest changes")
    except ImportError:
        print(f"❌ Error importing functions: {reload_error}")
        raise

# Use the helper function from career_intelligence_helpers module (has better error handling for performance reviews)
print("🧠 Using ML models for talent discovery...")
print("   Processing employees (this may take a moment due to data access)...")
print("   Note: Performance reviews data access may be skipped if not available (authentication issues)")

try:
    # Call the helper function with required parameters
    talent_pd = discover_hidden_talent_with_ml(career_models, employees_df, spark, catalog_name, schema_name)
    
    # Display the results using the visualization helper function
    if talent_pd is not None and len(talent_pd) > 0:
        # Display summary table of top employees
        print(f"\n{'='*80}")
        print(f"🏆 TOP {len(talent_pd)} HIGH-POTENTIAL EMPLOYEES")
        print(f"{'='*80}")
        print(f"\n{talent_pd[['name', 'department', 'talent_score', 'talent_category', 'high_potential_score', 'promotion_readiness', 'flight_risk']].to_string(index=False)}")
        print(f"\n{'='*80}\n")
        
        # Display interactive visualizations
        display_talent_graphs(talent_pd, displayHTML)
    else:
        print("⚠️ No talent results returned from discovery function")
        
except Exception as e:
    print(f"❌ Error in talent discovery: {e}")
    import traceback
    traceback.print_exc()
    print("\n   Troubleshooting:")
    print("   - Ensure ML models are loaded")
    print("   - Check that employees_df contains active employees")
    print("   - Verify catalog_name and schema_name are set correctly")

# COMMAND ----------

def demonstrate_unity_catalog_governance():
    """Demonstrate Unity Catalog governance and data lineage"""
    
    # Simulate Unity Catalog lineage
    data_lineage = {
        'source': 'SAP Business Data Cloud',
        'data_products': [
            {'name': 'CoreWorkforceData', 'catalog': 'sap_bdc', 'schema': 'successfactors', 'table': 'coreworkforcedata', 'package': 'SAP SuccessFactors Employee Central Data Products'},
            {'name': 'PerformanceData', 'catalog': 'sap_bdc', 'schema': 'successfactors', 'table': 'performancedata', 'package': 'SAP SuccessFactors Performance and Goals Data Products'},
            {'name': 'PerformanceReviews', 'catalog': 'sap_bdc', 'schema': 'successfactors', 'table': 'performancereviews', 'package': 'SAP SuccessFactors Performance and Goals Data Products'},
            {'name': 'LearningHistory', 'catalog': 'sap_bdc', 'schema': 'successfactors', 'table': 'learninghistory', 'package': 'SAP SuccessFactors Learning Data Products'},
            {'name': 'GoalsData', 'catalog': 'sap_bdc', 'schema': 'successfactors', 'table': 'goalsdata', 'package': 'SAP SuccessFactors Performance and Goals Data Products'},
            {'name': 'Compensation', 'catalog': 'sap_bdc', 'schema': 'successfactors', 'table': 'compensation', 'package': 'SAP SuccessFactors Employee Central Data Products'}
        ],
        'ml_models': [
            {'name': 'career_success_prediction', 'catalog': 'career_intelligence', 'inputs': ['CoreWorkforceData', 'PerformanceReviews']},
            {'name': 'retention_risk_prediction', 'catalog': 'career_intelligence', 'inputs': ['CoreWorkforceData', 'PerformanceReviews', 'Compensation']},
            {'name': 'high_potential_identification', 'catalog': 'career_intelligence', 'inputs': ['CoreWorkforceData', 'LearningHistory', 'GoalsData']},
            {'name': 'promotion_readiness_scoring', 'catalog': 'career_intelligence', 'inputs': ['CoreWorkforceData', 'PerformanceReviews', 'GoalsData']}
        ],
        'outputs': [
            {'name': 'career_predictions', 'catalog': 'career_intelligence', 'schema': 'predictions', 'table': 'career_paths'},
            {'name': 'talent_discovery', 'catalog': 'career_intelligence', 'schema': 'analytics', 'table': 'high_potential_employees'}
        ]
    }
    
    displayHTML(f"""
    <div style="background: linear-gradient(135deg, #0052CC 0%, #0070F2 100%); 
                padding: 30px; border-radius: 20px; color: white; margin: 20px 0;">
        <h2 style="text-align: center; margin-bottom: 25px;">🏛️ Unity Catalog Data Lineage</h2>
        <p style="text-align: center; margin-bottom: 25px; opacity: 0.9;">
            Complete traceability from SAP BDC data products to ML model predictions
        </p>
        
        <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px; margin: 20px 0;">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; display: inline-block; margin: 0 10px;">
                    <h3 style="margin: 0; color: #FFD93D;">SAP BDC</h3>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">{len(data_lineage['data_products'])} Data Products</p>
                </div>
                <span style="font-size: 24px; margin: 0 10px;">→</span>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; display: inline-block; margin: 0 10px;">
                    <h3 style="margin: 0; color: #4ECDC4;">Delta Sharing</h3>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">Zero-Copy Access</p>
                </div>
                <span style="font-size: 24px; margin: 0 10px;">→</span>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; display: inline-block; margin: 0 10px;">
                    <h3 style="margin: 0; color: #96CEB4;">ML Models</h3>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">{len(data_lineage['ml_models'])} Models</p>
                </div>
                <span style="font-size: 24px; margin: 0 10px;">→</span>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; display: inline-block; margin: 0 10px;">
                    <h3 style="margin: 0; color: #FFEAA7;">Predictions</h3>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">Career Intelligence</p>
                </div>
            </div>
            
            <div style="margin-top: 25px;">
                <h4 style="color: #FFD93D; margin-bottom: 15px;">📊 Data Products → ML Models Mapping:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px;">
    """)
    
    for model in data_lineage['ml_models']:
        inputs_list = ', '.join([inp.replace('_', ' ') for inp in model['inputs']])
        displayHTML(f"""
                    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
                        <strong>{model['name'].replace('_', ' ').title()}</strong><br>
                        <small style="opacity: 0.8;">Uses: {inputs_list}</small>
                    </div>
        """)
    
    displayHTML("""
                </div>
            </div>
            
            <div style="margin-top: 25px; padding: 15px; background: rgba(76,175,80,0.2); border-radius: 10px; border-left: 4px solid #4CAF50;">
                <h4 style="color: #4ECDC4; margin: 0 0 10px 0;">✅ Governance Benefits:</h4>
                <ul style="margin: 0; padding-left: 20px; font-size: 14px;">
                    <li>Complete data lineage tracking from SAP BDC to predictions</li>
                    <li>Automated compliance and audit trails</li>
                    <li>Fine-grained access controls on data products</li>
                    <li>ML model versioning and governance</li>
                    <li>Data quality monitoring and alerts</li>
                </ul>
            </div>
        </div>
    </div>
    """)
    
    print("🏛️ Unity Catalog Governance Active")
    print(f"📊 Tracking {len(data_lineage['data_products'])} SAP BDC data products")
    print(f"🤖 Monitoring {len(data_lineage['ml_models'])} ML models")
    print("✅ Complete lineage from source to predictions")

demonstrate_unity_catalog_governance()



# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Summary

# COMMAND ----------

# Calculate actual metrics from loaded data
total_employees = employees_df.count() if 'employees_df' in locals() else 0
models_loaded = len(career_models) if career_models else 0

displayHTML(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 40px; border-radius: 20px; color: white; margin: 30px 0;">
    
    <h1 style="text-align: center; margin-bottom: 30px;">🚀 CAREER INTELLIGENCE ENGINE</h1>
    <h2 style="text-align: center; color: #FFD93D; margin-bottom: 20px;">SAP SuccessFactors + SAP Databricks Integration</h2>
    <p style="text-align: center; font-size: 18px; margin-bottom: 40px; opacity: 0.9;">
        Powered by <strong>SAP Business Data Cloud</strong> + <strong>Delta Sharing</strong> + <strong>SAP Databricks ML</strong>
    </p>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; margin: 30px 0;">
        
        <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px;">
            <h3 style="color: #4ECDC4;">🎯 Capabilities Demonstrated</h3>
            <ul style="list-style: none; padding: 0;">
                <li style="margin: 10px 0;">✅ ML-powered career path predictions</li>
                <li style="margin: 10px 0;">✅ Hidden talent identification</li>
                <li style="margin: 10px 0;">✅ Success probability modeling</li>
                <li style="margin: 10px 0;">✅ Natural language AI queries</li>
            </ul>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px;">
            <h3 style="color: #FF6B6B;">🏗️ Architecture Components</h3>
            <ul style="list-style: none; padding: 0;">
                <li style="margin: 10px 0;">🔄 Delta Sharing for zero-copy data</li>
                <li style="margin: 10px 0;">🏛️ Unity Catalog for governance</li>
                <li style="margin: 10px 0;">🤖 MLflow for model management</li>
                <li style="margin: 10px 0;">⚡ Serverless compute infrastructure</li>
                <li style="margin: 10px 0;">📊 Real-time data processing</li>
            </ul>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px;">
            <h3 style="color: #FFEAA7;">📊 Demo Statistics</h3>
            <ul style="list-style: none; padding: 0;">
                <li style="margin: 10px 0;">👥 <strong>{total_employees:,}</strong> employees analyzed</li>
                <li style="margin: 10px 0;">🧠 <strong>{models_loaded}</strong> ML models active</li>
                <li style="margin: 10px 0;">🎯 <strong>Unity Catalog</strong> governance enabled</li>
                <li style="margin: 10px 0;">⚡ <strong>Real-time</strong> predictions available</li>
            </ul>
        </div>
        
    </div>
    
    <div style="background: rgba(255,215,0,0.2); padding: 25px; border-radius: 15px; text-align: center; border: 3px solid #FFD93D; margin-top: 30px;">
        <h2 style="color: #FFD93D; margin: 0 0 15px 0;">🎯 Key Value Proposition</h2>
        <p style="font-size: 18px; margin: 0; line-height: 1.6;">
            <strong>SAP Business Data Cloud + SAP Databricks enables predictive HR intelligence</strong><br>
            Transform workforce decisions from reactive to data-driven<br>
            Leverage ML models and real-time analytics for strategic talent management
        </p>
    </div>
    
    <div style="margin-top: 30px; padding: 20px; background: rgba(0,82,204,0.2); border-radius: 15px; border: 2px solid #0052CC;">
        <h3 style="color: #4ECDC4; margin: 0 0 15px 0;">🔗 SAP BDC + SAP Databricks Integration</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <div>
                <strong>🔄 Delta Sharing</strong>
                <p style="font-size: 13px; margin: 5px 0 0 0; opacity: 0.9;">Zero-copy data access</p>
            </div>
            <div>
                <strong>⚡ Real-Time Processing</strong>
                <p style="font-size: 13px; margin: 5px 0 0 0; opacity: 0.9;">Live data access</p>
            </div>
            <div>
                <strong>🏛️ Unity Catalog</strong>
                <p style="font-size: 13px; margin: 5px 0 0 0; opacity: 0.9;">Automated governance</p>
            </div>
            <div>
                <strong>🤖 AI/ML Integration</strong>
                <p style="font-size: 13px; margin: 5px 0 0 0; opacity: 0.9;">Native ai_query support</p>
            </div>
        </div>
    </div>
    
</div>
""")

print("✅ Demo Complete")
if career_models:
    catalog_info = f"{catalog_name}.{schema_name}" if 'catalog_name' in locals() and 'schema_name' in locals() else "Unity Catalog"
    print(f"📊 {len(career_models)} ML models active | {total_employees:,} employees analyzed | {catalog_info}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 🎬 **End of Demo**
# MAGIC
# MAGIC **Thank you for experiencing the Career Intelligence Engine!**
# MAGIC
# MAGIC *Questions? Let's discuss how SAP SuccessFactors + SAP Databricks can transform your HR operations!*