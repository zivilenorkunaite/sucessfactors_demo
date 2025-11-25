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
%pip install plotly>=5.18.0 mlflow>=2.8.0 

# COMMAND ----------

# Restart Python to ensure all libraries are properly loaded
%restart_python

# COMMAND ----------

# Import configuration and load data
%run ./setup_config.py

# COMMAND ----------

from career_intelligence_helpers import *

# Load data from Unity Catalog
print(f"📋 Loading from Unity Catalog: {catalog_name}.{schema_name}")

# Use the catalog and schema
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")

mlflow_client, employees_df = init_environment(catalog_name, schema_name,displayHTML,spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧠 ML Model Integration
# MAGIC ### *Load Trained ML Models from Unity Catalog*

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
# MAGIC ### Meet Alex Smith

# COMMAND ----------

demo_employee_data, demo_employee_id = get_demo_employee_data(employees_df, displayHTML)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💬 Ask Questions About Our Demo Employee
# MAGIC
# MAGIC Use AI-powered queries to analyze their career profile and get intelligent insights.
# MAGIC
# MAGIC **🔴 LIVE DEMO:** Modify the `example_question` variable below and re-run this cell to show instant AI responses!

# COMMAND ----------

def execute_career_ai_query(LLM_MODEL, prompt, question, context=""):
    """Execute career AI query and display formatted response"""
    
    # Prepare prompt with context - use more appropriate label based on query type
    context_label = "Employee Data Context" if "employee" in question.lower() or "demo employee" in question.lower() else "Organizational Data Context"

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



# COMMAND ----------

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
example_question = "What are the demo employee's biggest weaknesses?"

demo_employee_context = build_demo_employee_context(demo_employee_data, catalog_name, schema_name, spark)
execute_career_ai_query(LLM_MODEL, prompt, example_question, demo_employee_context)




# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔮 Career Path Predictions for Alex
# MAGIC ### *Powered by SAP Databricks ML & Historical Success Patterns*

# COMMAND ----------

get_demo_employees_career_predictions(demo_employee_data, career_models, employees_df, displayHTML, spark, catalog_name, schema_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Hidden Talent Discovery
# MAGIC ### *AI-Powered Identification of High-Potential Employees*

# COMMAND ----------

# Discover hidden talent
talent_pd = discover_hidden_talent_with_ml(career_models, employees_df, spark, catalog_name, schema_name)

display_talent_graphs(talent_pd, displayHTML)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏛️ Share data with SAP BDC via custom data product
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, from_json, to_json, struct, concat, lit, array, explode

summary_prompt = """You are an AI assistant specialized in analyzing SAP success factors master data.

Your task is to analyze the empoloyee record provided, including additional columns that were calculated earlier:

Provide a single sentence answer explaining why this employee is ready or not ready for promotion. Use employees first name in your answer.

IMPORTANT:
- Output must consist of only analysis — no introductory or closing statements.
- Keep the output clean and valid: analysis.

Employee record:
"""

# Convert pandas DataFrame to Spark DataFrame
talent_spark_df = spark.createDataFrame(talent_pd)

# Convert row to JSON text for prompt input
df_with_text = talent_spark_df.withColumn("employee_text", to_json(struct("*")))

# Create the full prompt by concatenating the prompt with the customer text
df_with_text = df_with_text.withColumn("full_prompt", concat(lit(summary_prompt), df_with_text.employee_text))

# Call ai_query with full prompt
ai_query_expr = f"""
  ai_query(
    endpoint => '{LLM_MODEL}',
    request => full_prompt,
    modelParameters => named_struct('temperature', 0.0)
  ) AS llm_response
"""

# Only add the LLM response column
df_result = df_with_text.selectExpr("*", ai_query_expr).drop("employee_text", "full_prompt")

# Show all original columns + one "llm_response" column on the far right
display(df_result)

# Write to Delta table
df_result.write.format("delta").option("mergeSchema", "true").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.calculated_talent_metrics"
)

# Check if table already has a primary key and create one if it does not
pk_check = spark.sql(f"""
    SELECT constraint_name
    FROM {catalog_name}.information_schema.table_constraints
    WHERE table_schema = '{schema_name}'
      AND table_name = 'calculated_talent_metrics'
      AND constraint_type = 'PRIMARY KEY'
""")

if pk_check.count() == 0:
    spark.sql(f"""
        ALTER TABLE {catalog_name}.{schema_name}.calculated_talent_metrics
        ALTER COLUMN employee_id SET NOT NULL
    """)
    spark.sql(f"""
        ALTER TABLE {catalog_name}.{schema_name}.calculated_talent_metrics
        ADD PRIMARY KEY (employee_id) RELY
    """)


# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## 🎯 Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 🎬 **End of Demo**
# MAGIC
# MAGIC **Thank you for checking out the Career Intelligence Engine!**
# MAGIC
# MAGIC *Questions? Let's discuss how SAP SuccessFactors + SAP Databricks can transform your HR operations!*
