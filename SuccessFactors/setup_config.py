"""
SAP Demo Configuration Module
Unity Catalog setup and configuration for SAP Career Intelligence demo
"""

# Import default configuration
try:
    from app_config import DEFAULT_CATALOG_NAME, DEFAULT_SCHEMA_NAME
except ImportError:
    # Fallback if app_config.py doesn't exist
    DEFAULT_CATALOG_NAME = "demos"
    DEFAULT_SCHEMA_NAME = "career_path_temp"

def setup_unity_catalog(catalog_name: str, schema_name: str):
    """Create Unity Catalog catalog and schema if they don't exist"""
    
    # Get spark from globals (available in SAP Databricks notebooks)
    spark_obj = globals().get('spark')
    if spark_obj is None:
        # Try to import from pyspark.sql
        try:
            from pyspark.sql import SparkSession
            spark_obj = SparkSession.getActiveSession()
            if spark_obj is None:
                raise NameError(
                    "❌ 'spark' is not available. "
                    "This module must be run in a SAP Databricks notebook environment. "
                    "Make sure you're running this via '%run ./setup_config.py' in a SAP Databricks notebook."
                )
        except ImportError:
            raise NameError(
                "❌ 'spark' is not available and cannot import SparkSession. "
                "This module must be run in a SAP Databricks notebook environment."
            )
    
    print(f"🏛️ Setting up Unity Catalog structure...")
    
    # Try to use the catalog first (it may already exist)
    catalog_exists = False
    try:
        spark_obj.sql(f"USE CATALOG {catalog_name}")
        print(f"✅ Using existing catalog '{catalog_name}'")
        catalog_exists = True
    except Exception:
        # Catalog doesn't exist, try to create it
        catalog_exists = False
    
    # Create catalog only if it doesn't exist
    if not catalog_exists:
        try:
            # Try creating without managed location first (works if default storage is configured)
            spark_obj.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
            spark_obj.sql(f"USE CATALOG {catalog_name}")
            print(f"✅ Catalog '{catalog_name}' created and ready")
        except Exception as e:
            # If creation fails (e.g., needs managed location), try to use it anyway
            # This handles the case where catalog was created via UI but CREATE command needs location
            try:
                spark_obj.sql(f"USE CATALOG {catalog_name}")
                print(f"✅ Using catalog '{catalog_name}' (may have been created via UI)")
            except Exception as e2:
                print(f"❌ Cannot access catalog '{catalog_name}': {e2}")
                print(f"💡 Tip: Create the catalog '{catalog_name}' via SAP Databricks UI or provide a managed location")
                raise
    
    # Create schema if it doesn't exist
    try:
        spark_obj.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        print(f"✅ Schema '{schema_name}' ready")
    except Exception as e:
        print(f"⚠️ Schema '{schema_name}' may already exist: {e}")
    
    # Use the schema
    spark_obj.sql(f"USE SCHEMA {schema_name}")
    
    print(f"✅ Unity Catalog setup complete: {catalog_name}.{schema_name}")
    
    return catalog_name, schema_name


def get_config(catalog: str = None, schema: str = None):
    """
    Get or setup Unity Catalog configuration
    
    Args:
        catalog: Catalog name (default: from app_config.DEFAULT_CATALOG_NAME)
        schema: Schema name (default: from app_config.DEFAULT_SCHEMA_NAME)
    
    Returns:
        tuple: (catalog_name, schema_name)
    """
    # Use defaults from app_config if not provided
    if catalog is None:
        catalog = DEFAULT_CATALOG_NAME
    if schema is None:
        schema = DEFAULT_SCHEMA_NAME
    # Setup Unity Catalog
    catalog_name, schema_name = setup_unity_catalog(catalog, schema)
    
    return catalog_name, schema_name


# If run directly (e.g., via %run), use widgets if available, otherwise use defaults
if __name__ == "__main__" or "dbutils" in globals():
    try:
        # Try to get from widgets (if called from notebook with widgets set)
        try:
            catalog_name = dbutils.widgets.get("catalog")
            schema_name = dbutils.widgets.get("schema")
            # If widgets return empty strings, use defaults from app_config
            if not catalog_name:
                catalog_name = DEFAULT_CATALOG_NAME
            if not schema_name:
                schema_name = DEFAULT_SCHEMA_NAME
        except:
            # Widgets not set, use defaults from app_config
            catalog_name = DEFAULT_CATALOG_NAME
            schema_name = DEFAULT_SCHEMA_NAME
    except NameError:
        # dbutils not available, use defaults from app_config
        catalog_name = DEFAULT_CATALOG_NAME
        schema_name = DEFAULT_SCHEMA_NAME
    
    print(f"📋 Configuration:")
    print(f"   Catalog: {catalog_name}")
    print(f"   Schema: {schema_name}")
    
    # Run setup
    catalog_name, schema_name = get_config(catalog_name, schema_name)

