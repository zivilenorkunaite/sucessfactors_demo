# Databricks notebook source
# MAGIC %md
# MAGIC # 📊 Data Generation
# MAGIC ## SAP SuccessFactors BDC Data Products Simulation
# MAGIC
# MAGIC This notebook generates realistic SAP SuccessFactors data matching SAP Business Data Cloud (BDC) data products schema.
# MAGIC
# MAGIC **Unity Catalog Setup:** This notebook automatically runs `%run ./setup_config.py` to configure Unity Catalog.
# MAGIC
# MAGIC **Run this notebook first** to generate data before training ML models or running the demo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Configuration & Setup
# MAGIC
# MAGIC **Note:** This notebook automatically runs `setup_config.py` which sets up Unity Catalog using defaults from `app_config.py` (`demos.career_path_temp`). To customize, either:
# MAGIC - Update `app_config.py` to change defaults, or
# MAGIC - Add widgets before running to override defaults

# COMMAND ----------

# Load configuration from app_config first
# Use %run to execute app_config.py and make constants available
%run ./app_config

# COMMAND ----------

# Import configuration from setup module
# This will set catalog_name and schema_name variables
%run ./setup_config.py

# COMMAND ----------

!pip install Faker


# COMMAND ----------


# Configuration is now available from setup_config module
print(f"📋 Using Unity Catalog:")
print(f"   Catalog: {catalog_name}")
print(f"   Schema: {schema_name}")

# Use the catalog and schema
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import numpy as np
from datetime import datetime, timedelta, date
import random
import warnings
warnings.filterwarnings('ignore')

# Configure for optimal display
displayHTML("<style>div.output_subarea { max-width: 100%; }</style>")

print("✅ Data generation environment ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🗃️ SAP SuccessFactors BDC Data Products Generation

# COMMAND ----------

# COMMAND ----------

# ============================================================================
# CONSTANTS & THRESHOLDS
# ============================================================================

# Employment Status Values
ACTIVE_STATUS_VALUES = ['A', 'ACTIVE', 'ACT']
EMPLOYMENT_TYPES = ['Full-time', 'Part-time', 'Contract']
EMPLOYMENT_STATUS_OPTIONS = ['Active', 'Terminated']

# Employment Type Code Mapping (from SAP SuccessFactors Data Product)
EMPLOYMENT_TYPE_CODE_MAPPING = {
    3631: 'Full-time',
    3637: 'Part-time',
    3638: 'Contract'
}
# Default for all other codes
DEFAULT_EMPLOYMENT_TYPE = 'Other'

# Employee IDs
ALEX_EMPLOYEE_ID = '100038'

# Job Level Thresholds
MIN_MANAGER_LEVEL = 2
MIN_SENIOR_LEVEL = 3
MAX_NON_MANAGER_LEVEL = 1  # < 2

# Performance Rating Thresholds
TOP_PERFORMER_THRESHOLD = 4.5
HIGH_PERFORMER_THRESHOLD = 4.0
GOOD_PERFORMER_THRESHOLD = 3.5
AVERAGE_PERFORMER_THRESHOLD = 3.0
DEFAULT_RATING = 3.0

# Tenure & Time Thresholds (in months/days)
MIN_TENURE_MONTHS_FOR_REVIEWS = 6
TENURE_FACTOR_BASE_MONTHS = 36.0  # 3 years
HIGH_PERFORMER_LEARNING_START_DAYS = 30
STANDARD_LEARNING_START_DAYS = 60
PROMOTION_WINDOW_DAYS = 90
REVIEW_RELEVANCE_WINDOW_DAYS = 180
MAX_LEARNING_OFFSET_DAYS = 365  # 1 year cap
GOAL_START_RANGE_DAYS = 90
GOAL_DURATION_MIN_DAYS = 90
GOAL_DURATION_MAX_DAYS = 365

# Achievement Percentage Thresholds
FULL_ACHIEVEMENT_THRESHOLD = 100
GOOD_ACHIEVEMENT_THRESHOLD = 80
MAX_ACHIEVEMENT_PCT = 120

# Score Thresholds
MIN_SCORE = 60
MAX_SCORE = 100
NULL_SCORE_PROBABILITY = 0.15

# Score Statistics (mean, std) by Performance Level
HIGH_PERFORMER_SCORE_MEAN = 88
HIGH_PERFORMER_SCORE_STD = 8
GOOD_PERFORMER_SCORE_MEAN = 82
GOOD_PERFORMER_SCORE_STD = 10
LOWER_PERFORMER_SCORE_MEAN = 75
LOWER_PERFORMER_SCORE_STD = 12

# Salary Increase Percentages by Performance Level
PROMOTION_INCREASE_MIN = 0.10
PROMOTION_INCREASE_MAX = 0.20
TOP_PERFORMER_INCREASE_MIN = 0.08
TOP_PERFORMER_INCREASE_MAX = 0.12
HIGH_PERFORMER_INCREASE_MIN = 0.06
HIGH_PERFORMER_INCREASE_MAX = 0.09
GOOD_PERFORMER_INCREASE_MIN = 0.04
GOOD_PERFORMER_INCREASE_MAX = 0.06
AVERAGE_PERFORMER_INCREASE_MIN = 0.02
AVERAGE_PERFORMER_INCREASE_MAX = 0.04
LOW_PERFORMER_INCREASE_MIN = 0.0
LOW_PERFORMER_INCREASE_MAX = 0.02
DEFAULT_INCREASE_MEAN = 0.04
DEFAULT_INCREASE_STD = 0.02

# Bonus Target Ranges (percentage of salary)
SENIOR_LEVEL_BONUS_MIN = 15
SENIOR_LEVEL_BONUS_MAX = 30
MANAGER_LEVEL_BONUS_MIN = 10
MANAGER_LEVEL_BONUS_MAX = 20

# Equity Ranges (in currency units)
SENIOR_LEVEL_EQUITY_MIN = 10000
SENIOR_LEVEL_EQUITY_MAX = 80000
MANAGER_LEVEL_EQUITY_MIN = 0
MANAGER_LEVEL_EQUITY_MAX = 30000

# Default Values for Data Product Transformations
DEFAULT_AGE = 30
DEFAULT_SALARY = 0
DEFAULT_TENURE_MONTHS = 0
DEFAULT_GOALS_ACHIEVEMENT = 60
DEFAULT_REVIEW_YEAR = 2024

# Completion Status Mappings (from Data Product)
COMPLETION_STATUS_ID_COMPLETED = "1"
COMPLETION_STATUS_ID_IN_PROGRESS = "2"
COMPLETION_STATUS_ID_NOT_STARTED = "3"
COMPLETION_STATUS_COMPLETED = "Completed"
COMPLETION_STATUS_IN_PROGRESS = "In Progress"
COMPLETION_STATUS_NOT_STARTED = "Not Started"
COMPLETION_STATUS_OVERDUE = "Overdue"

# Learning Status Weights [Completed, In Progress, Not Started] by Performance Level
HIGH_PERFORMER_LEARNING_WEIGHTS = [85, 12, 3]
GOOD_PERFORMER_LEARNING_WEIGHTS = [75, 20, 5]
LOWER_PERFORMER_LEARNING_WEIGHTS = [60, 25, 15]

# Goal Status Weights [Completed, In Progress, Overdue] by Achievement Level
FULL_ACHIEVEMENT_GOAL_WEIGHTS = [70, 20, 10]
GOOD_ACHIEVEMENT_GOAL_WEIGHTS = [50, 40, 10]
LOW_ACHIEVEMENT_GOAL_WEIGHTS = [30, 40, 30]

# Learning Category Keywords (for mapping from data product)
LEARNING_CATEGORY_KEYWORDS = {
    "technical": "Technical Skills",
    "leadership": "Leadership",
    "communication": "Communication",
    "project": "Project Management",
    "data": "Data Analysis",
    "product": "Product Management",
    "sales": "Sales Training",
    "compliance": "Compliance"
}

# Special Learning Categories (longer hours)
LONG_DURATION_CATEGORIES = ['Leadership', 'Strategic Planning']

# Job Title Pattern Matching (for job level derivation)
JOB_TITLE_PATTERN_SENIOR = "Manager|Director|VP|Chief"
JOB_TITLE_PATTERN_MID = "Senior|Lead|Principal|Staff"

# Date Column Name Variations (for deduplication)
START_DATE_COLUMN_VARIANTS = ['startDate', 'start_date', 'START_DATE', 'startDateCalc', 'effectiveStartDate']

# Row Number for Latest Record
LATEST_RECORD_ROW_NUM = 1

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def is_employee_active(employment_status):
    """
    Check if employee is active based on employment status.
    Handles various formats: 'Active', 'ACTIVE', 'A', 'Act', etc.
    
    Args:
        employment_status: Employment status value (string)
        
    Returns:
        bool: True if employee is active, False otherwise
    """
    if employment_status is None:
        return False
    emp_status = str(employment_status).strip().upper()
    # 'A' and 'ACTIVE' both mean active
    return emp_status in ACTIVE_STATUS_VALUES


def generate_employees():
    """Generate employee data"""
    random.seed(42)
    np.random.seed(42)
    
    print("🔄 Generating employees data...")
    employees = []
    
    # Realistic organizational data matching SAP SuccessFactors structure - Australia focused
    # Department codes: 1034=Engineering, 1035=Product, 1036=Sales, 1037=Marketing, 1038=Finance, 1039=HR, 1040=Operations, 1041=Legal
    departments = [1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041]
    locations = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide']  # Australian cities
    job_families = {
        1034: ['Software Engineer', 'Senior Software Engineer', 'Staff Engineer', 'Engineering Manager', 'Director Engineering'],  # Engineering
        1035: ['Product Analyst', 'Product Manager', 'Senior PM', 'Principal PM', 'VP Product'],  # Product
        1036: ['Sales Rep', 'Account Executive', 'Sales Manager', 'Regional Director', 'VP Sales'],  # Sales
        1037: ['Marketing Specialist', 'Marketing Manager', 'Senior Manager', 'Marketing Director', 'CMO'],  # Marketing
        1038: ['Financial Analyst', 'Senior Analyst', 'Finance Manager', 'Finance Director', 'CFO'],  # Finance
        1039: ['HR Generalist', 'HR Business Partner', 'HR Manager', 'HR Director', 'CHRO'],  # HR
        1040: ['Operations Analyst', 'Operations Manager', 'Senior Manager', 'Operations Director', 'COO'],  # Operations
        1041: ['Legal Counsel', 'Senior Counsel', 'Legal Director', 'General Counsel']  # Legal
    }
    
    employee_id = 100000
    
    # First, create Alex Smith with specific characteristics for demo
    alex_id = 100038  # Ensure Alex gets this ID
    alex_hire_date = date.today() - timedelta(days=540)  # 18 months ago
    alex_job_start = alex_hire_date + timedelta(days=30)
    
    employees.append({
        'employee_id': str(alex_id),
        'person_id': f'PER{alex_id + 50000}',
        'first_name': 'Alex',
        'last_name': 'Smith',
        'gender': 'Female',
        'age': 32,
        'hire_date': alex_hire_date,
        'current_job_start_date': alex_job_start,
        'department': 1034, #'Engineering',
        'department_name': DEPARTMENT_CODE_TO_NAME.get(1034, 'Unknown'),
        'job_title': 'Software Engineer',
        'job_level': 1,
        'location': 53,  # Australia location code
        'location_name': 'Australia',
        'employment_status': 'Active',
        'employment_type': 'Full-time',
        'base_salary': 110000,  # AUD - Sydney Software Engineer rate
        'tenure_months': 18,
        'months_in_current_role': 17,
        'manager_id': None  # Will be set later
    })
    
    for dept in departments:
        dept_size = random.randint(50, 200)  # Realistic department sizes
        
        for _ in range(dept_size):
            employee_id += 1
            
            # Skip if we already created Alex
            if employee_id == alex_id:
                employee_id += 1
            
            # Demographics
            gender = random.choices(['M', 'F', 'NB'], weights=[48, 48, 4])[0]
            age = random.randint(22, 62)
            
            # Employment details
            hire_date = date.today() - timedelta(days=random.randint(30, 2555))  # Up to 7 years
            
            # Job level progression - correlated with tenure (stronger signal)
            job_levels = job_families[dept]
            tenure_years = random.uniform(0.5, 7)
            # Higher tenure tends to have higher levels (but with variance)
            expected_level = min(len(job_levels) - 1, max(0, int(tenure_years / 2.5 + np.random.normal(0, 0.8))))
            current_level_idx = max(0, min(len(job_levels) - 1, expected_level))
            
            # Some employees may have been promoted (job start later than hire)
            if current_level_idx > 0 and random.random() > 0.3:
                # Recent promotion
                max_months_ago = max(1, int(tenure_years * 12))  # Ensure at least 1
                months_ago = random.randint(1, max_months_ago)
                current_job_start = date.today() - timedelta(days=months_ago * 30)
            else:
                # No promotion yet
                current_job_start = hire_date + timedelta(days=random.randint(0, 90))
            
            # Compensation based on level and department (Australian market - AUD)
            base_salary_ranges = {
                0: (70000, 100000),   # Entry level (AUD)
                1: (95000, 140000),   # Mid level (AUD)
                2: (130000, 180000),  # Senior level (AUD)
                3: (170000, 240000),  # Lead level (AUD)
                4: (220000, 350000)   # Executive level (AUD)
            }
            
            salary_range = base_salary_ranges.get(current_level_idx, (70000, 100000))
            # Add some variance but keep correlation strong
            min_salary = int(salary_range[0] * 0.95)
            max_salary = int(salary_range[1] * 1.05)
            # Ensure min <= max
            if min_salary > max_salary:
                min_salary, max_salary = max_salary, min_salary
            base_salary = random.randint(min_salary, max_salary)
            
            # Location affects salary (Australian market rates)
            # Use location code 53 (Australia) for all generated employees
            location_city = random.choice(locations)
            location_code = 53  # Australia location code
            location_name = 'Australia'
            location_multipliers = {
                'Sydney': 1.15,      # Highest cost of living, highest salaries
                'Melbourne': 1.05,   # Second major market
                'Brisbane': 0.95,    # Growing tech hub, slightly lower
                'Perth': 0.92,       # Mining/resources focus
                'Adelaide': 0.88     # Smaller market
            }
            base_salary = int(base_salary * location_multipliers.get(location_city, 1.0))
            
            employees.append({
                'employee_id': str(employee_id),
                'person_id': f'PER{employee_id + 50000}',
                'first_name': random.choice(['Alex', 'Sarah', 'Michael', 'Jessica', 'David', 'Emily', 'Chris', 'Amanda', 'Ryan', 'Lisa', 'John', 'Maria', 'James', 'Jennifer', 'Robert', 'Emma', 'Olivia', 'Charlotte', 'Sophia', 'Isabella', 'Noah', 'Oliver', 'William', 'Lucas', 'Benjamin', 'Mia', 'Harper', 'Evelyn', 'Abigail', 'Elijah', 'Mason', 'Alexander', 'Daniel', 'Matthew', 'Aiden']),
                'last_name': random.choice(['Smith', 'Jones', 'Williams', 'Brown', 'Wilson', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young', 'King', 'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 'Nelson', 'Carter', 'Mitchell', 'Roberts', 'Turner', 'Phillips', 'Campbell', 'Parker', 'Evans', 'Edwards', 'Collins', 'Stewart', 'Sanchez', 'Morris', 'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy', 'Bailey', 'Rivera', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Torres', 'Peterson', 'Gray', 'Ramirez', 'James', 'Watson', 'Brooks', 'Kelly', 'Sanders', 'Price', 'Bennett', 'Wood', 'Barnes', 'Ross', 'Henderson', 'Coleman', 'Jenkins', 'Perry', 'Powell', 'Long', 'Patterson', 'Hughes', 'Flores', 'Washington', 'Butler', 'Simmons', 'Foster', 'Gonzales', 'Bryant', 'Alexander', 'Russell', 'Griffin', 'Diaz', 'Hayes']),
                'gender': gender,
                'age': int(age),
                'hire_date': hire_date,
                'current_job_start_date': current_job_start,
                'department': dept,
                'department_name': DEPARTMENT_CODE_TO_NAME.get(int(dept) if dept is not None else None, 'Unknown'),
                'job_title': job_levels[current_level_idx],
                'job_level': int(current_level_idx),
                'location': location_code,
                'location_name': location_name,
                'employment_status': random.choices(EMPLOYMENT_STATUS_OPTIONS, weights=[92, 8])[0],
                'employment_type': random.choices(EMPLOYMENT_TYPES, weights=[85, 10, 5])[0],
                'base_salary': int(base_salary),
                'tenure_months': int((date.today() - hire_date).days // 30),
                'months_in_current_role': int((date.today() - current_job_start).days // 30),
                'manager_id': None  # Will be set after all employees created
            })
    
    # Set manager relationships (hierarchical structure)
    # Managers are typically at level 2+ and in same department
    managers_by_dept = {}
    for emp in employees:
        if emp['job_level'] >= MIN_MANAGER_LEVEL and is_employee_active(emp.get('employment_status')):
            dept = emp['department']
            if dept not in managers_by_dept:
                managers_by_dept[dept] = []
            managers_by_dept[dept].append(emp['employee_id'])
    
    # Assign managers to employees
    for emp in employees:
        if emp['job_level'] <= MAX_NON_MANAGER_LEVEL and is_employee_active(emp.get('employment_status')):
            dept = emp['department']
            if dept in managers_by_dept and managers_by_dept[dept]:
                emp['manager_id'] = random.choice(managers_by_dept[dept])
    
    # Ensure Alex has a manager
    for emp in employees:
        if emp['employee_id'] == ALEX_EMPLOYEE_ID and not emp.get('manager_id'):
            eng_managers = [e['employee_id'] for e in employees if e['department'] == 1034 and e['job_level'] >= MIN_MANAGER_LEVEL]
            if eng_managers:
                emp['manager_id'] = random.choice(eng_managers)
    
    print("✅ Generated employees data")
    return employees


# COMMAND ----------

def generate_performance_reviews(employees):
    """Generate performance review data based on employees"""
    random.seed(42)
    np.random.seed(42)
    
    print("🔄 Generating performance reviews data...")
    
    # Check input
    if not employees or len(employees) == 0:
        print("   ⚠️ Warning: No employees provided for performance review generation")
        return []
    
    print("   ℹ️ Processing employees for performance reviews")
    
    # Performance Management Data (PerformanceReviews from SAP BDC)
    # Strong signal: Performance correlates with job level, tenure, and creates patterns
    performance_reviews = []
    
    # Calculate base performance potential for each employee (hidden trait)
    employee_performance_potential = {}
    for emp in employees:
        # Check if employee is active (handles 'A', 'Active', 'ACTIVE', etc.)
        if is_employee_active(emp.get('employment_status')):
            # Base potential correlated with job level (higher level = higher base)
            job_level = emp.get('job_level', 1)
            if not isinstance(job_level, (int, float)):
                job_level = 1
            base_potential = 2.5 + (job_level * 0.4)
            # Add variance for individual differences
            individual_factor = np.random.normal(0, 0.5)
            # Tenure helps but with diminishing returns
            tenure_months = emp.get('tenure_months', 0)
            if not isinstance(tenure_months, (int, float)):
                tenure_months = 0
            tenure_factor = min(0.5, tenure_months / 36.0 * 0.3)
            employee_performance_potential[emp['employee_id']] = max(2.0, min(5.0, base_potential + individual_factor + tenure_factor))
    
    # Special handling for Alex Smith - make her a high performer
    employee_performance_potential[ALEX_EMPLOYEE_ID] = 4.2
    
    for emp in employees:
        # Check if employee is active (handles 'A', 'Active', 'ACTIVE', etc.)
        # Get tenure_months with proper handling
        tenure_months = emp.get('tenure_months', 0)
        if not isinstance(tenure_months, (int, float)) or tenure_months is None:
            tenure_months = 0
        
        if is_employee_active(emp.get('employment_status')) and tenure_months >= MIN_TENURE_MONTHS_FOR_REVIEWS:
            # Generate 1-3 performance reviews based on tenure
            num_reviews = min(3, max(1, emp['tenure_months'] // 12))
            base_potential = employee_performance_potential.get(emp['employee_id'], 3.0)
            
            for review_num in range(num_reviews):
                review_date = emp['hire_date'] + timedelta(days=365 * (review_num + 1))
                if review_date <= date.today():
                    
                    # Performance follows realistic patterns with stronger signal
                    if review_num > 0:
                        # Get last rating
                        last_rating = [r for r in performance_reviews if r['employee_id'] == emp['employee_id']][-1]['overall_rating']
                        # Tendency to stay near potential, but can improve or decline slightly
                        performance_change = np.random.normal(0, 0.25)
                        # Pull toward potential (stronger signal)
                        pull_factor = (base_potential - last_rating) * 0.2
                        base_performance = last_rating + performance_change + pull_factor
                    else:
                        # First review starts near potential
                        base_performance = base_potential + np.random.normal(0, 0.4)
                    
                    overall_rating = round(max(1, min(5, base_performance)), 1)
                    
                    # Goals achievement strongly correlated with performance (strong signal)
                    goals_base = overall_rating * 18  # 4.0 rating = 72% base
                    goals_achievement = int(max(0, min(120, np.random.normal(goals_base, 10))))
                    
                    # Competency rating correlates with overall but can vary
                    competency_base = overall_rating + np.random.normal(0, 0.3)
                    competency_rating = round(max(1, min(5, competency_base)), 1)
                    
                    performance_reviews.append({
                        'review_id': f'REV{random.randint(10000, 99999)}',
                        'employee_id': emp['employee_id'],
                        'review_period': int(review_date.year),
                        'review_date': review_date,
                        'overall_rating': float(overall_rating),
                        'goals_achievement': int(goals_achievement),
                        'competency_rating': float(competency_rating),
                        'reviewer_id': emp.get('manager_id', f'EMP{random.randint(100001, 199999)}'),
                        'status': 'Completed'
                    })
    
    print("✅ Generated performance reviews data")
    
    if len(performance_reviews) == 0:
        print("   ⚠️ Warning: No performance reviews generated!")
        print(f"   → Check: employment_status values and tenure_months >= {MIN_TENURE_MONTHS_FOR_REVIEWS}")
        # Show sample employee data for debugging
        if employees:
            sample_emp = employees[0]
            print(f"   → Sample employee: status='{sample_emp.get('employment_status')}', tenure={sample_emp.get('tenure_months')}, job_level={sample_emp.get('job_level')}")
    
    return performance_reviews


# COMMAND ----------

def generate_learning_records(employees, performance_reviews):
    """Generate learning records based on employees and performance"""
    random.seed(42)
    np.random.seed(42)
    
    print("🔄 Generating learning records data...")
    
    # Learning & Development Data (LearningHistory from SAP BDC)
    # Strong signal: High performers take more courses, complete more, score higher
    learning_records = []
    
    learning_categories = ['Technical Skills', 'Leadership', 'Communication', 'Project Management', 
                          'Data Analysis', 'Product Management', 'Sales Training', 'Compliance']
    
    # Calculate base performance potential for each employee (same logic as performance reviews)
    employee_performance_potential = {}
    for emp in employees:
        if is_employee_active(emp.get('employment_status')):
            base_potential = 2.5 + (emp['job_level'] * 0.4)
            individual_factor = np.random.normal(0, 0.5)
            tenure_factor = min(0.5, emp['tenure_months'] / TENURE_FACTOR_BASE_MONTHS * 0.3)
            employee_performance_potential[emp['employee_id']] = max(2.0, min(5.0, base_potential + individual_factor + tenure_factor))
    
    # Special handling for Alex Smith
    employee_performance_potential[ALEX_EMPLOYEE_ID] = 4.2
    
    # Get latest performance for each employee
    latest_performance = {}
    for review in performance_reviews:
        emp_id = review['employee_id']
        if emp_id not in latest_performance or review['review_date'] > latest_performance[emp_id]['review_date']:
            latest_performance[emp_id] = review
    
    for emp in employees:
        if is_employee_active(emp.get('employment_status')):
            # Get employee's performance level (strong signal)
            emp_perf = latest_performance.get(emp['employee_id'], {})
            perf_rating = emp_perf.get('overall_rating', employee_performance_potential.get(emp['employee_id'], 3.0))
            
            # High performers take more courses (strong correlation)
            base_num_courses = 3
            if perf_rating >= HIGH_PERFORMER_THRESHOLD:
                num_activities = random.randint(6, 12)  # High performers very active
            elif perf_rating >= GOOD_PERFORMER_THRESHOLD:
                num_activities = random.randint(4, 8)   # Good performers active
            else:
                num_activities = random.randint(2, 5)   # Lower performers less active
            
            # Alex Smith gets extra learning
            if emp['employee_id'] == ALEX_EMPLOYEE_ID:
                num_activities = 8
            
            # Select relevant categories based on department and level
            relevant_categories = learning_categories.copy()
            if emp['job_level'] >= MIN_MANAGER_LEVEL:
                relevant_categories.extend(['Leadership', 'Strategic Planning'])
            if emp['department'] == 1034: #'Engineering':
                relevant_categories.extend(['Technical Skills', 'Data Analysis'])
            
            for activity_num in range(num_activities):
                # Learning happens throughout tenure, more recently for high performers
                # Ensure valid range (start <= end)
                tenure_days = emp['tenure_months'] * 30
                
                if perf_rating >= HIGH_PERFORMER_THRESHOLD:
                    # High performers: can start learning earlier (within first 30 days)
                    min_days_offset = HIGH_PERFORMER_LEARNING_START_DAYS
                    max_days_offset = min(tenure_days, MAX_LEARNING_OFFSET_DAYS)  # Don't exceed tenure, cap at 1 year
                    if min_days_offset > max_days_offset:
                        min_days_offset = max(1, max_days_offset)  # Adjust if tenure is very short
                    days_offset = random.randint(min_days_offset, max_days_offset)
                else:
                    # Other performers: start learning later (after 60 days)
                    min_days_offset = STANDARD_LEARNING_START_DAYS
                    max_days_offset = min(tenure_days, MAX_LEARNING_OFFSET_DAYS)  # Don't exceed tenure, cap at 1 year
                    if min_days_offset > max_days_offset:
                        min_days_offset = max(1, max_days_offset)  # Adjust if tenure is very short
                    days_offset = random.randint(min_days_offset, max_days_offset)
                
                completion_date = emp['hire_date'] + timedelta(days=days_offset)
                if completion_date > date.today():
                    completion_date = date.today() - timedelta(days=random.randint(1, 90))
                
                # Scores correlate with performance (strong signal)
                if perf_rating >= HIGH_PERFORMER_THRESHOLD:
                    score_mean = HIGH_PERFORMER_SCORE_MEAN
                    score_std = HIGH_PERFORMER_SCORE_STD
                elif perf_rating >= GOOD_PERFORMER_THRESHOLD:
                    score_mean = GOOD_PERFORMER_SCORE_MEAN
                    score_std = GOOD_PERFORMER_SCORE_STD
                else:
                    score_mean = LOWER_PERFORMER_SCORE_MEAN
                    score_std = LOWER_PERFORMER_SCORE_STD
                
                score_value = int(np.random.normal(score_mean, score_std))
                score_value = max(MIN_SCORE, min(MAX_SCORE, score_value)) if random.random() > NULL_SCORE_PROBABILITY else None
                
                # Hours correlate with course importance (leadership courses longer)
                category = random.choice(relevant_categories)
                if category in LONG_DURATION_CATEGORIES:
                    hours = random.randint(8, 40)
                else:
                    hours = random.randint(2, 20)
                
                # Completion status correlates with performance
                if perf_rating >= HIGH_PERFORMER_THRESHOLD:
                    status_weights = HIGH_PERFORMER_LEARNING_WEIGHTS  # High completion rate
                elif perf_rating >= GOOD_PERFORMER_THRESHOLD:
                    status_weights = GOOD_PERFORMER_LEARNING_WEIGHTS
                else:
                    status_weights = LOWER_PERFORMER_LEARNING_WEIGHTS
                
                learning_records.append({
                    'learning_id': f'LRN{random.randint(10000, 99999)}',
                    'employee_id': emp['employee_id'],
                    'course_title': f"{category} Training {random.randint(100, 999)}",
                    'category': category,
                    'completion_date': completion_date,
                    'hours_completed': int(hours),
                    'completion_status': random.choices([COMPLETION_STATUS_COMPLETED, COMPLETION_STATUS_IN_PROGRESS, COMPLETION_STATUS_NOT_STARTED], weights=status_weights)[0],
                    'score': int(score_value) if score_value is not None else None
                })
    
    print("✅ Generated learning records data")
    return learning_records


# COMMAND ----------

def generate_goals(employees, performance_reviews):
    """Generate goals data based on employees and performance"""
    random.seed(42)
    np.random.seed(42)
    
    print("🔄 Generating goals data...")
    
    # Goal Management Data (GoalsData from SAP BDC)
    # Strong signal: Goal achievement correlates with performance ratings
    goals = []
    
    goal_types = ['Revenue Growth', 'Process Improvement', 'Skill Development', 'Team Building', 
                 'Innovation', 'Customer Satisfaction', 'Quality Improvement', 'Cost Reduction']
    
    # Calculate base performance potential for each employee (for all employees, not just active)
    employee_performance_potential = {}
    for emp in employees:
        base_potential = 2.5 + (emp.get('job_level', 1) * 0.4)
        individual_factor = np.random.normal(0, 0.5)
        tenure_factor = min(0.5, emp.get('tenure_months', 0) / TENURE_FACTOR_BASE_MONTHS * 0.3)
        employee_performance_potential[emp['employee_id']] = max(2.0, min(5.0, base_potential + individual_factor + tenure_factor))
    
    # Special handling for Alex Smith
    employee_performance_potential[ALEX_EMPLOYEE_ID] = 4.2
    
    # Get latest performance for each employee
    latest_performance = {}
    for review in performance_reviews:
        emp_id = review['employee_id']
        if emp_id not in latest_performance or review['review_date'] > latest_performance[emp_id]['review_date']:
            latest_performance[emp_id] = review
    
    # Generate goals for ALL employees (not just active)
    for emp in employees:
        # Get employee's performance level
        emp_perf = latest_performance.get(emp['employee_id'], {})
        perf_rating = emp_perf.get('overall_rating', employee_performance_potential.get(emp['employee_id'], 3.0))
        goals_achievement_avg = emp_perf.get('goals_achievement', 85)
        
        # Determine number of goals based on employment status
        is_active = is_employee_active(emp.get('employment_status'))
        if is_active:
            # Active employees: High performers may have more goals
            if perf_rating >= HIGH_PERFORMER_THRESHOLD:
                num_goals = random.randint(4, 6)
            else:
                num_goals = random.randint(2, 5)
        else:
            # Inactive employees: Generate minimal goals (1 goal with lower achievement)
            num_goals = 1
            goals_achievement_avg = max(50, goals_achievement_avg - 20)  # Lower achievement for inactive
            
            for goal_num in range(num_goals):
                goal_start = emp['current_job_start_date'] + timedelta(days=random.randint(0, GOAL_START_RANGE_DAYS))
                goal_end = goal_start + timedelta(days=random.randint(GOAL_DURATION_MIN_DAYS, GOAL_DURATION_MAX_DAYS))
                
                # Achievement strongly correlates with performance (very strong signal)
                # Use the goals_achievement from performance review as base
                achievement_base = goals_achievement_avg
                # Add some variance per goal
                achievement_pct = int(max(0, min(MAX_ACHIEVEMENT_PCT, np.random.normal(achievement_base, 12))))
                
                # Status correlates with achievement
                if achievement_pct >= FULL_ACHIEVEMENT_THRESHOLD:
                    status_weights = FULL_ACHIEVEMENT_GOAL_WEIGHTS  # Mostly completed
                elif achievement_pct >= GOOD_ACHIEVEMENT_THRESHOLD:
                    status_weights = GOOD_ACHIEVEMENT_GOAL_WEIGHTS
                else:
                    status_weights = LOW_ACHIEVEMENT_GOAL_WEIGHTS  # More overdue for low achievers
                
                goals.append({
                    'goal_id': f'GOAL{random.randint(10000, 99999)}',
                    'employee_id': emp['employee_id'],
                    'goal_title': f"{random.choice(goal_types)} Initiative",
                    'goal_type': random.choice(goal_types),
                    'start_date': goal_start,
                    'target_date': goal_end,
                    'achievement_percentage': int(achievement_pct),
                    'weight': int(random.randint(10, 40)),
                    'status': random.choices([COMPLETION_STATUS_COMPLETED, COMPLETION_STATUS_IN_PROGRESS, COMPLETION_STATUS_OVERDUE], weights=status_weights)[0]
                })
    
    print("✅ Generated goals data")
    return goals


# COMMAND ----------

def generate_compensation(employees, performance_reviews):
    """Generate compensation history based on employees and performance"""
    random.seed(42)
    np.random.seed(42)
    
    print("🔄 Generating compensation history data...")
    
    # Compensation History (Compensation from SAP BDC)
    # Strong signal: Salary increases correlate with performance and promotions
    compensation_history = []
    
    # Generate compensation for ALL employees (not just active)
    for emp in employees:
        is_active = is_employee_active(emp.get('employment_status'))
        # Get employee's performance history
        emp_reviews = [r for r in performance_reviews if r['employee_id'] == emp['employee_id']]
        emp_reviews.sort(key=lambda x: x['review_date'])
        
        if is_active:
            # Active employees: Generate salary history with performance-based increases
            num_adjustments = max(1, emp.get('tenure_months', 0) // 12)  # Annual adjustments
            current_salary = emp.get('base_salary', DEFAULT_SALARY)
            starting_salary = int(current_salary * 0.85)  # Starting salary lower
        else:
            # Inactive employees: Generate only initial hire record
            num_adjustments = 1
            current_salary = emp.get('base_salary', DEFAULT_SALARY)
            starting_salary = int(current_salary * 0.85)
        
        for adj_num in range(num_adjustments):
            effective_date = emp.get('hire_date', date.today() - timedelta(days=365 * adj_num))
            if effective_date > date.today():
                effective_date = date.today() - timedelta(days=365 * adj_num)
            
            if adj_num == 0:
                salary = starting_salary
                reason = 'Initial Hire'
            elif not is_active:
                # Inactive employees: only initial hire
                continue
            else:
                # Check if there was a promotion around this time
                job_start = emp.get('current_job_start_date', effective_date)
                days_diff = abs((effective_date - job_start).days)
                
                if days_diff < PROMOTION_WINDOW_DAYS:  # Promotion around this time
                    # Promotion increase: 10-20%
                    increase_pct = random.uniform(PROMOTION_INCREASE_MIN, PROMOTION_INCREASE_MAX)
                    salary = int(salary * (1 + increase_pct))
                    reason = 'Promotion'
                else:
                    # Annual review increase based on performance
                    # Find performance rating around this time
                    relevant_review = None
                    for review in emp_reviews:
                        if abs((review['review_date'] - effective_date).days) < REVIEW_RELEVANCE_WINDOW_DAYS:
                            relevant_review = review
                            break
                    
                    if relevant_review:
                        perf_rating = relevant_review['overall_rating']
                        # Performance-based increases (strong signal)
                        if perf_rating >= TOP_PERFORMER_THRESHOLD:
                            increase_pct = random.uniform(TOP_PERFORMER_INCREASE_MIN, TOP_PERFORMER_INCREASE_MAX)  # Top performers
                        elif perf_rating >= HIGH_PERFORMER_THRESHOLD:
                            increase_pct = random.uniform(HIGH_PERFORMER_INCREASE_MIN, HIGH_PERFORMER_INCREASE_MAX)  # High performers
                        elif perf_rating >= GOOD_PERFORMER_THRESHOLD:
                            increase_pct = random.uniform(GOOD_PERFORMER_INCREASE_MIN, GOOD_PERFORMER_INCREASE_MAX)  # Good performers
                        elif perf_rating >= AVERAGE_PERFORMER_THRESHOLD:
                            increase_pct = random.uniform(AVERAGE_PERFORMER_INCREASE_MIN, AVERAGE_PERFORMER_INCREASE_MAX)  # Average performers
                        else:
                            increase_pct = random.uniform(LOW_PERFORMER_INCREASE_MIN, LOW_PERFORMER_INCREASE_MAX)   # Low performers
                    else:
                        # Default increase if no review
                        increase_pct = random.normalvariate(DEFAULT_INCREASE_MEAN, DEFAULT_INCREASE_STD)
                    
                    salary = int(salary * (1 + increase_pct))
                    reason = 'Annual Review' if not relevant_review or relevant_review['overall_rating'] >= AVERAGE_PERFORMER_THRESHOLD else 'Merit Increase'
                
            # Bonus target correlates with level and performance
            job_level = emp.get('job_level', 1)
            if job_level >= MIN_SENIOR_LEVEL:
                bonus_target = random.randint(SENIOR_LEVEL_BONUS_MIN, SENIOR_LEVEL_BONUS_MAX)
            elif job_level >= MIN_MANAGER_LEVEL:
                bonus_target = random.randint(MANAGER_LEVEL_BONUS_MIN, MANAGER_LEVEL_BONUS_MAX)
            else:
                bonus_target = random.randint(0, 15)
            
            # Equity for higher levels
            if job_level >= MIN_MANAGER_LEVEL:
                equity = random.randint(SENIOR_LEVEL_EQUITY_MIN, SENIOR_LEVEL_EQUITY_MAX) if job_level >= MIN_SENIOR_LEVEL else random.randint(MANAGER_LEVEL_EQUITY_MIN, MANAGER_LEVEL_EQUITY_MAX)
            else:
                equity = 0
            
            compensation_history.append({
                'comp_id': f'COMP{random.randint(10000, 99999)}',
                'employee_id': emp['employee_id'],
                'effective_date': effective_date,
                'base_salary': int(salary),
                'bonus_target_pct': int(bonus_target),
                'equity_value': int(equity),
                'adjustment_reason': reason
            })
    
    print("✅ Generated compensation data")
    return compensation_history


# COMMAND ----------

# ============================================================================
# HELPER FUNCTIONS FOR ENSURING COMPLETE DATA
# ============================================================================

# COMMAND ----------

def generate_alex_employee_record():
    """
    Generate Alex Smith's employee record with specific characteristics for demo.
    Returns a dictionary with Alex's employee data.
    """
    alex_id = 100038
    alex_hire_date = date.today() - timedelta(days=540)  # 18 months ago
    alex_job_start = alex_hire_date + timedelta(days=30)
    
    return {
        'employee_id': str(alex_id),
        'person_id': f'PER{alex_id + 50000}',
        'first_name': 'Alex',
        'last_name': 'Smith',
        'gender': 'F',
        'age': 32,
        'hire_date': alex_hire_date,
        'current_job_start_date': alex_job_start,
        'department': 1034, #'Engineering',
        'department_name': DEPARTMENT_CODE_TO_NAME.get(1034, 'Unknown'),
        'job_title': 'Software Engineer',
        'job_level': 1,
        'location': 43,  # UK location code (for demo purposes)
        'location_name': LOCATION_CODE_TO_NAME.get(43, 'Australia'),
        'employment_status': 'Active',
        'employment_type': 'Full-time',
        'base_salary': 110000,  # AUD - Sydney Software Engineer rate
        'tenure_months': 18,
        'months_in_current_role': 17,
        'manager_id': None
    }


# COMMAND ----------

def ensure_alex_in_employees_df(employees_df):
    """
    Ensure Alex Smith's generated record is always in the employees DataFrame.
    If Alex exists, replace with generated version. If not, add it.
    
    Args:
        employees_df: DataFrame with employees data
        
    Returns:
        DataFrame with Alex's record guaranteed to be present
    """
    alex_record = generate_alex_employee_record()
    
    # Check if Alex exists in the DataFrame
    alex_exists = employees_df.filter(F.col("employee_id") == ALEX_EMPLOYEE_ID).count() > 0
    
    if alex_exists:
        # Remove existing Alex record and add generated one
        print(f"   🔄 Replacing existing Alex record with generated demo data...")
        employees_df = employees_df.filter(F.col("employee_id") != ALEX_EMPLOYEE_ID)
    else:
        print(f"   ➕ Adding Alex Smith demo record (employee_id: {ALEX_EMPLOYEE_ID})...")
    
    # Create Alex's DataFrame - include all fields that are in the final schema
    # Now includes hire_date and current_job_start_date columns
    alex_record_for_df = {
        'employee_id': str(alex_record['employee_id']),
        'person_id': str(alex_record['person_id']),
        'age': int(alex_record['age']),
        'gender': str(alex_record['gender']),
        'department': str(alex_record['department']),
        'department_name': str(alex_record.get('department_name', DEPARTMENT_CODE_TO_NAME.get(int(alex_record['department']) if alex_record['department'] is not None else None, 'Unknown'))),
        'job_title': str(alex_record['job_title']),
        'job_level': int(alex_record['job_level']),
        'location': str(alex_record['location']),
        'location_name': str(alex_record.get('location_name', LOCATION_CODE_TO_NAME.get(int(alex_record['location']) if alex_record['location'] is not None else None, 'Australia'))),
        'employment_type': str(alex_record['employment_type']),
        'base_salary': int(alex_record['base_salary']),
        'tenure_months': int(alex_record['tenure_months']),
        'months_in_current_role': int(alex_record['months_in_current_role']),
        'employment_status': str(alex_record['employment_status']),
        'first_name': str(alex_record['first_name']),
        'last_name': str(alex_record['last_name']),
        'hire_date': alex_record['hire_date'],
        'current_job_start_date': alex_record['current_job_start_date']
    }
    
    # Create DataFrame - explicit type casting ensures schema compatibility
    alex_df = spark.createDataFrame([alex_record_for_df]).select(
        F.col("employee_id").alias("employee_id"),
        F.col("person_id").alias("person_id"),
        F.col("age").cast("integer").alias("age"),
        F.col("gender").alias("gender"),
        F.col("department").alias("department"),
        F.col("department_name").alias("department_name"),
        F.col("job_title").alias("job_title"),
        F.col("job_level").cast("integer").alias("job_level"),
        F.col("location").alias("location"),
        F.col("location_name").alias("location_name"),
        F.col("employment_type").alias("employment_type"),
        F.col("base_salary").cast("integer").alias("base_salary"),
        F.col("tenure_months").cast("integer").alias("tenure_months"),
        F.col("months_in_current_role").cast("integer").alias("months_in_current_role"),
        F.col("employment_status").alias("employment_status"),
        F.col("first_name").alias("first_name"),
            F.col("last_name").alias("last_name"),
            F.col("hire_date").cast("date").alias("hire_date"),
            F.col("current_job_start_date").cast("date").alias("current_job_start_date")
        )
        
        # Add department_name column based on department code mapping
        # Handle both numeric codes and string codes (including 'null')
        dept_mapping_expr = F.lit("Unknown")
        for dept_code, dept_name in DEPARTMENT_CODE_TO_NAME.items():
            if dept_code is None or dept_code == 'null' or dept_code == '':
                # Handle null/empty values
                dept_mapping_expr = F.when(
                    (F.col("department").isNull()) | (F.col("department") == 'null') | (F.col("department") == ''),
                    F.lit(dept_name)
                ).otherwise(dept_mapping_expr)
            else:
                # Handle numeric codes - compare as both int and string to handle type variations
                dept_mapping_expr = F.when(
                    (F.col("department") == dept_code) | (F.col("department").cast("string") == str(dept_code)),
                    F.lit(dept_name)
                ).otherwise(dept_mapping_expr)
        employees_df = employees_df.withColumn("department_name", dept_mapping_expr)
        
        # Add location_name column based on location code mapping
        loc_mapping_expr = F.lit("Australia")  # Default to Australia
        for loc_code, loc_name in LOCATION_CODE_TO_NAME.items():
            if loc_code is None or loc_code == 'null' or loc_code == '':
                loc_mapping_expr = F.when(
                    (F.col("location").isNull()) | (F.col("location") == 'null') | (F.col("location") == ''),
                    F.lit(loc_name)
                ).otherwise(loc_mapping_expr)
            else:
                loc_mapping_expr = F.when(
                    (F.col("location") == loc_code) | (F.col("location").cast("string") == str(loc_code)),
                    F.lit(loc_name)
                ).otherwise(loc_mapping_expr)
        employees_df = employees_df.withColumn("location_name", loc_mapping_expr)
    
    # Union Alex's record with the rest
    employees_df = employees_df.unionByName(alex_df)
    print(f"   ✅ Alex Smith demo record included")
    
    return employees_df

# COMMAND ----------

def ensure_all_employees_have_performance_records(performance_df, employees_df, employees_list=None):
    """
    Ensure every employee has at least one performance record.
    Generates missing records for employees not in performance_df.
    
    Args:
        performance_df: DataFrame with performance records (may be incomplete)
        employees_df: DataFrame with all employees
        employees_list: Optional list of employee dicts (for generation)
        
    Returns:
        DataFrame with performance records for all employees
    """
    # Get list of employees that have performance records
    employees_with_perf = performance_df.select("employee_id").distinct()
    
    # Get list of all employees
    all_employees = employees_df.select("employee_id").distinct()
    
    # Find missing employees
    missing_employees_df = all_employees.join(employees_with_perf, on="employee_id", how="left_anti")
    missing_exists = missing_employees_df.limit(1).count() > 0
    
    if missing_exists:
        print("   ⚠️ Found employees without performance records. Generating missing records...")
        
        # Collect missing employee IDs
        missing_employee_ids = [row['employee_id'] for row in missing_employees_df.toLocalIterator()]
        
        # Get employee data directly from DataFrame for missing employees
        missing_employees_df_full = employees_df.filter(F.col("employee_id").isin(missing_employee_ids))
        
        # Generate at least one performance review for each missing employee
        generated_reviews = []
        for row in missing_employees_df_full.toLocalIterator():
            # Calculate review_date from tenure_months
            # Spark Row objects don't have .get() method, use direct access with try-except
            try:
                tenure_months = row['tenure_months']
                if tenure_months is None or (isinstance(tenure_months, (int, float)) and tenure_months <= 0):
                    tenure_months = 0
            except (KeyError, IndexError):
                tenure_months = 0
            
            if isinstance(tenure_months, (int, float)) and tenure_months > 0:
                review_date = date.today() - timedelta(days=int(tenure_months * 30))
            else:
                review_date = date.today() - timedelta(days=365)
            
            if review_date > date.today():
                review_date = date.today() - timedelta(days=365)
            
            # Add randomization for missing employee's performance review
            overall_rating = round(random.uniform(2.5, 4.8), 2)
            goals_achievement = random.randint(75, 110)
            competency_rating = round(random.uniform(2.5, 4.8), 2)
            # Optionally, ensure ratings closer to DEFAULT_RATING, but variable:
            if random.random() < 0.15:  # 15% chance to be 'outlier'
                overall_rating = round(random.uniform(1.2, 5.0), 2)
                competency_rating = round(random.uniform(1.2, 5.0), 2)
                goals_achievement = random.randint(60, 115)
            generated_reviews.append({
                'review_id': f'REV{random.randint(10000, 99999)}',
                'employee_id': row['employee_id'],
                'review_period': int(review_date.year),
                'review_date': review_date,
                'overall_rating': float(overall_rating),
                'goals_achievement': int(goals_achievement),
                'competency_rating': float(competency_rating),
                'reviewer_id': f'EMP{random.randint(100001, 199999)}',
                'status': COMPLETION_STATUS_COMPLETED
            })
            
        if generated_reviews:
            # Convert to DataFrame and union with existing
            missing_reviews_df = spark.createDataFrame(generated_reviews).select(
                F.col("employee_id").alias("employee_id"),
                F.col("review_date").alias("review_date"),
                F.col("overall_rating").cast("double").alias("overall_rating"),
                F.col("competency_rating").cast("double").alias("competency_rating"),
                F.col("goals_achievement").cast("integer").alias("goals_achievement"),
                F.col("review_id").cast("string").alias("review_id"),
                F.col("review_period").cast("integer").alias("review_period"),
                F.col("reviewer_id").alias("reviewer_id"),
                F.col("status").alias("status")
            )
            
            performance_df = performance_df.unionByName(missing_reviews_df)
            print("   ✅ Generated performance records for missing employees")
    
    return performance_df


# COMMAND ----------

def ensure_all_employees_have_learning_records(learning_df, employees_df, employees_list=None, performance_reviews_list=None):
    """
    Ensure every employee has at least one learning record.
    Generates missing records for employees not in learning_df.
    
    Args:
        learning_df: DataFrame with learning records (may be incomplete)
        employees_df: DataFrame with all employees
        employees_list: Optional list of employee dicts (for generation)
        performance_reviews_list: Optional list of performance reviews (for generation)
        
    Returns:
        DataFrame with learning records for all employees
    """
    # Get list of employees that have learning records
    employees_with_learning = learning_df.select("employee_id").distinct()
    
    # Get list of all employees
    all_employees = employees_df.select("employee_id").distinct()
    
    # Find missing employees
    missing_employees_df = all_employees.join(employees_with_learning, on="employee_id", how="left_anti")
    missing_exists = missing_employees_df.limit(1).count() > 0
    
    if missing_exists:
        print("   ⚠️ Found employees without learning records. Generating missing records...")
        
        # Collect missing employee IDs
        missing_employee_ids = [row['employee_id'] for row in missing_employees_df.toLocalIterator()]
        
        # Get employee data directly from DataFrame for missing employees
        missing_employees_df_full = employees_df.filter(F.col("employee_id").isin(missing_employee_ids))
        
        # Generate at least one learning record for each missing employee
        generated_learning = []
        for row in missing_employees_df_full.toLocalIterator():
            # Calculate completion_date from tenure_months
            # Spark Row objects don't have .get() method, use direct access with try-except
            try:
                tenure_months = row['tenure_months']
                if tenure_months is None or (isinstance(tenure_months, (int, float)) and tenure_months <= 0):
                    tenure_months = 0
            except (KeyError, IndexError):
                tenure_months = 0
            
            if isinstance(tenure_months, (int, float)) and tenure_months > 0:
                completion_date = date.today() - timedelta(days=int(min(tenure_months * 30, 180)))
            else:
                completion_date = date.today() - timedelta(days=180)
            
            if completion_date > date.today():
                completion_date = date.today() - timedelta(days=180)
            
            generated_learning.append({
                'learning_id': f'LRN{random.randint(10000, 99999)}',
                'employee_id': row['employee_id'],
                'course_title': 'Onboarding Training',
                'category': 'Technical Skills',
                'completion_date': completion_date,
                'hours_completed': int(random.randint(2, 8)),
                'completion_status': COMPLETION_STATUS_COMPLETED,
                'score': int(random.randint(70, 90))
            })
            
        if generated_learning:
            # Convert to DataFrame and union with existing
            missing_learning_df = spark.createDataFrame(generated_learning).select(
                F.col("learning_id").cast("string").alias("learning_id"),
                F.col("employee_id").cast("string").alias("employee_id"),
                F.col("course_title").cast("string").alias("course_title"),
                F.col("category").cast("string").alias("category"),
                F.col("completion_date").cast("date").alias("completion_date"),
                F.col("hours_completed").cast("integer").alias("hours_completed"),
                F.col("completion_status").cast("string").alias("completion_status"),
                F.col("score").cast("integer").alias("score")
            )
            
            learning_df = learning_df.unionByName(missing_learning_df)
            print("   ✅ Generated learning records for missing employees")
    
    return learning_df


# COMMAND ----------

# ============================================================================
# DATA PRODUCT LOADING FUNCTIONS
# ============================================================================

# COMMAND ----------

def load_employees_from_data_product(generated_employees=None):
    """
    Load employees from SAP SuccessFactors Data Product.
    Falls back to generated data if data product load fails.
    
    Args:
        generated_employees: Optional list of generated employee dicts for fallback
        
    Returns:
        Spark DataFrame with employees data (from DP or generated)
    """
    print("📊 Loading employees from SAP SuccessFactors Data Product...")
    print(f"   Source: {EMPLOYEES_DATA_PRODUCT_TABLE}")
    try:
        employees_df_raw = spark.sql(
            f"SELECT * FROM {EMPLOYEES_DATA_PRODUCT_TABLE}"
        )
        
        
        print("✅ Successfully loaded employees from DATA PRODUCT")
        print(f"   📦 Data Source: SAP SuccessFactors Data Product (Delta Sharing)")
        
        # Deduplicate: Keep only the latest record per employee_id based on startDate
        # Handle case where startDate might be null or have different column name variations
        start_date_col = None
        for col_name in START_DATE_COLUMN_VARIANTS:
            if col_name in employees_df_raw.columns:
                start_date_col = col_name
                break
        
        if start_date_col:
            print(f"   🔍 Deduplicating by employee_id using {start_date_col} (keeping latest record per employee)...")
            # Optimize window function: repartition by userId to ensure proper data distribution
            # This reduces shuffling during window operation and improves performance
            default_partitions = int(spark.conf.get("spark.sql.shuffle.partitions", "200"))
            num_partitions = max(1, min(200, default_partitions))
            employees_df_raw = employees_df_raw.repartition(num_partitions, "userId")
            
            # Use window function to rank records by startDate, keeping latest (descending order)
            # Handle NULL dates by putting them last (using coalesce with a very old date)
            # Partitioning by userId ensures all records for same user are on same partition
            # This eliminates cross-partition shuffling during window operation
            window_spec = Window.partitionBy("userId").orderBy(
                F.coalesce(F.col(start_date_col), F.lit("1900-01-01").cast("date")).desc()
            )
            employees_df_raw = employees_df_raw.withColumn("row_num", F.row_number().over(window_spec)) \
                                               .filter(F.col("row_num") == LATEST_RECORD_ROW_NUM) \
                                               .drop("row_num")
            
            # Coalesce after deduplication to reduce partition count (we have fewer rows now)
            # This improves performance for subsequent operations
            employees_df_raw = employees_df_raw.coalesce(max(1, num_partitions // 2))
            
            print("   ✅ Deduplicated employee records (kept latest per employee)")
        else:
            print(f"   ⚠️ Warning: Could not find startDate column. Available columns: {', '.join(employees_df_raw.columns[:10])}...")
            # Fallback: use any date column or just take first record per employee_id
            print(f"   → Using simple deduplication (keeping first record per employee_id)")
            employees_df_raw = employees_df_raw.dropDuplicates(["userId"])
            print("   ✅ Deduplicated employee records using simple dropDuplicates")
        
        # Map columns to expected names (same as notebook 03)
        # Use try_cast to handle malformed values (e.g., '<10' in age field)
        # Generate person_id: use personId if it exists and is not null, otherwise generate from userId
        # Format: PER{employee_id_number + 50000} to match generated data pattern
        has_person_id_col = "personId" in employees_df_raw.columns
        
        # Check which columns exist before referencing them
        available_cols = set(employees_df_raw.columns)
        has_age = "age" in available_cols
        has_gender = "gender" in available_cols
        has_department = "department" in available_cols
        has_jobTitle = "jobTitle" in available_cols
        has_location = "location" in available_cols
        has_employmentType = "employmentType" in available_cols
        has_annualSalary = "annualSalary" in available_cols
        has_minimumPay = "minimumPay" in available_cols
        has_maximumPay = "maximumPay" in available_cols
        has_totalOrgTenureCalc = "totalOrgTenureCalc" in available_cols
        has_totalPositionTenureCalc = "totalPositionTenureCalc" in available_cols
        has_employmentStatus = "employmentStatus" in available_cols
        
        if has_person_id_col:
            person_id_expr = F.coalesce(
                F.col("personId"),
                F.when(
                    F.expr("try_cast(regexp_replace(userId, '[^0-9]', '') as int)").isNotNull(),
                    F.concat(
                        F.lit("PER"),
                        (F.expr("try_cast(regexp_replace(userId, '[^0-9]', '') as int)") + 50000).cast("string")
                    )
                ).otherwise(
                    F.concat(F.lit("PER"), F.abs(F.hash(F.col("userId"))).cast("string"))
                )
            )
        else:
            # Generate person_id from userId when personId column doesn't exist
            person_id_expr = F.when(
                F.expr("try_cast(regexp_replace(userId, '[^0-9]', '') as int)").isNotNull(),
                F.concat(
                    F.lit("PER"),
                    (F.expr("try_cast(regexp_replace(userId, '[^0-9]', '') as int)") + 50000).cast("string")
                )
            ).otherwise(
                F.concat(F.lit("PER"), F.abs(F.hash(F.col("userId"))).cast("string"))
            )
        
        # Build select expressions, only referencing columns that exist
        select_exprs = [
            F.col("userId").alias("employee_id"),
            person_id_expr.alias("person_id"),
        ]
        
        # Age - only cast if column exists
        if has_age:
            select_exprs.append(F.coalesce(F.expr("try_cast(age as int)"), F.lit(DEFAULT_AGE)).alias("age"))
        else:
            select_exprs.append(F.lit(DEFAULT_AGE).alias("age"))
        
        # Gender
        if has_gender:
            select_exprs.append(F.col("gender").alias("gender"))
        else:
            select_exprs.append(F.lit("Unknown").alias("gender"))
        
        # Department
        if has_department:
            select_exprs.append(F.col("department").alias("department"))
        else:
            select_exprs.append(F.lit("Unknown").alias("department"))
        
        # Job Title and Job Level
        if has_jobTitle:
            select_exprs.append(F.col("jobTitle").alias("job_title"))
            select_exprs.append(
                F.when(F.col("jobTitle").rlike(JOB_TITLE_PATTERN_SENIOR), 3)
                 .when(F.col("jobTitle").rlike(JOB_TITLE_PATTERN_MID), 2)
                 .otherwise(1).alias("job_level")
            )
        else:
            select_exprs.append(F.lit("Unknown").alias("job_title"))
            select_exprs.append(F.lit(1).alias("job_level"))
        
        # Location
        if has_location:
            select_exprs.append(F.col("location").alias("location"))
        else:
            select_exprs.append(F.lit(0).alias("location"))  # Default to 0 (Australia) if missing
        
        # Employment Type - only cast if column exists
        if has_employmentType:
            select_exprs.append(
                F.when(F.expr("try_cast(employmentType as int)") == 3631, "Full-time")
                 .when(F.expr("try_cast(employmentType as int)") == 3637, "Part-time")
                 .when(F.expr("try_cast(employmentType as int)") == 3638, "Contract")
                 .when(F.col("employmentType").isNull(), DEFAULT_EMPLOYMENT_TYPE)
                 .otherwise(DEFAULT_EMPLOYMENT_TYPE).alias("employment_type")
            )
        else:
            select_exprs.append(F.lit(DEFAULT_EMPLOYMENT_TYPE).alias("employment_type"))
        
        # Base Salary
        if has_annualSalary:
            if has_minimumPay and has_maximumPay:
                select_exprs.append(
                    F.when(
                        F.col("annualSalary").isNotNull(),
                        F.expr("try_cast(annualSalary as int)")
                    ).when(
                        (F.col("minimumPay").isNotNull()) & (F.col("maximumPay").isNotNull()),
                        F.expr("cast(floor(rand() * (try_cast(maximumPay as int) - try_cast(minimumPay as int) + 1)) + try_cast(minimumPay as int) as int)")
                    ).otherwise(F.lit(DEFAULT_SALARY)).alias("base_salary")
                )
            else:
                select_exprs.append(
                    F.when(
                        F.col("annualSalary").isNotNull(),
                        F.expr("try_cast(annualSalary as int)")
                    ).otherwise(F.lit(DEFAULT_SALARY)).alias("base_salary")
                )
        elif has_minimumPay and has_maximumPay:
            select_exprs.append(
                F.when(
                    (F.col("minimumPay").isNotNull()) & (F.col("maximumPay").isNotNull()),
                    F.expr("cast(floor(rand() * (try_cast(maximumPay as int) - try_cast(minimumPay as int) + 1)) + try_cast(minimumPay as int) as int)")
                ).otherwise(F.lit(DEFAULT_SALARY)).alias("base_salary")
            )
        else:
            select_exprs.append(F.lit(DEFAULT_SALARY).alias("base_salary"))
        
        # Tenure Months - only cast if column exists
        if has_totalOrgTenureCalc:
            select_exprs.append(
                F.coalesce(F.expr("try_cast(totalOrgTenureCalc as int)"), F.lit(DEFAULT_TENURE_MONTHS)).alias("tenure_months")
            )
        else:
            select_exprs.append(F.lit(DEFAULT_TENURE_MONTHS).alias("tenure_months"))
        
        # Months in Current Role - only cast if column exists
        if has_totalPositionTenureCalc:
            select_exprs.append(
                F.coalesce(F.expr("try_cast(totalPositionTenureCalc as int)"), F.lit(DEFAULT_TENURE_MONTHS)).alias("months_in_current_role")
            )
        else:
            select_exprs.append(F.lit(DEFAULT_TENURE_MONTHS).alias("months_in_current_role"))
        
        # Employment Status
        if has_employmentStatus:
            select_exprs.append(
                F.when(F.upper(F.trim(F.col("employmentStatus"))).isin(['A', 'ACTIVE', 'ACT']), 'Active')
                 .otherwise(F.col("employmentStatus")).alias("employment_status")
            )
        else:
            select_exprs.append(F.lit("Active").alias("employment_status"))
        
        # First and Last Name (will be generated later)
        select_exprs.extend([
            F.lit("Unknown").alias("first_name"),
            F.lit("Unknown").alias("last_name")
        ])
        
        employees_df = employees_df_raw.select(*select_exprs)
        
        # Generate deterministic names using Spark SQL functions to preserve lineage
        # Use hash-based selection from predefined name lists - all Spark operations, no collect()
        # This approach maintains lineage from data product source
        
        # Predefined name lists (common first and last names)
        # Embedded in UDF functions for Spark Connect compatibility (no broadcast variables)
        first_names_list = ['Alex', 'Sarah', 'Michael', 'Jessica', 'David', 'Emily', 'Chris', 'Amanda', 
                           'Ryan', 'Lisa', 'John', 'Maria', 'James', 'Jennifer', 'Robert', 'Emma', 
                           'Olivia', 'Charlotte', 'Sophia', 'Isabella', 'Noah', 'Oliver', 'William', 
                           'Lucas', 'Benjamin', 'Mia', 'Harper', 'Evelyn', 'Abigail', 'Elijah', 'Mason', 
                           'Alexander', 'Daniel', 'Matthew', 'Aiden', 'Sophie', 'Grace', 'Lily', 'Chloe']
        
        last_names_list = ['Smith', 'Jones', 'Williams', 'Brown', 'Wilson', 'Taylor', 'Anderson', 'Thomas', 
                          'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 
                          'Robinson', 'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 
                          'Young', 'King', 'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 
                          'Nelson', 'Carter', 'Mitchell', 'Roberts', 'Turner', 'Phillips', 'Campbell', 
                          'Parker', 'Evans', 'Edwards', 'Collins', 'Stewart', 'Sanchez', 'Morris']
        
        # Define UDFs that use hash-based selection from name lists
        # Name lists are embedded in closure for Spark Connect compatibility
        # This preserves lineage as all operations stay in Spark execution plan
        def get_first_name(emp_id):
            """Get deterministic first name based on employee_id hash"""
            if emp_id is None:
                return "Unknown"
            names = first_names_list  # Embedded in closure
            if isinstance(emp_id, str):
                idx = hash(emp_id) % len(names)
            else:
                idx = int(emp_id) % len(names)
            return names[abs(idx)]
        
        def get_last_name(emp_id):
            """Get deterministic last name based on employee_id hash"""
            if emp_id is None:
                return "Unknown"
            names = last_names_list  # Embedded in closure
            if isinstance(emp_id, str):
                idx = (hash(emp_id) + 1000) % len(names)
            else:
                idx = (int(emp_id) + 1000) % len(names)
            return names[abs(idx)]
        
        # Register UDFs - executed on executors, preserves lineage
        # Name lists are serialized with UDF functions (Spark Connect compatible)
        get_first_name_udf = F.udf(get_first_name, StringType())
        get_last_name_udf = F.udf(get_last_name, StringType())
        
        # Generate names directly in Spark DataFrame operations - preserves lineage
        employees_df = employees_df.withColumn(
            "first_name",
            get_first_name_udf(F.col("employee_id"))
        ).withColumn(
            "last_name",
            get_last_name_udf(F.col("employee_id"))
        )
        
        # Add department_name column based on department code mapping
        # Create a mapping expression using when/otherwise
        # Handle both numeric codes and string codes (including 'null')
        dept_mapping_expr = F.lit("Unknown")
        for dept_code, dept_name in DEPARTMENT_CODE_TO_NAME.items():
            if dept_code is None or dept_code == 'null' or dept_code == '':
                # Handle null/empty values
                dept_mapping_expr = F.when(
                    (F.col("department").isNull()) | (F.col("department") == 'null') | (F.col("department") == ''),
                    F.lit(dept_name)
                ).otherwise(dept_mapping_expr)
            else:
                # Handle numeric codes - compare as both int and string to handle type variations
                dept_mapping_expr = F.when(
                    (F.col("department") == dept_code) | (F.col("department").cast("string") == str(dept_code)),
                    F.lit(dept_name)
                ).otherwise(dept_mapping_expr)
        
        employees_df = employees_df.withColumn("department_name", dept_mapping_expr)
        
        # Add location_name column based on location code mapping
        loc_mapping_expr = F.lit("Australia")  # Default to Australia
        for loc_code, loc_name in LOCATION_CODE_TO_NAME.items():
            if loc_code is None or loc_code == 'null' or loc_code == '':
                loc_mapping_expr = F.when(
                    (F.col("location").isNull()) | (F.col("location") == 'null') | (F.col("location") == ''),
                    F.lit(loc_name)
                ).otherwise(loc_mapping_expr)
            else:
                loc_mapping_expr = F.when(
                    (F.col("location") == loc_code) | (F.col("location").cast("string") == str(loc_code)),
                    F.lit(loc_name)
                ).otherwise(loc_mapping_expr)
        employees_df = employees_df.withColumn("location_name", loc_mapping_expr)
        
        # Add computed date columns needed for goals and compensation generation
        # This eliminates the need for _prepare_employees_df_for_generation()
        employees_df = employees_df.withColumn(
            "hire_date",
            F.date_sub(F.current_date(), F.col("tenure_months") * 30)
        ).withColumn(
            "current_job_start_date",
            F.date_sub(F.current_date(), F.col("months_in_current_role") * 30)
        )
        
        print("✅ Transformed employees data")
        
        # Always ensure Alex's generated demo record is included
        employees_df = ensure_alex_in_employees_df(employees_df)
        if employees_df is not None:
            print("   ✅ Ensured Alex demo record is present")
        
        print(f"   📊 Final Status: Using DATA PRODUCT data (with Alex demo record)")
        return employees_df, 'DATA PRODUCT'  # Return source indicator
        
    except Exception as e:
        print(f"⚠️ Error loading employees from data product: {e}")
        print(f"   🔄 FALLBACK: Switching to generated employees data...")
        print(f"   📦 Data Source: Generated (simulated)")
        
        # Fallback to generated data - generate only when needed
        if generated_employees is None:
            print("   → Generating employees data (on-demand)...")
            generated_employees = generate_employees()
        
        # Convert generated data to DataFrame with correct schema
        employees_df_generated = spark.createDataFrame(generated_employees)
        employees_df = employees_df_generated.select(
            F.col("employee_id").alias("employee_id"),
            F.col("person_id").alias("person_id"),
            F.col("age").cast("integer").alias("age"),
            F.col("gender").alias("gender"),
            F.col("department").alias("department"),
            F.col("job_title").alias("job_title"),
            F.col("job_level").cast("integer").alias("job_level"),
            F.col("location").alias("location"),
            F.col("employment_type").alias("employment_type"),
            F.col("base_salary").cast("integer").alias("base_salary"),
            F.col("tenure_months").cast("integer").alias("tenure_months"),
            F.col("months_in_current_role").cast("integer").alias("months_in_current_role"),
            F.col("employment_status").alias("employment_status"),
            F.col("first_name").alias("first_name"),
            F.col("last_name").alias("last_name")
        )
        
        # Add department_name column based on department code mapping
        # Handle both numeric codes and string codes (including 'null')
        dept_mapping_expr = F.lit("Unknown")
        for dept_code, dept_name in DEPARTMENT_CODE_TO_NAME.items():
            if dept_code is None or dept_code == 'null' or dept_code == '':
                # Handle null/empty values
                dept_mapping_expr = F.when(
                    (F.col("department").isNull()) | (F.col("department") == 'null') | (F.col("department") == ''),
                    F.lit(dept_name)
                ).otherwise(dept_mapping_expr)
            else:
                # Handle numeric codes - compare as both int and string to handle type variations
                dept_mapping_expr = F.when(
                    (F.col("department") == dept_code) | (F.col("department").cast("string") == str(dept_code)),
                    F.lit(dept_name)
                ).otherwise(dept_mapping_expr)
        employees_df = employees_df.withColumn("department_name", dept_mapping_expr)
        
        # Add location_name column based on location code mapping
        loc_mapping_expr = F.lit("Australia")  # Default to Australia
        for loc_code, loc_name in LOCATION_CODE_TO_NAME.items():
            if loc_code is None or loc_code == 'null' or loc_code == '':
                loc_mapping_expr = F.when(
                    (F.col("location").isNull()) | (F.col("location") == 'null') | (F.col("location") == ''),
                    F.lit(loc_name)
                ).otherwise(loc_mapping_expr)
            else:
                loc_mapping_expr = F.when(
                    (F.col("location") == loc_code) | (F.col("location").cast("string") == str(loc_code)),
                    F.lit(loc_name)
                ).otherwise(loc_mapping_expr)
        employees_df = employees_df.withColumn("location_name", loc_mapping_expr)
        
        # Add computed date columns needed for goals and compensation generation
        # This eliminates the need for _prepare_employees_df_for_generation()
        employees_df = employees_df.withColumn(
            "hire_date",
            F.date_sub(F.current_date(), F.col("tenure_months") * 30)
        ).withColumn(
            "current_job_start_date",
            F.date_sub(F.current_date(), F.col("months_in_current_role") * 30)
        )
        
        # Always ensure Alex's generated demo record is included
        employees_df = ensure_alex_in_employees_df(employees_df)
        print("✅ Using generated employees data")
        print(f"   📊 Final Status: Using GENERATED data (fallback, with Alex demo record)")
        return employees_df, 'GENERATED'  # Return source indicator


# COMMAND ----------

def load_performance_from_data_product(generated_performance_reviews=None, employees=None, employees_df=None):
    """
    Load performance reviews from SAP SuccessFactors Data Product.
    Falls back to generated data if data product load fails.
    
    Args:
        generated_performance_reviews: Optional list of generated performance review dicts for fallback
        employees: Optional list of employee dicts (needed if generating fallback)
        
    Returns:
        Spark DataFrame with performance reviews data (from DP or generated)
    """
    print("📊 Loading performance reviews from SAP SuccessFactors Data Product...")
    print(f"   Source: {PERFORMANCE_DATA_PRODUCT_TABLE}")
    try:
        performance_df_raw = spark.sql(
            f"SELECT * FROM {PERFORMANCE_DATA_PRODUCT_TABLE}"
        )
        
        
        print("✅ Successfully loaded performance records from DATA PRODUCT")
        print(f"   📦 Data Source: SAP SuccessFactors Data Product (Delta Sharing)")
        print("   ℹ️ Keeping all performance records per employee (no deduplication)")
        
        # Map columns to expected names
        # Use try_cast to handle malformed values
        performance_df = performance_df_raw.select(
            F.col("userId").alias("employee_id"),
            F.col("reviewPeriodEndDt").alias("review_date"),
            F.coalesce(F.expr("try_cast(currentPerformanceRating as double)"), F.lit(DEFAULT_RATING)).alias("overall_rating"),  # Default to 3.0 if invalid
            F.coalesce(F.expr("try_cast(normalizedCurrentPerformanceRating as double)"), F.lit(DEFAULT_RATING)).alias("competency_rating"),  # Default to 3.0 if invalid
            F.coalesce(F.expr("try_cast(normalizedCurrentPerformanceRating * 20 as int)"), F.lit(DEFAULT_GOALS_ACHIEVEMENT)).alias("goals_achievement"),  # Default to 60 if invalid
            F.col("feedbackId").cast("string").alias("review_id"),
            F.coalesce(F.expr("try_cast(year(reviewPeriodStartDt) as int)"), F.lit(DEFAULT_REVIEW_YEAR)).alias("review_period"),  # Default to 2024 if invalid
            F.lit(None).cast("string").alias("reviewer_id"),
            F.lit(COMPLETION_STATUS_COMPLETED).alias("status")
        ).filter(F.col("currentPerformanceRating").isNotNull() & (F.col("currentPerformanceRating") > 0))
        
        
        print("✅ Transformed performance data")
        
        # Ensure all employees have at least one performance record
        if employees_df is not None:
            performance_df = ensure_all_employees_have_performance_records(
                performance_df, employees_df, employees
            )
            print("   ✅ Ensured all employees have performance records")
        
        print(f"   📊 Final Status: Using DATA PRODUCT data")
        return performance_df, 'DATA PRODUCT'  # Return source indicator
        
    except Exception as e:
        print(f"⚠️ Error loading performance from data product: {e}")
        print(f"   🔄 FALLBACK: Switching to generated performance reviews data...")
        print(f"   📦 Data Source: Generated (simulated)")
        
        # Fallback to generated data - generate only when needed
        if generated_performance_reviews is None:
            # Collect employees list only if needed (lazy collection)
            if employees is None or len(employees) == 0:
                if employees_df is not None:
                    print("   → Collecting employees list from DataFrame for performance reviews generation...")
                    employees = _collect_employees_list_if_needed(employees_df)
                else:
                    print("   → Generating employees and performance reviews data (on-demand)...")
                    employees = generate_employees()
                    print("   ✅ Generated employees data for fallback")
            
            if employees is None or len(employees) == 0:
                print("   → Generating employees and performance reviews data (on-demand)...")
                employees = generate_employees()
                print("   ✅ Generated employees data for fallback")
            else:
                print("   → Generating performance reviews data (on-demand) for employees...")
                # Debug: show sample employee structure
                if employees:
                    sample = employees[0]
                    print(f"   ℹ️ Sample employee keys: {list(sample.keys())}")
                    print(f"   ℹ️ Sample employee: status='{sample.get('employment_status')}', tenure={sample.get('tenure_months')}, job_level={sample.get('job_level')}")
            generated_performance_reviews = generate_performance_reviews(employees)
        
        # Convert generated data to DataFrame with correct schema
        performance_df_generated = spark.createDataFrame(generated_performance_reviews)
        performance_df = performance_df_generated.select(
            F.col("employee_id").alias("employee_id"),
            F.col("review_date").alias("review_date"),
            F.col("overall_rating").cast("double").alias("overall_rating"),
            F.col("competency_rating").cast("double").alias("competency_rating"),
            F.col("goals_achievement").cast("integer").alias("goals_achievement"),
            F.col("review_id").cast("string").alias("review_id"),
            F.col("review_period").cast("integer").alias("review_period"),
            F.col("reviewer_id").alias("reviewer_id"),
            F.col("status").alias("status")
        )
        
        # Ensure all employees have at least one performance record
        if employees_df is not None:
            performance_df = ensure_all_employees_have_performance_records(
                performance_df, employees_df, employees
            )
            print("   ✅ Ensured all employees have performance records")
        
        print("✅ Using generated performance reviews data")
        print(f"   📊 Final Status: Using GENERATED data (fallback)")
        return performance_df, 'GENERATED'  # Return source indicator


# COMMAND ----------

def load_learning_from_data_product(generated_learning_records=None, employees_df=None, employees_list=None, performance_reviews_df=None, performance_reviews_list=None):
    """
    Load learning records from SAP SuccessFactors Data Product.
    Falls back to generated data if data product load fails.
    
    Args:
        generated_learning_records: Optional list of generated learning record dicts for fallback
        employees_df: Optional DataFrame of employees (preferred, avoids collection)
        employees_list: Optional list of employee dicts (fallback, will collect from DataFrame if needed)
        performance_reviews_df: Optional DataFrame of performance reviews (preferred, avoids collection)
        performance_reviews_list: Optional list of performance review dicts (fallback, will collect from DataFrame if needed)
        
    Returns:
        Spark DataFrame with learning records data (from DP or generated)
    """
    print("📊 Loading learning records from SAP SuccessFactors Data Product...")
    print(f"   Source: {LEARNING_DATA_PRODUCT_TABLE}")
    try:
        learning_df_raw = spark.sql(
            f"SELECT * FROM {LEARNING_DATA_PRODUCT_TABLE}"
        )
        
        
        print("✅ Successfully loaded learning records from DATA PRODUCT")
        print(f"   📦 Data Source: SAP SuccessFactors Data Product (Delta Sharing)")
        
        # Generate learningItemName from componentID if learningItemName is missing or null
        # Map componentID values to appropriate course names based on category patterns
        has_component_id = "componentID" in learning_df_raw.columns or "componentId" in learning_df_raw.columns
        component_id_col = "componentID" if "componentID" in learning_df_raw.columns else ("componentId" if "componentId" in learning_df_raw.columns else None)
        
        # Check if learningItemName column exists (may have different casing)
        has_learning_item_name = "learningItemName" in learning_df_raw.columns
        learning_item_name_col = "learningItemName" if has_learning_item_name else None
        
        # Check if completionStatus column exists (may have different casing)
        has_completion_status = "completionStatus" in learning_df_raw.columns or "completion_status" in learning_df_raw.columns
        completion_status_col = "completionStatus" if "completionStatus" in learning_df_raw.columns else ("completion_status" if "completion_status" in learning_df_raw.columns else None)
        
        # Build the learningItemName generation expression
        if has_component_id and component_id_col:
            # Map specific componentID values to course names
            name_generation_expr = (
                F.when(F.lower(F.col(component_id_col)) == "hr-601", "HR Policies and Procedures Training")
                 .when(F.lower(F.col(component_id_col)) == "harvardmm_3001", "Harvard Business Management Course")
                 .when(F.lower(F.col(component_id_col)).rlike("hr_302|hr_302_a"), "HR Compliance and Regulations")
                 .when(F.lower(F.col(component_id_col)) == "hr_0001", "HR Fundamentals")
                 .when(F.lower(F.col(component_id_col)) == "hr_300", "HR Advanced Practices")
                 .when(F.lower(F.col(component_id_col)) == "hr_301", "HR Management Skills")
                 .when(F.lower(F.col(component_id_col)) == "hr-344", "HR Policy Update")
                 .when(F.lower(F.col(component_id_col)).rlike("hr.*sop|hr-sop-disc|sa_hr_sop"), "HR Standard Operating Procedures")
                 .when(F.lower(F.col(component_id_col)).rlike("lead-dt01|lead-dt02"), "Leadership Development Program")
                 .when(F.lower(F.col(component_id_col)) == "glen_logbook1", "Professional Development Logbook")
                 .when(F.lower(F.col(component_id_col)) == "pm001", "Project Management Fundamentals")
                 .when(F.lower(F.col(component_id_col)) == "tech-dt01", "Technical Skills Development")
                 .when(F.lower(F.col(component_id_col)) == "t-454", "Technical Training Course T-454")
                 .when(F.lower(F.col(component_id_col)).rlike("sf_learning|sf_overview|mobile_sf_overview"), "SuccessFactors Platform Overview")
                 .when(F.lower(F.col(component_id_col)) == "coursera_2021", "Coursera Professional Development")
                 .when(F.lower(F.col(component_id_col)).rlike("han_safety|safety|confined_space|sunfun_confined_space"), "Safety and Compliance Training")
                 .when(F.lower(F.col(component_id_col)).rlike("flift|license"), "Forklift License Certification")
                 .when(F.lower(F.col(component_id_col)) == "rm", "Risk Management Fundamentals")
                 .when(F.lower(F.col(component_id_col)) == "rs-2", "Risk Management Advanced")
                 .when(F.lower(F.col(component_id_col)) == "vpol_ethics", "Ethics and Compliance Policy")
                 .when(F.lower(F.col(component_id_col)) == "opensesame_4005", "OpenSesame Learning Course")
                 .when(F.lower(F.col(component_id_col)).rlike("black_resp|black_respsourcing"), "Responsible Sourcing Training")
                 .when(F.lower(F.col(component_id_col)).rlike("sales.*ilt|sales_5000470028"), "Sales Training Program")
                 .when(F.lower(F.col(component_id_col)).rlike("mobile"), "Mobile Learning Module")
                 # For any other componentID, generate a generic course name
                 .otherwise(F.concat(F.lit("Learning Course - "), F.col(component_id_col)))
            )
            
            # Generate learningItemName: use existing value if column exists and has value, otherwise generate from componentID
            if has_learning_item_name and learning_item_name_col:
                # Column exists - use it if not null/empty, otherwise generate from componentID
                learning_df_raw = learning_df_raw.withColumn(
                    "learningItemName",
                    F.when(
                        (F.col(learning_item_name_col).isNotNull()) & (F.trim(F.col(learning_item_name_col)) != ""),
                        F.col(learning_item_name_col)
                    ).otherwise(name_generation_expr)
                )
            else:
                # Column doesn't exist - always generate from componentID
                learning_df_raw = learning_df_raw.withColumn("learningItemName", name_generation_expr)
        elif not has_learning_item_name:
            # No componentID and no learningItemName - use default
            learning_df_raw = learning_df_raw.withColumn("learningItemName", F.lit("Unknown Course"))
        
        # Check for column existence before referencing them
        available_cols = set(learning_df_raw.columns)
        
        # Check for userId column (required)
        has_user_id = "userId" in available_cols or "userID" in available_cols
        user_id_col = "userId" if "userId" in available_cols else ("userID" if "userID" in available_cols else None)
        if not has_user_id:
            raise ValueError("Required column 'userId' or 'userID' not found in learning data product")
        
        # Check for learningItemId
        has_learning_item_id = "learningItemId" in available_cols or "learningItemID" in available_cols
        learning_item_id_col = "learningItemId" if "learningItemId" in available_cols else ("learningItemID" if "learningItemID" in available_cols else None)
        
        # Check for completionDate
        has_completion_date = "completionDate" in available_cols or "completion_date" in available_cols
        completion_date_col = "completionDate" if "completionDate" in available_cols else ("completion_date" if "completion_date" in available_cols else None)
        
        # Check for hours columns (try multiple possible names)
        has_hours_completed = "hoursCompleted" in available_cols
        has_total_hours = "totalHours" in available_cols
        has_cpe_hours = "cpeHours" in available_cols
        hours_col = None
        if has_hours_completed:
            hours_col = "hoursCompleted"
        elif has_total_hours:
            hours_col = "totalHours"
        elif has_cpe_hours:
            hours_col = "cpeHours"
        
        # Check for score columns
        has_score = "score" in available_cols
        has_final_score = "finalScore" in available_cols
        score_col = "score" if has_score else ("finalScore" if has_final_score else None)
        
        # Check for completionStatusId
        has_completion_status_id = "completionStatusId" in available_cols or "completionStatusID" in available_cols
        completion_status_id_col = "completionStatusId" if "completionStatusId" in available_cols else ("completionStatusID" if "completionStatusID" in available_cols else None)
        
        # Map columns to expected names
        # Use try_cast to handle malformed values
        # Map completion status IDs to strings: "1"->Completed, "2"->In Progress, "3"->Not Started
        
        # Build completion status expression outside of select
        if has_completion_status_id and completion_status_id_col:
            completion_status_expr = (F.when(F.col(completion_status_id_col) == COMPLETION_STATUS_ID_COMPLETED, COMPLETION_STATUS_COMPLETED)
             .when(F.col(completion_status_id_col) == COMPLETION_STATUS_ID_IN_PROGRESS, COMPLETION_STATUS_IN_PROGRESS)
             .when(F.col(completion_status_id_col) == COMPLETION_STATUS_ID_NOT_STARTED, COMPLETION_STATUS_NOT_STARTED))
        else:
            # Start with default if completionStatusId doesn't exist
            completion_status_expr = F.lit(COMPLETION_STATUS_NOT_STARTED)
        
        # If completionStatus column exists, check for recognized codes
        if has_completion_status and completion_status_col:
            completion_status_expr = completion_status_expr.when(
                F.upper(F.trim(F.col(completion_status_col))).isin([
                    "COURSE-COMPL", "CERT-RECERT", "BRIEF-COMPL", "ONLINE-COMPL", 
                    "MOOC-CMPL", "COURSE-ATND", "DOC-READ", "TASK-C", 
                    "CERT-REINST", "CERT-COMPL", "COMPLETED", "IN PROGRESS", "NOT STARTED"
                ]), F.upper(F.trim(F.col(completion_status_col)))
            ).when(
                F.col(completion_status_col).isNotNull() & (F.trim(F.col(completion_status_col)) != ""), 
                F.col(completion_status_col)
            )
        
        completion_status_expr = completion_status_expr.otherwise(F.lit(COMPLETION_STATUS_NOT_STARTED)).alias("completion_status")
        
        # Build select expressions with proper column existence checks
        select_exprs = [
            F.col(user_id_col).alias("employee_id"),
        ]
        
        # learning_id
        if has_learning_item_id and learning_item_id_col:
            select_exprs.append(
                F.coalesce(F.col(learning_item_id_col), F.concat(F.lit("LRN"), F.abs(F.hash(F.col(user_id_col))).cast("string"))).alias("learning_id")
            )
        else:
            select_exprs.append(
                F.concat(F.lit("LRN"), F.abs(F.hash(F.col(user_id_col))).cast("string")).alias("learning_id")
            )
        
        # course_title
        select_exprs.append(
            F.coalesce(F.col("learningItemName"), F.lit("Unknown Course")).alias("course_title")
        )
        
        # category (based on learningItemName)
        select_exprs.append(
            F.when(F.lower(F.col("learningItemName")).rlike("technical|tech|programming|coding"), "Technical Skills")
             .when(F.lower(F.col("learningItemName")).rlike("leadership|manage|lead"), "Leadership")
             .when(F.lower(F.col("learningItemName")).rlike("communication|communicate|present"), "Communication")
             .when(F.lower(F.col("learningItemName")).rlike("project|pm|agile"), "Project Management")
             .when(F.lower(F.col("learningItemName")).rlike("data|analytics|analysis"), "Data Analysis")
             .when(F.lower(F.col("learningItemName")).rlike("product|pm"), "Product Management")
             .when(F.lower(F.col("learningItemName")).rlike("sales|sell"), "Sales Training")
             .when(F.lower(F.col("learningItemName")).rlike("compliance|legal|policy"), "Compliance")
             .otherwise("Technical Skills").alias("category")
        )
        
        # completion_date
        if has_completion_date and completion_date_col:
            select_exprs.append(
                F.coalesce(F.expr(f"try_cast({completion_date_col} as date)"), F.current_date()).alias("completion_date")
            )
        else:
            select_exprs.append(F.current_date().alias("completion_date"))
        
        # hours_completed
        if hours_col:
            select_exprs.append(
                F.coalesce(F.expr(f"try_cast({hours_col} as int)"), F.lit(0)).alias("hours_completed")
            )
        else:
            select_exprs.append(F.lit(0).alias("hours_completed"))
        
        # completion_status
        select_exprs.append(completion_status_expr)
        
        # score
        if score_col:
            select_exprs.append(
                F.coalesce(F.expr(f"try_cast({score_col} as int)"), F.lit(None).cast("int")).alias("score")
            )
        else:
            select_exprs.append(F.lit(None).cast("int").alias("score"))
        
        learning_df = learning_df_raw.select(*select_exprs).filter(F.col(user_id_col).isNotNull())
        
        
        print("✅ Transformed learning data")
        
        # Ensure all employees have at least one learning record
        if employees_df is not None:
            learning_df = ensure_all_employees_have_learning_records(
                learning_df, employees_df, employees_list, performance_reviews_list
            )
            print("   ✅ Ensured all employees have learning records")
        
        print(f"   📊 Final Status: Using DATA PRODUCT data")
        return learning_df, 'DATA PRODUCT'  # Return source indicator
        
    except Exception as e:
        print(f"⚠️ Error loading learning from data product: {e}")
        print(f"   🔄 FALLBACK: Switching to generated learning records data...")
        print(f"   📦 Data Source: Generated (simulated)")
        
        # Fallback to generated data - generate only when needed
        if generated_learning_records is None:
            # Collect employees list only if needed for generation (lazy collection)
            if employees_list is None:
                if employees_df is not None:
                    print("   → Collecting employees list from DataFrame for learning records generation...")
                    employees_list = _collect_employees_list_if_needed(employees_df)
                else:
                    print("   → Generating employees for learning records (on-demand)...")
                    employees_list = generate_employees()
                    print("   ✅ Generated employees data for learning fallback")
            
            if employees_list is None or len(employees_list) == 0:
                print("   → Generating employees for learning records (on-demand)...")
                employees_list = generate_employees()
                print("   ✅ Generated employees data for learning fallback")
            
            # Collect performance reviews list only if needed for generation (lazy collection)
            if performance_reviews_list is None:
                if performance_reviews_df is not None:
                    print("   → Collecting performance reviews list from DataFrame for learning records generation...")
                    try:
                        performance_reviews_list = [
                            {
                                'employee_id': row['employee_id'],
                                'review_date': row['review_date'],
                                'overall_rating': row['overall_rating'],
                                'competency_rating': row['competency_rating'],
                                'goals_achievement': row['goals_achievement'],
                                'review_id': row['review_id'],
                                'review_period': row['review_period'],
                                'reviewer_id': row['reviewer_id'],
                                'status': row['status']
                            }
                            for row in performance_reviews_df.toLocalIterator()  # Use toLocalIterator instead of collect()
                        ]
                    except Exception as e:
                        print(f"   ⚠️ Could not collect performance reviews: {e}")
                        performance_reviews_list = None
                else:
                    print("   → Generating performance reviews for learning records (on-demand)...")
                    performance_reviews_list = generate_performance_reviews(employees_list)
                    print("   ✅ Generated performance reviews for learning fallback")
            
            if performance_reviews_list is None or len(performance_reviews_list) == 0:
                print("   → Generating performance reviews for learning records (on-demand)...")
                performance_reviews_list = generate_performance_reviews(employees_list)
                print("   ✅ Generated performance reviews for learning fallback")
            
            print("   → Generating learning records data (on-demand) for employees...")
            generated_learning_records = generate_learning_records(employees_list, performance_reviews_list)
        
        # Convert generated data to DataFrame with correct schema
        learning_df_generated = spark.createDataFrame(generated_learning_records)
        learning_df = learning_df_generated.select(
            F.col("learning_id").cast("string").alias("learning_id"),
            F.col("employee_id").cast("string").alias("employee_id"),
            F.col("course_title").cast("string").alias("course_title"),
            F.col("category").cast("string").alias("category"),
            F.col("completion_date").cast("date").alias("completion_date"),
            F.col("hours_completed").cast("integer").alias("hours_completed"),
            F.col("completion_status").cast("string").alias("completion_status"),
            F.col("score").cast("integer").alias("score")
        )
        
        
        # Ensure all employees have at least one learning record
        if employees_df is not None:
            learning_df = ensure_all_employees_have_learning_records(
                learning_df, employees_df, employees_list, performance_reviews_list
            )
            print("   ✅ Ensured all employees have learning records")
        
        print("✅ Using generated learning records data")
        print(f"   📊 Final Status: Using GENERATED data (fallback)")
        return learning_df, 'GENERATED'  # Return source indicator


# ============================================================================
# MAIN DATA LOADING FUNCTION
# ============================================================================

# COMMAND ----------

def _prepare_employees_df_for_generation(employees_df):
    """
    Prepare employees DataFrame for use in generation functions.
    Adds computed date columns needed for goals and compensation generation.
    Returns DataFrame (not collected) to avoid driver memory issues.
    """
    try:
        # Calculate hire_date and current_job_start_date from tenure information
        # These are needed for goals and compensation generation
        employees_df_with_dates = employees_df.withColumn(
            "hire_date",
            F.date_sub(F.current_date(), F.col("tenure_months") * 30)
        ).withColumn(
            "current_job_start_date",
            F.date_sub(F.current_date(), F.col("months_in_current_role") * 30)
        )
        
        return employees_df_with_dates.select(
            'employee_id', 'age', 'gender', 'department', 'job_title', 'job_level',
            'location', 'employment_type', 'base_salary', 'tenure_months',
            'months_in_current_role', 'employment_status', 'first_name', 'last_name',
            'hire_date', 'current_job_start_date'
        )
    except Exception as e:
        print(f"   ⚠️ Warning: Could not prepare employees DataFrame: {e}")
        import traceback
        traceback.print_exc()
        return None


# COMMAND ----------

def _collect_employees_list_if_needed(employees_df_prepared):
    """
    Collect employees list from DataFrame only when fallback generation is needed.
    This avoids unnecessary collection when data products are available.
    employees_df now includes hire_date and current_job_start_date columns.
    """
    if employees_df_prepared is None:
        return None
    
    try:
        employees_list = []
        # Use toLocalIterator() to process in batches instead of collecting all at once
        # This reduces driver memory pressure for large datasets
        for row in employees_df_prepared.toLocalIterator():
            # Extract tenure values up front so they are always available
            try:
                tenure_months = row['tenure_months']
                if tenure_months is None:
                    tenure_months = 0
            except (KeyError, IndexError):
                tenure_months = 0
            
            try:
                months_in_current_role = row['months_in_current_role']
                if months_in_current_role is None:
                    months_in_current_role = 0
            except (KeyError, IndexError):
                months_in_current_role = 0
            
            # Use existing date columns if available, otherwise calculate from tenure
            try:
                hire_date_val = row['hire_date']
                if hire_date_val is not None:
                    if isinstance(hire_date_val, str):
                        hire_date_val = datetime.strptime(hire_date_val, '%Y-%m-%d').date()
                    elif hasattr(hire_date_val, 'date'):
                        hire_date_val = hire_date_val.date()
                else:
                    hire_date_val = date.today() - timedelta(days=int(tenure_months * 30))
            except (KeyError, IndexError):
                hire_date_val = date.today() - timedelta(days=int(tenure_months * 30))
            
            try:
                job_start_val = row['current_job_start_date']
                if job_start_val is not None:
                    if isinstance(job_start_val, str):
                        job_start_val = datetime.strptime(job_start_val, '%Y-%m-%d').date()
                    elif hasattr(job_start_val, 'date'):
                        job_start_val = job_start_val.date()
                else:
                    job_start_val = date.today() - timedelta(days=int(months_in_current_role * 30))
            except (KeyError, IndexError):
                job_start_val = date.today() - timedelta(days=int(months_in_current_role * 30))
            
            employees_list.append({
                'employee_id': row['employee_id'],
                'age': row['age'],
                'gender': row['gender'],
                'department': row['department'],
                'job_title': row['job_title'],
                'job_level': row['job_level'],
                'location': row['location'],
                'employment_type': row['employment_type'],
                'base_salary': row['base_salary'],
                'tenure_months': tenure_months,
                'months_in_current_role': months_in_current_role,
                'employment_status': row['employment_status'],
                'first_name': row['first_name'],
                'last_name': row['last_name'],
                'hire_date': hire_date_val,
                'current_job_start_date': job_start_val
            })
        
        return employees_list
    except Exception as e:
        print(f"   ⚠️ Warning: Could not collect employees list: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_or_generate_data():
    """
    Main function: Try to load from data products, fall back to generated data if needed.
    Only generates data when data product load fails (lazy generation).
    Returns dictionary with all dataframes.
    """
    print("="*80)
    print("📊 Loading/Generating SAP SuccessFactors Data")
    print("="*80)
    print("   Strategy: Try data products first, generate only if needed")
    
    # Track data sources for summary
    data_sources = {}
    employees_list_for_dependencies = None  # Will be set after employees are loaded
    
    # Try to load employees from data product (with fallback to generated)
    print("\n" + "="*80)
    print("EMPLOYEES DATASET")
    print("="*80)
    employees_df, employees_source = load_employees_from_data_product(generated_employees=None)

    data_sources['employees'] = employees_source
    
    # employees_df now includes hire_date and current_job_start_date columns
    # No need for _prepare_employees_df_for_generation() - use employees_df directly
    print("   ℹ️ Employees DataFrame ready for dependent dataset generation")
    # Will collect only if fallback generation is needed (lazy collection)
    employees_list_for_dependencies = None  # Will be collected only when needed
    
    # Try to load performance from data product (with fallback to generated)
    print("\n" + "="*80)
    print("PERFORMANCE REVIEWS DATASET")
    print("="*80)
    performance_df, performance_source = load_performance_from_data_product(
        generated_performance_reviews=None,
        employees=employees_list_for_dependencies,
        employees_df=employees_df  # Pass full employees_df to ensure completeness
    )

    data_sources['performance'] = performance_source
    
    # Keep performance reviews as DataFrame - will collect only if fallback generation is needed
    # This avoids unnecessary collection when data products are available
    performance_reviews_df = performance_df.select(
        'employee_id', 'review_date', 'overall_rating', 'competency_rating',
        'goals_achievement', 'review_id', 'review_period', 'reviewer_id', 'status'
    )
    performance_reviews_list = None  # Will be collected only when needed for fallback generation
    
    # Try to load learning from data product (with fallback to generated)
    print("\n" + "="*80)
    print("LEARNING RECORDS DATASET")
    print("="*80)
    learning_df, learning_source = load_learning_from_data_product(
        generated_learning_records=None,
        employees_df=employees_df,  # Pass full employees_df to ensure completeness
        employees_list=employees_list_for_dependencies,  # Keep for backward compatibility
        performance_reviews_df=performance_reviews_df,  # Pass DataFrame instead of list
        performance_reviews_list=performance_reviews_list  # Keep for backward compatibility
    )

    data_sources['learning'] = learning_source
    
    # Goals and Compensation - always use generated data (no data products available yet)
    print("\n" + "="*80)
    print("GOALS & COMPENSATION DATASETS")
    print("="*80)
    print("📊 Generating goals and compensation data...")
    print("   📦 Data Source: Generated (simulated) - No data products available")
    
    # Generate only what's needed for goals and compensation
    # Collect employees list only if needed (lazy collection)
    if employees_list_for_dependencies is None:
        if employees_df is not None:
            print("   → Collecting employees list from DataFrame for goals/compensation generation...")
            employees_list_for_dependencies = _collect_employees_list_if_needed(employees_df)
        else:
            print("   → Generating employees for goals/compensation...")
            employees_list_for_dependencies = generate_employees()
            print("   ✅ Generated employees data for goals/compensation")
    
    if employees_list_for_dependencies is None or len(employees_list_for_dependencies) == 0:
        print("   → Generating employees for goals/compensation...")
        employees_list_for_dependencies = generate_employees()
        print("   ✅ Generated employees data for goals/compensation")
    
    # Generate performance reviews if not already available (needed for goals/compensation)
    # Collect performance reviews list only if needed (lazy collection)
    if performance_reviews_list is None:
        if performance_reviews_df is not None:
            print("   → Collecting performance reviews list from DataFrame for goals/compensation generation...")
            try:
                performance_reviews_list = [
                    {
                        'employee_id': row['employee_id'],
                        'review_date': row['review_date'],
                        'overall_rating': row['overall_rating'],
                        'competency_rating': row['competency_rating'],
                        'goals_achievement': row['goals_achievement'],
                        'review_id': row['review_id'],
                        'review_period': row['review_period'],
                        'reviewer_id': row['reviewer_id'],
                        'status': row['status']
                    }
                    for row in performance_reviews_df.toLocalIterator()  # Use toLocalIterator instead of collect()
                ]
            except Exception as e:
                print(f"   ⚠️ Could not collect performance reviews: {e}")
                performance_reviews_list = None
    
    if performance_reviews_list is None or len(performance_reviews_list) == 0:
        print("   → Generating performance reviews for goals/compensation...")
        performance_reviews_list = generate_performance_reviews(employees_list_for_dependencies)
        print("   ✅ Generated performance reviews for goals/compensation")
    
    # Generate goals and compensation
    print("   → Generating goals data...")
    goals_data = generate_goals(employees_list_for_dependencies, performance_reviews_list)
    
    print("   → Generating compensation data...")
    compensation_data = generate_compensation(employees_list_for_dependencies, performance_reviews_list)
    
    # Check if we have data before creating DataFrames
    if len(goals_data) == 0:
        print("   ⚠️ Warning: No goals generated. Creating empty DataFrame with schema...")
        # Create empty DataFrame with proper schema
        goals_schema = StructType([
            StructField("goal_id", StringType(), True),
            StructField("employee_id", StringType(), True),
            StructField("goal_title", StringType(), True),
            StructField("goal_type", StringType(), True),
            StructField("start_date", DateType(), True),
            StructField("target_date", DateType(), True),
            StructField("achievement_percentage", IntegerType(), True),
            StructField("weight", IntegerType(), True),
            StructField("status", StringType(), True)
        ])
        goals_df = spark.createDataFrame([], schema=goals_schema)
    else:
        goals_df = spark.createDataFrame(goals_data)
    
    
    if len(compensation_data) == 0:
        print("   ⚠️ Warning: No compensation records generated. Creating empty DataFrame with schema...")
        # Create empty DataFrame with proper schema
        compensation_schema = StructType([
            StructField("comp_id", StringType(), True),
            StructField("employee_id", StringType(), True),
            StructField("effective_date", DateType(), True),
            StructField("base_salary", IntegerType(), True),
            StructField("bonus_target_pct", IntegerType(), True),
            StructField("equity_value", IntegerType(), True),
            StructField("adjustment_reason", StringType(), True)
        ])
        compensation_df = spark.createDataFrame([], schema=compensation_schema)
    else:
        compensation_df = spark.createDataFrame(compensation_data)
    
    
    data_sources['goals'] = 'GENERATED'
    data_sources['compensation'] = 'GENERATED'
    
    return {
        'employees': employees_df,
        'performance': performance_df,
        'learning': learning_df,
        'goals': goals_df,
        'compensation': compensation_df
    }, None


# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Load or Generate Data
# MAGIC ### *Load from data products first, generate if needed*

# COMMAND ----------

# Initialize tracking variables
print("="*80)
print("📊 Loading/Generating SAP SuccessFactors Data")
print("="*80)
print("   Strategy: Try data products first, generate only if needed")

data_sources = {}
employees_list_for_dependencies = None
performance_reviews_df = None
performance_reviews_list = None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load/Generate Employees

# COMMAND ----------

print("\n" + "="*80)
print("EMPLOYEES DATASET")
print("="*80)
employees_df, employees_source = load_employees_from_data_product(generated_employees=None)
data_sources['employees'] = employees_source

# employees_df now includes hire_date and current_job_start_date columns
# No need for _prepare_employees_df_for_generation() - use employees_df directly
print("   ℹ️ Employees DataFrame ready for dependent dataset generation")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load/Generate Performance Reviews

# COMMAND ----------

print("\n" + "="*80)
print("PERFORMANCE REVIEWS DATASET")
print("="*80)
performance_df, performance_source = load_performance_from_data_product(
    generated_performance_reviews=None,
    employees=employees_list_for_dependencies,
    employees_df=employees_df  # Pass full employees_df to ensure completeness
)
data_sources['performance'] = performance_source

# Keep performance reviews as DataFrame - will collect only if fallback generation is needed
performance_reviews_df = performance_df.select(
    'employee_id', 'review_date', 'overall_rating', 'competency_rating',
    'goals_achievement', 'review_id', 'review_period', 'reviewer_id', 'status'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load/Generate Learning Records

# COMMAND ----------

print("\n" + "="*80)
print("LEARNING RECORDS DATASET")
print("="*80)
learning_df, learning_source = load_learning_from_data_product(
    generated_learning_records=None,
    employees_df=employees_df,  # Pass full employees_df to ensure completeness
    employees_list=employees_list_for_dependencies,
    performance_reviews_df=performance_reviews_df,
    performance_reviews_list=performance_reviews_list
)
data_sources['learning'] = learning_source

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Goals

# COMMAND ----------

print("\n" + "="*80)
print("GOALS DATASET")
print("="*80)
print("📊 Generating goals data...")
print("   📦 Data Source: Generated (simulated) - No data products available")

# Collect employees list only if needed (lazy collection)
if employees_list_for_dependencies is None:
    if employees_df is not None:
        print("   → Collecting employees list from DataFrame for goals generation...")
        employees_list_for_dependencies = _collect_employees_list_if_needed(employees_df)
    else:
        print("   → Generating employees for goals...")
        employees_list_for_dependencies = generate_employees()
        print("   ✅ Generated employees data for goals")

if employees_list_for_dependencies is None or len(employees_list_for_dependencies) == 0:
    print("   → Generating employees for goals...")
    employees_list_for_dependencies = generate_employees()
    print("   ✅ Generated employees data for goals")

# Generate performance reviews if not already available (needed for goals)
if performance_reviews_list is None:
    if performance_reviews_df is not None:
        print("   → Collecting performance reviews list from DataFrame for goals generation...")
        try:
            performance_reviews_list = [
                {
                    'employee_id': row['employee_id'],
                    'review_date': row['review_date'],
                    'overall_rating': row['overall_rating'],
                    'competency_rating': row['competency_rating'],
                    'goals_achievement': row['goals_achievement'],
                    'review_id': row['review_id'],
                    'review_period': row['review_period'],
                    'reviewer_id': row['reviewer_id'],
                    'status': row['status']
                }
                for row in performance_reviews_df.toLocalIterator()
            ]
        except Exception as e:
            print(f"   ⚠️ Could not collect performance reviews: {e}")
            performance_reviews_list = None

if performance_reviews_list is None or len(performance_reviews_list) == 0:
    print("   → Generating performance reviews for goals...")
    performance_reviews_list = generate_performance_reviews(employees_list_for_dependencies)
    print("   ✅ Generated performance reviews for goals")

# Generate goals
print("   → Generating goals data...")
goals_data = generate_goals(employees_list_for_dependencies, performance_reviews_list)
print("   ✅ Generated goals data for notebook cell")

# Create goals DataFrame
if len(goals_data) == 0:
    print("   ⚠️ Warning: No goals generated. Creating empty DataFrame with schema...")
    goals_schema = StructType([
        StructField("goal_id", StringType(), True),
        StructField("employee_id", StringType(), True),
        StructField("goal_title", StringType(), True),
        StructField("goal_type", StringType(), True),
        StructField("start_date", DateType(), True),
        StructField("target_date", DateType(), True),
        StructField("achievement_percentage", IntegerType(), True),
        StructField("weight", IntegerType(), True),
        StructField("status", StringType(), True)
    ])
    goals_df = spark.createDataFrame([], schema=goals_schema)
else:
    goals_df = spark.createDataFrame(goals_data)

data_sources['goals'] = 'GENERATED'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Compensation

# COMMAND ----------

print("\n" + "="*80)
print("COMPENSATION DATASET")
print("="*80)
print("📊 Generating compensation data...")
print("   📦 Data Source: Generated (simulated) - No data products available")

# Generate compensation
print("   → Generating compensation data...")
compensation_data = generate_compensation(employees_list_for_dependencies, performance_reviews_list)
print("   ✅ Generated compensation data for notebook cell")

# Create compensation DataFrame
if len(compensation_data) == 0:
    print("   ⚠️ Warning: No compensation records generated. Creating empty DataFrame with schema...")
    compensation_schema = StructType([
        StructField("comp_id", StringType(), True),
        StructField("employee_id", StringType(), True),
        StructField("effective_date", DateType(), True),
        StructField("base_salary", IntegerType(), True),
        StructField("bonus_target_pct", IntegerType(), True),
        StructField("equity_value", IntegerType(), True),
        StructField("adjustment_reason", StringType(), True)
    ])
    compensation_df = spark.createDataFrame([], schema=compensation_schema)
else:
    compensation_df = spark.createDataFrame(compensation_data)

data_sources['compensation'] = 'GENERATED'

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Save Data to Unity Catalog Tables

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Employees Table

# COMMAND ----------

# Ensure consistent numeric types for employees
# Use try_cast to safely handle malformed values
employees_df = employees_df.withColumn("age", F.coalesce(F.expr("try_cast(age as int)"), F.lit(0))) \
                           .withColumn("base_salary", F.coalesce(F.expr("try_cast(base_salary as int)"), F.lit(0))) \
                           .withColumn("job_level", F.coalesce(F.expr("try_cast(job_level as int)"), F.lit(0))) \
                           .withColumn("tenure_months", F.coalesce(F.expr("try_cast(tenure_months as int)"), F.lit(0))) \
                           .withColumn("months_in_current_role", F.coalesce(F.expr("try_cast(months_in_current_role as int)"), F.lit(0))) \
                           .withColumn("first_name", F.coalesce(F.col("first_name"), F.lit("Unknown"))) \
                           .withColumn("last_name", F.coalesce(F.col("last_name"), F.lit("Unknown")))

# Save employees table
full_table_name = f"{catalog_name}.{schema_name}.employees"
employees_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table_name)
print(f"✅ Created table: {full_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Performance Reviews Table

# COMMAND ----------

# Ensure consistent numeric types for performance reviews
# Use try_cast to safely handle malformed values
performance_df = performance_df.withColumn("overall_rating", F.coalesce(F.expr("try_cast(overall_rating as double)"), F.lit(3.5))) \
                                .withColumn("competency_rating", F.coalesce(F.expr("try_cast(competency_rating as double)"), F.lit(3.5))) \
                                .withColumn("goals_achievement", F.coalesce(F.expr("try_cast(goals_achievement as int)"), F.lit(0)))

# Save performance table
full_table_name = f"{catalog_name}.{schema_name}.performance"
performance_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table_name)
print(f"✅ Created table: {full_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Learning Records Table

# COMMAND ----------

# Save learning table
full_table_name = f"{catalog_name}.{schema_name}.learning"
learning_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table_name)
print(f"✅ Created table: {full_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Goals Table

# COMMAND ----------

# Ensure consistent numeric types for goals
# Use try_cast to safely handle malformed values
goals_df = goals_df.withColumn("achievement_percentage", F.coalesce(F.expr("try_cast(achievement_percentage as int)"), F.lit(0))) \
                   .withColumn("weight", F.coalesce(F.expr("try_cast(weight as int)"), F.lit(0)))

# Save goals table
full_table_name = f"{catalog_name}.{schema_name}.goals"
goals_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table_name)
print(f"✅ Created table: {full_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Compensation Table

# COMMAND ----------

# Ensure consistent numeric types for compensation
# Use try_cast to safely handle malformed values
compensation_df = compensation_df.withColumn("base_salary", F.coalesce(F.expr("try_cast(base_salary as int)"), F.lit(0))) \
                                 .withColumn("bonus_target_pct", F.coalesce(F.expr("try_cast(bonus_target_pct as int)"), F.lit(0))) \
                                 .withColumn("equity_value", F.coalesce(F.expr("try_cast(equity_value as int)"), F.lit(0)))

# Save compensation table
full_table_name = f"{catalog_name}.{schema_name}.compensation"
compensation_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table_name)
print(f"✅ Created table: {full_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary

# COMMAND ----------

table_counts = {
    'employees': employees_df.count(),
    'performance': performance_df.count(),
    'learning': learning_df.count(),
    'goals': goals_df.count(),
    'compensation': compensation_df.count()
}

print(f"\n✅ All data saved to Unity Catalog: {catalog_name}.{schema_name}")

displayHTML(f"""
<div style="background: linear-gradient(135deg, #0052CC 0%, #0070F2 100%); 
            padding: 25px; border-radius: 15px; color: white; margin: 20px 0;">
    <h2 style="text-align: center; margin-bottom: 20px;">✅ Data Generation Complete</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Employees</h3>
            <p style="font-size: 24px; margin: 10px 0;">{table_counts['employees']:,}</p>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Performance Reviews</h3>
            <p style="font-size: 24px; margin: 10px 0;">{table_counts['performance']:,}</p>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Learning Records</h3>
            <p style="font-size: 24px; margin: 10px 0;">{table_counts['learning']:,}</p>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Goals</h3>
            <p style="font-size: 24px; margin: 10px 0;">{table_counts['goals']:,}</p>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Compensation</h3>
            <p style="font-size: 24px; margin: 10px 0;">{table_counts['compensation']:,}</p>
        </div>
    </div>
    <p style="text-align: center; margin-top: 20px; opacity: 0.9;">
        Data saved to Unity Catalog: <strong>{catalog_name}.{schema_name}</strong>
    </p>
    <p style="text-align: center; margin-top: 10px; opacity: 0.9;">
        Tables: employees, performance, learning, goals, compensation
    </p>
</div>
""")

print(f"\n🎉 Data generation complete! Tables available in {catalog_name}.{schema_name}")
print("✅ Ready for ML model training in next notebook")
