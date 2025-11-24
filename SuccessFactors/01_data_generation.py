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

# ============================================================================
# CONSTANTS & THRESHOLDS
# ============================================================================

# Employment Status Values
ACTIVE_STATUS_VALUES = ['A', 'ACTIVE', 'ACT']
EMPLOYMENT_TYPES = ['Full-time', 'Part-time', 'Contract']
EMPLOYMENT_STATUS_OPTIONS = ['Active', 'Terminated']

# Employee IDs
ALEX_EMPLOYEE_ID = 'EMP100038'

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
    departments = ['Engineering', 'Product', 'Sales', 'Marketing', 'Finance', 'HR', 'Operations', 'Legal']
    locations = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide']  # Australian cities
    job_families = {
        'Engineering': ['Software Engineer', 'Senior Software Engineer', 'Staff Engineer', 'Engineering Manager', 'Director Engineering'],
        'Product': ['Product Analyst', 'Product Manager', 'Senior PM', 'Principal PM', 'VP Product'],
        'Sales': ['Sales Rep', 'Account Executive', 'Sales Manager', 'Regional Director', 'VP Sales'],
        'Marketing': ['Marketing Specialist', 'Marketing Manager', 'Senior Manager', 'Marketing Director', 'CMO'],
        'Finance': ['Financial Analyst', 'Senior Analyst', 'Finance Manager', 'Finance Director', 'CFO'],
        'HR': ['HR Generalist', 'HR Business Partner', 'HR Manager', 'HR Director', 'CHRO'],
        'Operations': ['Operations Analyst', 'Operations Manager', 'Senior Manager', 'Operations Director', 'COO'],
        'Legal': ['Legal Counsel', 'Senior Counsel', 'Legal Director', 'General Counsel']
    }
    
    employee_id = 100000
    
    # First, create Alex Smith with specific characteristics for demo
    alex_id = 100038  # Ensure Alex gets this ID
    alex_hire_date = date.today() - timedelta(days=540)  # 18 months ago
    alex_job_start = alex_hire_date + timedelta(days=30)
    
    employees.append({
        'employee_id': f'EMP{alex_id}',
        'person_id': f'PER{alex_id + 50000}',
        'first_name': 'Alex',
        'last_name': 'Smith',
        'gender': 'Female',
        'age': 32,
        'hire_date': alex_hire_date,
        'current_job_start_date': alex_job_start,
        'department': 'Engineering',
        'job_title': 'Software Engineer',
        'job_level': 1,
        'location': 'Sydney',
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
            gender = random.choices(['Male', 'Female', 'Non-binary'], weights=[48, 48, 4])[0]
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
            location = random.choice(locations)
            location_multipliers = {
                'Sydney': 1.15,      # Highest cost of living, highest salaries
                'Melbourne': 1.05,   # Second major market
                'Brisbane': 0.95,    # Growing tech hub, slightly lower
                'Perth': 0.92,       # Mining/resources focus
                'Adelaide': 0.88     # Smaller market
            }
            base_salary = int(base_salary * location_multipliers.get(location, 1.0))
            
            employees.append({
                'employee_id': f'EMP{employee_id}',
                'person_id': f'PER{employee_id + 50000}',
                'first_name': random.choice(['Alex', 'Sarah', 'Michael', 'Jessica', 'David', 'Emily', 'Chris', 'Amanda', 'Ryan', 'Lisa', 'John', 'Maria', 'James', 'Jennifer', 'Robert', 'Emma', 'Olivia', 'Charlotte', 'Sophia', 'Isabella', 'Noah', 'Oliver', 'William', 'Lucas', 'Benjamin', 'Mia', 'Harper', 'Evelyn', 'Abigail', 'Elijah', 'Mason', 'Alexander', 'Daniel', 'Matthew', 'Aiden']),
                'last_name': random.choice(['Smith', 'Jones', 'Williams', 'Brown', 'Wilson', 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young', 'King', 'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 'Nelson', 'Carter', 'Mitchell', 'Roberts', 'Turner', 'Phillips', 'Campbell', 'Parker', 'Evans', 'Edwards', 'Collins', 'Stewart', 'Sanchez', 'Morris', 'Rogers', 'Reed', 'Cook', 'Morgan', 'Bell', 'Murphy', 'Bailey', 'Rivera', 'Cooper', 'Richardson', 'Cox', 'Howard', 'Ward', 'Torres', 'Peterson', 'Gray', 'Ramirez', 'James', 'Watson', 'Brooks', 'Kelly', 'Sanders', 'Price', 'Bennett', 'Wood', 'Barnes', 'Ross', 'Henderson', 'Coleman', 'Jenkins', 'Perry', 'Powell', 'Long', 'Patterson', 'Hughes', 'Flores', 'Washington', 'Butler', 'Simmons', 'Foster', 'Gonzales', 'Bryant', 'Alexander', 'Russell', 'Griffin', 'Diaz', 'Hayes']),
                'gender': gender,
                'age': int(age),
                'hire_date': hire_date,
                'current_job_start_date': current_job_start,
                'department': dept,
                'job_title': job_levels[current_level_idx],
                'job_level': int(current_level_idx),
                'location': location,
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
            eng_managers = [e['employee_id'] for e in employees if e['department'] == 'Engineering' and e['job_level'] >= MIN_MANAGER_LEVEL]
            if eng_managers:
                emp['manager_id'] = random.choice(eng_managers)
    
    print(f"✅ Generated {len(employees)} employees")
    return employees


def generate_performance_reviews(employees):
    """Generate performance review data based on employees"""
    random.seed(42)
    np.random.seed(42)
    
    print("🔄 Generating performance reviews data...")
    
    # Check input
    if not employees or len(employees) == 0:
        print("   ⚠️ Warning: No employees provided for performance review generation")
        return []
    
    print(f"   ℹ️ Processing {len(employees):,} employees for performance reviews")
    
    # Performance Management Data (PerformanceReviews from SAP BDC)
    # Strong signal: Performance correlates with job level, tenure, and creates patterns
    performance_reviews = []
    
    # Calculate base performance potential for each employee (hidden trait)
    employee_performance_potential = {}
    active_count = 0
    for emp in employees:
        # Check if employee is active (handles 'A', 'Active', 'ACTIVE', etc.)
        if is_employee_active(emp.get('employment_status')):
            active_count += 1
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
    
    print(f"   ℹ️ Found {active_count:,} active employees")
    
    # Special handling for Alex Smith - make her a high performer
    employee_performance_potential['EMP100038'] = 4.2
    
    eligible_count = 0
    for emp in employees:
        # Check if employee is active (handles 'A', 'Active', 'ACTIVE', etc.)
        # Get tenure_months with proper handling
        tenure_months = emp.get('tenure_months', 0)
        if not isinstance(tenure_months, (int, float)) or tenure_months is None:
            tenure_months = 0
        
        if is_employee_active(emp.get('employment_status')) and tenure_months >= MIN_TENURE_MONTHS_FOR_REVIEWS:
            eligible_count += 1
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
    
    print(f"   ℹ️ Eligible employees (Active + tenure >= {MIN_TENURE_MONTHS_FOR_REVIEWS} months): {eligible_count:,}")
    print(f"✅ Generated {len(performance_reviews)} performance reviews")
    
    if len(performance_reviews) == 0:
        print("   ⚠️ Warning: No performance reviews generated!")
        print(f"   → Check: employment_status values and tenure_months >= {MIN_TENURE_MONTHS_FOR_REVIEWS}")
        # Show sample employee data for debugging
        if employees:
            sample_emp = employees[0]
            print(f"   → Sample employee: status='{sample_emp.get('employment_status')}', tenure={sample_emp.get('tenure_months')}, job_level={sample_emp.get('job_level')}")
    
    return performance_reviews


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
            if emp['department'] == 'Engineering':
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
    
    print(f"✅ Generated {len(learning_records)} learning records")
    return learning_records


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
    
    # Calculate base performance potential for each employee
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
            # Get employee's performance level
            emp_perf = latest_performance.get(emp['employee_id'], {})
            perf_rating = emp_perf.get('overall_rating', employee_performance_potential.get(emp['employee_id'], 3.0))
            goals_achievement_avg = emp_perf.get('goals_achievement', 85)
            
            # High performers may have more goals
            if perf_rating >= HIGH_PERFORMER_THRESHOLD:
                num_goals = random.randint(4, 6)
            else:
                num_goals = random.randint(2, 5)
            
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
    
    print(f"✅ Generated {len(goals)} goals")
    return goals


def generate_compensation(employees, performance_reviews):
    """Generate compensation history based on employees and performance"""
    random.seed(42)
    np.random.seed(42)
    
    print("🔄 Generating compensation history data...")
    
    # Compensation History (Compensation from SAP BDC)
    # Strong signal: Salary increases correlate with performance and promotions
    compensation_history = []
    
    for emp in employees:
        if is_employee_active(emp.get('employment_status')):
            # Get employee's performance history
            emp_reviews = [r for r in performance_reviews if r['employee_id'] == emp['employee_id']]
            emp_reviews.sort(key=lambda x: x['review_date'])
            
            # Generate salary history with performance-based increases
            num_adjustments = max(1, emp['tenure_months'] // 12)  # Annual adjustments
            current_salary = emp['base_salary']
            starting_salary = int(current_salary * 0.85)  # Starting salary lower
            
            for adj_num in range(num_adjustments):
                effective_date = emp['hire_date'] + timedelta(days=365 * adj_num)
                
                if adj_num == 0:
                    salary = starting_salary
                    reason = 'Initial Hire'
                else:
                    # Check if there was a promotion around this time
                    job_start = emp['current_job_start_date']
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
                if emp['job_level'] >= MIN_SENIOR_LEVEL:
                    bonus_target = random.randint(SENIOR_LEVEL_BONUS_MIN, SENIOR_LEVEL_BONUS_MAX)
                elif emp['job_level'] >= MIN_MANAGER_LEVEL:
                    bonus_target = random.randint(MANAGER_LEVEL_BONUS_MIN, MANAGER_LEVEL_BONUS_MAX)
                else:
                    bonus_target = random.randint(0, 15)
                
                # Equity for higher levels
                if emp['job_level'] >= MIN_MANAGER_LEVEL:
                    equity = random.randint(SENIOR_LEVEL_EQUITY_MIN, SENIOR_LEVEL_EQUITY_MAX) if emp['job_level'] >= MIN_SENIOR_LEVEL else random.randint(MANAGER_LEVEL_EQUITY_MIN, MANAGER_LEVEL_EQUITY_MAX)
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
    
    print(f"✅ Generated {len(compensation_history)} compensation records")
    return compensation_history


# ============================================================================
# DATA PRODUCT LOADING FUNCTIONS
# ============================================================================

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
    print("   Source: core_workforce_data_dp.coreworkforcedata.coreworkforce_standardfields")
    try:
        employees_df_raw = spark.sql(
            "SELECT * FROM core_workforce_data_dp.coreworkforcedata.coreworkforce_standardfields"
        )
        
        record_count = employees_df_raw.count()
        print(f"✅ Successfully loaded {record_count:,} employee records from DATA PRODUCT")
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
            # Use window function to rank records by startDate, keeping latest (descending order)
            window_spec = Window.partitionBy("userId").orderBy(F.col(start_date_col).desc())
            employees_df_raw = employees_df_raw.withColumn("row_num", F.row_number().over(window_spec)) \
                                               .filter(F.col("row_num") == LATEST_RECORD_ROW_NUM) \
                                               .drop("row_num")
            
            deduped_count = employees_df_raw.count()
            print(f"   ✅ Deduplicated to {deduped_count:,} unique employee records ({record_count - deduped_count:,} duplicates removed)")
        else:
            print(f"   ⚠️ Warning: Could not find startDate column. Available columns: {', '.join(employees_df_raw.columns[:10])}...")
            # Fallback: use any date column or just take first record per employee_id
            print(f"   → Using simple deduplication (keeping first record per employee_id)")
            employees_df_raw = employees_df_raw.dropDuplicates(["userId"])
            deduped_count = employees_df_raw.count()
            print(f"   ✅ Deduplicated to {deduped_count:,} unique employee records")
        
        # Map columns to expected names (same as notebook 03)
        # Use try_cast to handle malformed values (e.g., '<10' in age field)
        employees_df = employees_df_raw.select(
            F.col("userId").alias("employee_id"),
            F.coalesce(F.expr("try_cast(age as int)"), F.lit(DEFAULT_AGE)).alias("age"),  # Default to 30 if invalid
            F.col("gender").alias("gender"),
            F.col("department").alias("department"),
            F.col("jobTitle").alias("job_title"),
            F.when(F.col("jobTitle").rlike(JOB_TITLE_PATTERN_SENIOR), 3)
             .when(F.col("jobTitle").rlike(JOB_TITLE_PATTERN_MID), 2)
             .otherwise(1).alias("job_level"),
            F.col("location").alias("location"),
            F.col("employmentType").alias("employment_type"),
            F.coalesce(F.expr("try_cast(annualSalary as int)"), F.lit(DEFAULT_SALARY)).alias("base_salary"),  # Default to 0 if invalid
            F.coalesce(F.expr("try_cast(totalOrgTenureCalc / 30 as int)"), F.lit(DEFAULT_TENURE_MONTHS)).alias("tenure_months"),  # Default to 0 if invalid
            F.coalesce(F.expr("try_cast(totalPositionTenureCalc / 30 as int)"), F.lit(DEFAULT_TENURE_MONTHS)).alias("months_in_current_role"),  # Default to 0 if invalid
            F.col("employmentStatus").alias("employment_status"),
            F.lit("Unknown").alias("first_name"),
            F.lit("Unknown").alias("last_name")
        )
        
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
        
        final_count = employees_df.count()
        print(f"✅ Transformed employees data: {final_count:,} records")
        print(f"   📊 Final Status: Using DATA PRODUCT data")
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
        
        final_count = employees_df.count()
        print(f"✅ Using generated employees data: {final_count:,} records")
        print(f"   📊 Final Status: Using GENERATED data (fallback)")
        return employees_df, 'GENERATED'  # Return source indicator


def load_performance_from_data_product(generated_performance_reviews=None, employees=None):
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
    print("   Source: performance_reviews_dp.performancereviews.commentfeedback")
    try:
        performance_df_raw = spark.sql(
            "SELECT * FROM performance_reviews_dp.performancereviews.commentfeedback"
        )
        
        record_count = performance_df_raw.count()
        print(f"✅ Successfully loaded {record_count:,} performance review records from DATA PRODUCT")
        print(f"   📦 Data Source: SAP SuccessFactors Data Product (Delta Sharing)")
        
        # Map columns to expected names
        # Use try_cast to handle malformed values
        performance_df = performance_df_raw.select(
            F.col("subject_8995a2862a8343bd8390aaa82c46e881").alias("employee_id"),
            F.col("modifiedAt").alias("review_date"),
            F.coalesce(F.expr("try_cast(numberValue as double)"), F.lit(DEFAULT_RATING)).alias("overall_rating"),  # Default to 3.0 if invalid
            F.coalesce(F.expr("try_cast(numberValue as double)"), F.lit(DEFAULT_RATING)).alias("competency_rating"),  # Default to 3.0 if invalid
            F.coalesce(F.expr("try_cast(numberValue * 20 as int)"), F.lit(DEFAULT_GOALS_ACHIEVEMENT)).alias("goals_achievement"),  # Default to 60 if invalid
            F.col("id").cast("string").alias("review_id"),
            F.coalesce(F.expr("try_cast(year(modifiedAt) as int)"), F.lit(DEFAULT_REVIEW_YEAR)).alias("review_period"),  # Default to 2024 if invalid
            F.lit(None).cast("string").alias("reviewer_id"),
            F.lit(COMPLETION_STATUS_COMPLETED).alias("status")
        ).filter(F.col("numberValue").isNotNull() & (F.col("numberValue") > 0))
        
        final_count = performance_df.count()
        print(f"✅ Transformed performance data: {final_count:,} records")
        print(f"   📊 Final Status: Using DATA PRODUCT data")
        return performance_df, 'DATA PRODUCT'  # Return source indicator
        
    except Exception as e:
        print(f"⚠️ Error loading performance from data product: {e}")
        print(f"   🔄 FALLBACK: Switching to generated performance reviews data...")
        print(f"   📦 Data Source: Generated (simulated)")
        
        # Fallback to generated data - generate only when needed
        if generated_performance_reviews is None:
            if employees is None or len(employees) == 0:
                print("   → Generating employees and performance reviews data (on-demand)...")
                employees = generate_employees()
                print(f"   ✅ Generated {len(employees):,} employees")
            else:
                print(f"   → Generating performance reviews data (on-demand) for {len(employees):,} employees...")
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
        
        final_count = performance_df.count()
        print(f"✅ Using generated performance reviews data: {final_count:,} records")
        print(f"   📊 Final Status: Using GENERATED data (fallback)")
        return performance_df, 'GENERATED'  # Return source indicator


def load_learning_from_data_product(generated_learning_records=None, employees=None, performance_reviews=None):
    """
    Load learning records from SAP SuccessFactors Data Product.
    Falls back to generated data if data product load fails.
    
    Args:
        generated_learning_records: Optional list of generated learning record dicts for fallback
        employees: Optional list of employee dicts (needed if generating fallback)
        performance_reviews: Optional list of performance review dicts (needed if generating fallback)
        
    Returns:
        Spark DataFrame with learning records data (from DP or generated)
    """
    print("📊 Loading learning records from SAP SuccessFactors Data Product...")
    print("   Source: learning_history_dp.learninghistory.learningcompletion")
    try:
        learning_df_raw = spark.sql(
            "SELECT * FROM learning_history_dp.learninghistory.learningcompletion"
        )
        
        record_count = learning_df_raw.count()
        print(f"✅ Successfully loaded {record_count:,} learning records from DATA PRODUCT")
        print(f"   📦 Data Source: SAP SuccessFactors Data Product (Delta Sharing)")
        
        # Map completionStatusID to completion_status string
        learning_df = learning_df_raw.withColumn(
            "completion_status",
            F.when(F.col("completionStatusID") == COMPLETION_STATUS_ID_COMPLETED, COMPLETION_STATUS_COMPLETED)
             .when(F.col("completionStatusID") == COMPLETION_STATUS_ID_IN_PROGRESS, COMPLETION_STATUS_IN_PROGRESS)
             .when(F.col("completionStatusID") == COMPLETION_STATUS_ID_NOT_STARTED, COMPLETION_STATUS_NOT_STARTED)
             .when(F.col("completionStatusID").isNull(), COMPLETION_STATUS_NOT_STARTED)
             .when(F.lower(F.col("completionStatusID")).contains("complete"), COMPLETION_STATUS_COMPLETED)
             .when(F.lower(F.col("completionStatusID")).contains("progress"), COMPLETION_STATUS_IN_PROGRESS)
             .otherwise(COMPLETION_STATUS_NOT_STARTED)
        )
        
        # Map userID to employee_id
        learning_df = learning_df.withColumn(
            "employee_id",
            F.coalesce(
                F.col("userID"),
                F.col("personId"),
                F.col("subject_8995a2862a8343bd8390aaa82c46e881")
            )
        )
        
        # Derive category from learningItemType
        learning_df = learning_df.withColumn(
            "category",
            F.when(F.lower(F.col("learningItemType")).contains("technical"), LEARNING_CATEGORY_KEYWORDS["technical"])
             .when(F.lower(F.col("learningItemType")).contains("leadership"), LEARNING_CATEGORY_KEYWORDS["leadership"])
             .when(F.lower(F.col("learningItemType")).contains("communication"), LEARNING_CATEGORY_KEYWORDS["communication"])
             .when(F.lower(F.col("learningItemType")).contains("project"), LEARNING_CATEGORY_KEYWORDS["project"])
             .when(F.lower(F.col("learningItemType")).contains("data"), LEARNING_CATEGORY_KEYWORDS["data"])
             .when(F.lower(F.col("learningItemType")).contains("product"), LEARNING_CATEGORY_KEYWORDS["product"])
             .when(F.lower(F.col("learningItemType")).contains("sales"), LEARNING_CATEGORY_KEYWORDS["sales"])
             .when(F.lower(F.col("learningItemType")).contains("compliance"), LEARNING_CATEGORY_KEYWORDS["compliance"])
             .when(F.col("courseId").isNotNull(), LEARNING_CATEGORY_KEYWORDS["technical"])
             .when(F.col("programId").isNotNull(), LEARNING_CATEGORY_KEYWORDS["leadership"])
             .otherwise("General Training")
        )
        
        # Use learningCompletionEventSysGUID as learning_id
        learning_df = learning_df.withColumn(
            "learning_id",
            F.coalesce(
                F.col("learningCompletionEventSysGUID"),
                F.col("componentID"),
                F.concat(F.lit("LRN"), F.col("componentKey").cast("string"))
            )
        )
        
        # Map totalHours to hours_completed
        # Use try_cast to handle malformed values
        learning_df = learning_df.withColumn(
            "hours_completed",
            F.coalesce(
                F.expr("try_cast(totalHours as int)"),
                F.expr("try_cast((coalesce(creditHours, 0) + coalesce(cpeHours, 0) + coalesce(contactHours, 0)) as int)"),
                F.lit(0)
            )
        )
        
        # Score is not available, set to NULL
        learning_df = learning_df.withColumn("score", F.lit(None).cast("integer"))
        
        # Map completionDate
        learning_df = learning_df.withColumn(
            "completion_date",
            F.coalesce(
                F.col("completionDate"),
                F.col("revisionDate"),
                F.col("lastUpdatedTimestamp")
            )
        )
        
        # Select and rename columns to match expected schema
        learning_df = learning_df.select(
            F.col("learning_id").alias("learning_id"),
            F.col("employee_id").alias("employee_id"),
            F.col("category").alias("category"),
            F.col("completion_date").alias("completion_date"),
            F.col("hours_completed").alias("hours_completed"),
            F.col("completion_status").alias("completion_status"),
            F.col("score").alias("score"),
            F.col("componentID").alias("component_id"),
            F.col("courseId").alias("course_id"),
            F.col("programId").alias("program_id"),
            F.col("learningItemType").alias("learning_item_type")
        )
        
        # Filter out records with null employee_id
        learning_df = learning_df.filter(F.col("employee_id").isNotNull())
        
        final_count = learning_df.count()
        print(f"✅ Transformed learning data: {final_count:,} records")
        print(f"   📊 Final Status: Using DATA PRODUCT data")
        return learning_df, 'DATA PRODUCT'  # Return source indicator
        
    except Exception as e:
        print(f"⚠️ Error loading learning from data product: {e}")
        print(f"   🔄 FALLBACK: Switching to generated learning records data...")
        print(f"   📦 Data Source: Generated (simulated)")
        
        # Fallback to generated data - generate only when needed
        if generated_learning_records is None:
            if employees is None:
                print("   → Generating employees data (on-demand)...")
                employees = generate_employees()
            if performance_reviews is None:
                print("   → Generating performance reviews data (on-demand)...")
                performance_reviews = generate_performance_reviews(employees)
            print("   → Generating learning records data (on-demand)...")
            generated_learning_records = generate_learning_records(employees, performance_reviews)
        
        # Convert generated data to DataFrame with correct schema
        learning_df_generated = spark.createDataFrame(generated_learning_records)
        learning_df = learning_df_generated.select(
            F.col("learning_id").alias("learning_id"),
            F.col("employee_id").alias("employee_id"),
            F.col("category").alias("category"),
            F.col("completion_date").alias("completion_date"),
            F.col("hours_completed").cast("integer").alias("hours_completed"),
            F.col("completion_status").alias("completion_status"),
            F.when(F.col("score").isNotNull(), F.col("score").cast("integer")).otherwise(None).alias("score"),
            F.lit(None).cast("string").alias("component_id"),
            F.lit(None).cast("string").alias("course_id"),
            F.lit(None).cast("string").alias("program_id"),
            F.lit(None).cast("string").alias("learning_item_type")
        )
        
        final_count = learning_df.count()
        print(f"✅ Using generated learning records data: {final_count:,} records")
        print(f"   📊 Final Status: Using GENERATED data (fallback)")
        return learning_df, 'GENERATED'  # Return source indicator


# ============================================================================
# MAIN DATA LOADING FUNCTION
# ============================================================================

def _extract_employees_list_from_df(employees_df):
    """Helper function to extract employees list from DataFrame for use in generation functions"""
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
        
        employees_list = []
        for row in employees_df_with_dates.select(
            'employee_id', 'age', 'gender', 'department', 'job_title', 'job_level',
            'location', 'employment_type', 'base_salary', 'tenure_months',
            'months_in_current_role', 'employment_status', 'first_name', 'last_name',
            'hire_date', 'current_job_start_date'
        ).collect():
            # Convert Spark date objects to Python date objects
            hire_date_val = row['hire_date']
            if hire_date_val is not None:
                if isinstance(hire_date_val, str):
                    hire_date_val = datetime.strptime(hire_date_val, '%Y-%m-%d').date()
                elif hasattr(hire_date_val, 'date'):
                    hire_date_val = hire_date_val.date()
            
            job_start_val = row['current_job_start_date']
            if job_start_val is not None:
                if isinstance(job_start_val, str):
                    job_start_val = datetime.strptime(job_start_val, '%Y-%m-%d').date()
                elif hasattr(job_start_val, 'date'):
                    job_start_val = job_start_val.date()
            
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
                'tenure_months': row['tenure_months'],
                'months_in_current_role': row['months_in_current_role'],
                'employment_status': row['employment_status'],
                'first_name': row['first_name'],
                'last_name': row['last_name'],
                'hire_date': hire_date_val if hire_date_val else (date.today() - timedelta(days=row['tenure_months'] * 30)),
                'current_job_start_date': job_start_val if job_start_val else (date.today() - timedelta(days=row['months_in_current_role'] * 30))
            })
        
        return employees_list
    except Exception as e:
        print(f"   ⚠️ Warning: Could not extract employees list from DataFrame: {e}")
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
    
    # Extract employees list from DataFrame for use in dependent datasets
    employees_list_for_dependencies = _extract_employees_list_from_df(employees_df)
    if employees_list_for_dependencies:
        print(f"   ℹ️ Extracted {len(employees_list_for_dependencies):,} employees for dependent dataset generation")
    else:
        # If extraction failed, use the generated employees (shouldn't happen, but fallback)
        print("   ⚠️ Could not extract employees list, will generate if needed")
    
    # Try to load performance from data product (with fallback to generated)
    print("\n" + "="*80)
    print("PERFORMANCE REVIEWS DATASET")
    print("="*80)
    performance_df, performance_source = load_performance_from_data_product(
        generated_performance_reviews=None,
        employees=employees_list_for_dependencies
    )
    data_sources['performance'] = performance_source
    
    # Extract performance reviews list for learning generation if fallback needed
    performance_reviews_list = None
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
            for row in performance_df.select(
                'employee_id', 'review_date', 'overall_rating', 'competency_rating',
                'goals_achievement', 'review_id', 'review_period', 'reviewer_id', 'status'
            ).collect()
        ]
    except Exception:
        performance_reviews_list = None
    
    # Try to load learning from data product (with fallback to generated)
    print("\n" + "="*80)
    print("LEARNING RECORDS DATASET")
    print("="*80)
    learning_df, learning_source = load_learning_from_data_product(
        generated_learning_records=None,
        employees=employees_list_for_dependencies,
        performance_reviews=performance_reviews_list
    )
    data_sources['learning'] = learning_source
    
    # Goals and Compensation - always use generated data (no data products available yet)
    print("\n" + "="*80)
    print("GOALS & COMPENSATION DATASETS")
    print("="*80)
    print("📊 Generating goals and compensation data...")
    print("   📦 Data Source: Generated (simulated) - No data products available")
    
    # Generate only what's needed for goals and compensation
    if employees_list_for_dependencies is None or len(employees_list_for_dependencies) == 0:
        print("   → Generating employees for goals/compensation...")
        employees_list_for_dependencies = generate_employees()
        print(f"   ✅ Generated {len(employees_list_for_dependencies):,} employees")
    
    # Generate performance reviews if not already available (needed for goals/compensation)
    if performance_reviews_list is None or len(performance_reviews_list) == 0:
        print("   → Generating performance reviews for goals/compensation...")
        performance_reviews_list = generate_performance_reviews(employees_list_for_dependencies)
        print(f"   ✅ Generated {len(performance_reviews_list):,} performance reviews")
    
    # Generate goals and compensation
    print("   → Generating goals data...")
    goals_data = generate_goals(employees_list_for_dependencies, performance_reviews_list)
    print(f"   ✅ Generated {len(goals_data):,} goals")
    
    print("   → Generating compensation data...")
    compensation_data = generate_compensation(employees_list_for_dependencies, performance_reviews_list)
    print(f"   ✅ Generated {len(compensation_data):,} compensation records")
    
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
    print(f"✅ Goals: {goals_df.count():,} records (Generated)")
    print(f"✅ Compensation: {compensation_df.count():,} records (Generated)")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 DATA SOURCE SUMMARY")
    print("="*80)
    print("Dataset                    | Source          | Records")
    print("-" * 80)
    print(f"Employees                  | {data_sources['employees']:<15} | {employees_df.count():>10,}")
    print(f"Performance Reviews         | {data_sources['performance']:<15} | {performance_df.count():>10,}")
    print(f"Learning Records           | {data_sources['learning']:<15} | {learning_df.count():>10,}")
    print(f"Goals                      | {data_sources['goals']:<15} | {goals_df.count():>10,}")
    print(f"Compensation               | {data_sources['compensation']:<15} | {compensation_df.count():>10,}")
    print("="*80)
    print("✅ All data loaded/generated successfully")
    print("="*80)
    
    return {
        'employees': employees_df,
        'performance': performance_df,
        'learning': learning_df,
        'goals': goals_df,
        'compensation': compensation_df
    }


# Load or generate all data
dataframes = load_or_generate_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Save Data to Unity Catalog Tables

# COMMAND ----------

# Extract dataframes from the loaded/generated data
employees_df = dataframes['employees']
performance_df = dataframes['performance']
learning_df = dataframes['learning']
goals_df = dataframes['goals']
compensation_df = dataframes['compensation']

# Ensure consistent numeric types to avoid merge conflicts
# Handle potential None values and ensure proper types
# Use try_cast to safely handle malformed values
performance_df = performance_df.withColumn("overall_rating", F.coalesce(F.expr("try_cast(overall_rating as double)"), F.lit(3.5))) \
                                .withColumn("competency_rating", F.coalesce(F.expr("try_cast(competency_rating as double)"), F.lit(3.5))) \
                                .withColumn("goals_achievement", F.coalesce(F.expr("try_cast(goals_achievement as int)"), F.lit(0)))

goals_df = goals_df.withColumn("achievement_percentage", F.coalesce(F.expr("try_cast(achievement_percentage as int)"), F.lit(0))) \
                   .withColumn("weight", F.coalesce(F.expr("try_cast(weight as int)"), F.lit(0)))

compensation_df = compensation_df.withColumn("base_salary", F.coalesce(F.expr("try_cast(base_salary as int)"), F.lit(0))) \
                                 .withColumn("bonus_target_pct", F.coalesce(F.expr("try_cast(bonus_target_pct as int)"), F.lit(0))) \
                                 .withColumn("equity_value", F.coalesce(F.expr("try_cast(equity_value as int)"), F.lit(0)))

# Ensure consistent numeric types for employees (already handled in transformation, but ensure types are correct)
# Use try_cast to safely handle malformed values
employees_df = employees_df.withColumn("age", F.coalesce(F.expr("try_cast(age as int)"), F.lit(0))) \
                           .withColumn("base_salary", F.coalesce(F.expr("try_cast(base_salary as int)"), F.lit(0))) \
                           .withColumn("job_level", F.coalesce(F.expr("try_cast(job_level as int)"), F.lit(0))) \
                           .withColumn("tenure_months", F.coalesce(F.expr("try_cast(tenure_months as int)"), F.lit(0))) \
                           .withColumn("months_in_current_role", F.coalesce(F.expr("try_cast(months_in_current_role as int)"), F.lit(0))) \
                           .withColumn("first_name", F.coalesce(F.col("first_name"), F.lit("Unknown"))) \
                           .withColumn("last_name", F.coalesce(F.col("last_name"), F.lit("Unknown")))

# Save as Unity Catalog tables (Delta tables)
table_names = {
    'employees': employees_df,
    'performance': performance_df,
    'learning': learning_df,  # From SAP SuccessFactors Data Product (or generated if data product unavailable)
    'goals': goals_df,
    'compensation': compensation_df
}

for table_name, df in table_names.items():
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    df.write.mode("overwrite").saveAsTable(full_table_name)
    print(f"✅ Created table: {full_table_name} ({df.count():,} rows)")

print(f"\n✅ All data saved to Unity Catalog: {catalog_name}.{schema_name}")

# Display summary
displayHTML(f"""
<div style="background: linear-gradient(135deg, #0052CC 0%, #0070F2 100%); 
            padding: 25px; border-radius: 15px; color: white; margin: 20px 0;">
    <h2 style="text-align: center; margin-bottom: 20px;">✅ Data Generation Complete</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Employees</h3>
            <p style="font-size: 24px; margin: 10px 0;">{employees_df.count():,}</p>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Performance Reviews</h3>
            <p style="font-size: 24px; margin: 10px 0;">{performance_df.count():,}</p>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Learning Records</h3>
            <p style="font-size: 24px; margin: 10px 0;">{learning_df.count():,}</p>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Goals</h3>
            <p style="font-size: 24px; margin: 10px 0;">{goals_df.count():,}</p>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="margin: 0; color: #FFD93D;">Compensation</h3>
            <p style="font-size: 24px; margin: 10px 0;">{compensation_df.count():,}</p>
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

