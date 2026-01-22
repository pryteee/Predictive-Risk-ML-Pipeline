"""
Configuration file for Student Dropout Prediction project
"""
from pathlib import Path
import numpy as np

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
TEST_DIR = BASE_DIR / "tests"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, NOTEBOOKS_DIR, TEST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model paths
MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "encoders.pkl"

# Data settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Features
NUMERIC_FEATURES = [
    'age', 
    'previous_gpa', 
    'attendance_rate', 
    'study_hours_per_week', 
    'absences'
]

CATEGORICAL_FEATURES = [
    'gender', 
    'major', 
    'scholarship', 
    'parental_education', 
    'part_time_job'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'dropout'

# LightGBM parameters
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4
API_DEBUG = True

# Data generation settings
N_SAMPLES = 1000
AGE_RANGE = (18, 25)
GPA_RANGE = (1.5, 4.0)
ATTENDANCE_RANGE = (40, 100)
STUDY_HOURS_RANGE = (5, 40)
ABSENCES_RANGE = (0, 30)

CATEGORIES = {
    'gender': ['M', 'F'],
    'major': ['Engineering', 'Business', 'Arts', 'Science', 'Medicine', 'Law'],
    'scholarship': ['Yes', 'No'],
    'parental_education': ['High School', 'Bachelor', 'Master', 'PhD'],
    'part_time_job': ['Yes', 'No']
}