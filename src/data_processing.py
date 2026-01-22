import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(filepath):
    """Load raw data from CSV."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean and preprocess the data."""
    df_clean = df.copy()
    
    # Handle missing values
    df_clean = df_clean.dropna()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_income = LabelEncoder()
    le_education = LabelEncoder()
    
    df_clean['gender_encoded'] = le_gender.fit_transform(df_clean['gender'])
    df_clean['income_encoded'] = le_income.fit_transform(df_clean['family_income'])
    df_clean['education_encoded'] = le_education.fit_transform(df_clean['parent_education'])
    
    # Save encoders for later use
    joblib.dump(le_gender, 'models/gender_encoder.pkl')
    joblib.dump(le_income, 'models/income_encoder.pkl')
    joblib.dump(le_education, 'models/education_encoder.pkl')
    
    return df_clean

def save_processed_data(df, filepath):
    """Save processed data."""
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")

if __name__ == "__main__":
    # Load data
    df = load_data('data/raw/students.csv')
    print(f"Loaded {len(df)} records")
    
    # Clean data
    df_clean = clean_data(df)
    print(f"Cleaned data: {len(df_clean)} records")
    
    # Save
    save_processed_data(df_clean, 'data/processed/cleaned_students.csv')