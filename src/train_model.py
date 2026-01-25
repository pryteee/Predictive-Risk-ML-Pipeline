import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import joblib

def load_processed_data(filepath):
    """Load processed data."""
    return pd.read_csv(filepath)

def prepare_features(df):
    """Prepare features and target."""
    feature_columns = [
        'age', 'gender_encoded', 'attendance_rate', 'cgpa', 
        'income_encoded', 'study_hours_per_week', 'failed_courses',
        'extracurricular_activities', 'education_encoded', 'distance_from_home'
    ]
    
    X = df[feature_columns]
    y = df['dropped_out']
    
    return X, y

def train_model(X_train, y_train):
    """Train LightGBM model."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return metrics

if __name__ == "__main__":
    # Load data
    df = load_processed_data('data/processed/cleaned_students.csv')
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    joblib.dump(model, 'models/dropout_model.pkl')
    print("\nModel saved to models/dropout_model.pkl")