import joblib
import pandas as pd

def load_model():
    """Load trained model and encoders."""
    model = joblib.load('models/dropout_model.pkl')
    gender_encoder = joblib.load('models/gender_encoder.pkl')
    income_encoder = joblib.load('models/income_encoder.pkl')
    education_encoder = joblib.load('models/education_encoder.pkl')
    
    return model, gender_encoder, income_encoder, education_encoder

def predict_dropout(student_data, model, gender_enc, income_enc, education_enc):
    """Predict dropout probability for a student."""
    # Encode categorical features
    student_data['gender_encoded'] = gender_enc.transform([student_data['gender']])[0]
    student_data['income_encoded'] = income_enc.transform([student_data['family_income']])[0]
    student_data['education_encoded'] = education_enc.transform([student_data['parent_education']])[0]
    
    # Prepare features
    features = [[
        student_data['age'],
        student_data['gender_encoded'],
        student_data['attendance_rate'],
        student_data['cgpa'],
        student_data['income_encoded'],
        student_data['study_hours_per_week'],
        student_data['failed_courses'],
        student_data['extracurricular_activities'],
        student_data['education_encoded'],
        student_data['distance_from_home']
    ]]
    
    # Predict
    dropout_probability = model.predict(features)[0]
    
    return dropout_probability