from fastapi.testclient import TestClient
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app


client = TestClient(app)


def test_read_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["status"] == "active"


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True


def test_predict_valid_input():
    """Test prediction with valid student data"""
    student_data = {
        "age": 18,
        "gender": "F",
        "attendance_rate": 0.85,
        "cgpa": 7.5,
        "family_income": "Medium",
        "study_hours_per_week": 15,
        "failed_courses": 1,
        "extracurricular_activities": 2,
        "parent_education": "Bachelor",
        "distance_from_home": 10.5
    }
    
    response = client.post("/predict", json=student_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "dropout_probability" in data
    assert "risk_level" in data
    assert "recommendation" in data
    assert 0 <= data["dropout_probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_invalid_gender():
    """Test prediction with invalid gender"""
    student_data = {
        "age": 18,
        "gender": "X",
        "attendance_rate": 0.85,
        "cgpa": 7.5,
        "family_income": "Medium",
        "study_hours_per_week": 15,
        "failed_courses": 1,
        "extracurricular_activities": 2,
        "parent_education": "Bachelor",
        "distance_from_home": 10.5
    }
    
    response = client.post("/predict", json=student_data)
    assert response.status_code == 422  


def test_predict_invalid_gpa():
    """Test prediction with invalid GPA"""
    student_data = {
        "age": 18,
        "gender": "F",
        "attendance_rate": 0.85,
        "cgpa": 7.5, 
        "family_income": "Medium",
        "study_hours_per_week": 15,
        "failed_courses": 1,
        "extracurricular_activities": 2,
        "parent_education": "Bachelor",
        "distance_from_home": 10.5
    }
    
    response = client.post("/predict", json=student_data)
    assert response.status_code == 422


def test_predict_missing_field():
    """Test prediction with missing required field"""
    student_data = {
        "age": 18,
        "gender": "F",
        "cgpa": 7.5,
        "family_income": "Medium",
        "study_hours_per_week": 15,
        "failed_courses": 1,
        "extracurricular_activities": 2,
        "parent_education": "Bachelor",
        "distance_from_home": 10.5
    }
    
    response = client.post("/predict", json=student_data)
    assert response.status_code == 422