from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import sys
sys.path.append('..')
from src.predict import load_model, predict_dropout

app = FastAPI(
    title="Student Dropout Prediction API",
    description="Predict student dropout risk using machine learning",
    version="1.0.0"
)

# Load model at startup
model, gender_enc, income_enc, education_enc = load_model()

class StudentData(BaseModel):
    age: int = Field(..., ge=16, le=100, description="Student age")
    gender: str = Field(..., pattern="^(M|F)$", description="Gender: M or F")
    attendance_rate: float = Field(..., ge=0.0, le=1.0, description="Attendance rate (0-1)")
    gpa: float = Field(..., ge=0.0, le=4.0, description="GPA (0-4)")
    family_income: str = Field(..., pattern="^(Low|Medium|High)$", description="Family income level")
    study_hours_per_week: int = Field(..., ge=0, le=168, description="Study hours per week")
    failed_courses: int = Field(..., ge=0, description="Number of failed courses")
    extracurricular_activities: int = Field(..., ge=0, description="Number of activities")
    parent_education: str = Field(..., pattern="^(High School|Bachelor|Master|PhD)$", description="Parent education")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in km")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 18,
                "gender": "F",
                "attendance_rate": 0.85,
                "gpa": 3.2,
                "family_income": "Medium",
                "study_hours_per_week": 15,
                "failed_courses": 1,
                "extracurricular_activities": 2,
                "parent_education": "Bachelor",
                "distance_from_home": 10.5
            }
        }

class PredictionResponse(BaseModel):
    dropout_probability: float
    risk_level: str
    recommendation: str

@app.get("/")
def read_root():
    return {
        "message": "Student Dropout Prediction API",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentData):
    try:
        # Convert to dict
        student_dict = student.dict()
        
        # Get prediction
        dropout_prob = predict_dropout(
            student_dict, model, gender_enc, income_enc, education_enc
        )
        
        # Determine risk level
        if dropout_prob < 0.3:
            risk_level = "Low"
            recommendation = "Student is doing well. Continue current support."
        elif dropout_prob < 0.6:
            risk_level = "Medium"
            recommendation = "Monitor student closely. Consider additional academic support."
        else:
            risk_level = "High"
            recommendation = "Immediate intervention recommended. Schedule counseling session."
        
        return PredictionResponse(
            dropout_probability=round(float(dropout_prob), 4),
            risk_level=risk_level,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))