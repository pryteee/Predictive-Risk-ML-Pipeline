# Predictive-Risk-ML-Pipeline
An end-to-end machine learning pipeline that predicts student dropout risk using LightGBM, deployed as a REST API with FastAPI and Docker.

## About
This project implements a complete machine learning pipeline that:
- Predicts whether a student is at risk of dropping out based on various factors
- Provides risk levels (Low, Medium, High) with actionable recommendations
- Exposes predictions through a REST API
- Runs in a Docker container for easy deployment
  
## Features
- **Data Processing**: Automated data cleaning and feature engineering with Pandas
- **Machine Learning**: LightGBM classifier with optimized hyperparameters
- **REST API**: Fast, production-ready API built with FastAPI
- **Interactive Documentation**: Auto-generated Swagger UI for easy testing
- **Dockerized**: Fully containerized application for consistent deployment
- **Input Validation**: Pydantic models ensure data quality
- **Real-time Predictions**: Instant risk assessment for student profiles
  
## Tech Stack
### Core Technologies
- **Python 3.9**: Primary programming language
- **Pandas**: Data manipulation and cleaning
- **LightGBM**: Gradient boosting framework for ML
- **Scikit-learn**: Model evaluation and preprocessing
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **Docker**: Containerization platform

### Libraries
- **joblib**: Model serialization
- **pydantic**: Data validation
- **numpy**: Numerical computing

## How to run
### Run Locally (Without Docker)

#### Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies
```
pip install -r requirements.txt
```

#### Generate Sample Data
```
python generate_sample_data.py
```
### Output:
Sample dataset created!  
File saved to: data/raw/students.csv  
Total students: 1000  

#### Process the Data
```
python src/data_processing.py
```
#### Output:
Loaded 1000 records  
Cleaned data: 1000 records  
Processed data saved to data/processed/cleaned_students.csv  

#### Train the Model
```
python src/train_model.py
```

#### Output:
Training set: 800 samples  
Test set: 200 samples  
Training model..  
Evaluating model..    

Model Performance:  
accuracy: 0.7450  
precision: 0.6923  
recall: 0.5400  
f1_score: 0.6071  


Model saved to models/dropout_model.pkl  

#### Start the API Server
```
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### With Docker

```
docker-compose up --build
```

Wait 2-3 minutes for build, then open:  
http://localhost:8000/docs  

## Usage
Access the API  
Once the server is running (either locally or with Docker), you can access:  

- **Interactive API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Root Endpoint**: http://localhost:8000/

## Making Predictions  
Using the Interactive Web Interface  

- Open your browser and go to: http://localhost:8000/docs
- Click on POST /predict endpoint
- Click "Try it out" button
- Modify the JSON data with student information
- Click "Execute"
- See the prediction result below!

### Example 
``Input:``

{  
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

``Output:``

{  
  "dropout_probability": 0.2345,  
  "risk_level": "Low",  
  "recommendation": "Student is doing well. Continue current support."  
}  

## Key Learning Outcomes

1. How to build end-to-end ML pipelines
2. Data cleaning and preprocessing with Pandas
3. Training models with LightGBM
4. Creating REST APIs with FastAPI
5. Containerizing applications with Docker
6. Model deployment best practices
7. API documentation and testing
