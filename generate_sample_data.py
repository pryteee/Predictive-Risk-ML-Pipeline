import pandas as pd
import numpy as np

np.random.seed(42)

n_students = 1000

data = {
    'student_id': range(1, n_students + 1),
    'age': np.random.randint(16, 25, n_students),
    'gender': np.random.choice(['M', 'F'], n_students),
    'attendance_rate': np.random.uniform(0.5, 1.0, n_students),
    'cgpa': np.random.uniform(4.0, 10.0, n_students),
    'family_income': np.random.choice(['Low', 'Medium', 'High'], n_students),
    'study_hours_per_week': np.random.randint(0, 40, n_students),
    'failed_courses': np.random.randint(0, 5, n_students),
    'extracurricular_activities': np.random.randint(0, 5, n_students),
    'parent_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_students),
    'distance_from_home': np.random.uniform(1, 50, n_students),
    'dropped_out': np.random.choice([0, 1], n_students, p=[0.75, 0.25])
}

df = pd.DataFrame(data)
df.to_csv('data/raw/students.csv', index=False)
print("Sample dataset created!")
print(f"File saved to: data/raw/students.csv")
print(f"Total students: {len(df)}")