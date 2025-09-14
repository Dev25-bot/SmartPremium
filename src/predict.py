# src/predict.py
"""
Prediction helper: load saved pipeline and predict for single samples or DataFrames.
"""
import joblib
import pandas as pd
from typing import Dict, Any

def load_pipeline(path: str = "models/pipeline.pkl"):
    pipeline = joblib.load(path)
    return pipeline

def predict_single(sample: Dict[str, Any], pipeline_path: str = "models/pipeline.pkl"):
    """
    sample: dict mapping feature->value
    returns: predicted numeric premium
    """
    pipeline = load_pipeline(pipeline_path)
    df = pd.DataFrame([sample])
    pred = pipeline.predict(df)
    return float(pred[0])

if __name__ == "__main__":
    # quick manual test
    example = {
        "Age": 35,
        "Gender": "Male",
        "Annual Income": 500000,
        "Marital Status": "Single",
        "Number of Dependents": 0,
        "Education Level": "Bachelor's",
        "Occupation": "Employed",
        "Health Score": 80,
        "Location": "Urban",
        "Policy Type": "Comprehensive",
        "Previous Claims": 0,
        "Vehicle Age": 3,
        "Credit Score": 700,
        "Insurance Duration": 1,
        "Policy Start Date": "2023-01-01",
        "Customer Feedback": "",
        "Smoking Status": "No",
        "Exercise Frequency": "Weekly",
        "Property Type": "Apartment"
    }
    print(predict_single(example))
