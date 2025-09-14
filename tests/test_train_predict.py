# tests/test_train_predict.py
import os
import pandas as pd
import numpy as np
from src.train import train_pipeline
from src.predict import predict_single

def make_dummy_dataset(path="data/dummy_insurance.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 200
    df = pd.DataFrame({
        "Age": np.random.randint(20, 70, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Annual Income": np.random.randint(100000, 2000000, n),
        "Marital Status": np.random.choice(["Single", "Married", "Divorced"], n),
        "Number of Dependents": np.random.randint(0,4,n),
        "Education Level": np.random.choice(["High School","Bachelor's","Master's","PhD"], n),
        "Occupation": np.random.choice(["Employed","Self-Employed","Unemployed"], n),
        "Health Score": np.random.randint(30,100,n),
        "Location": np.random.choice(["Urban","Suburban","Rural"], n),
        "Policy Type": np.random.choice(["Basic","Comprehensive","Premium"], n),
        "Previous Claims": np.random.poisson(0.5, n),
        "Vehicle Age": np.random.randint(0,10,n),
        "Credit Score": np.random.randint(300,900,n),
        "Insurance Duration": np.random.randint(0,5,n),
        "Premium Amount": np.random.randint(2000,20000,n),
        "Policy Start Date": "2023-01-01",
        "Customer Feedback": "",
        "Smoking Status": np.random.choice(["Yes","No"], n),
        "Exercise Frequency": np.random.choice(["Daily","Weekly","Monthly","Rarely"], n),
        "Property Type": np.random.choice(["House","Apartment","Condo"], n),
    })
    df.to_csv(path, index=False)
    return path

def test_training_and_prediction(tmp_path):
    csv = make_dummy_dataset(path=str(tmp_path / "dummy_insurance.csv"))
    outdir = str(tmp_path / "models")
    model_path, metrics = train_pipeline(csv_path=csv, output_dir=outdir)
    assert os.path.exists(model_path)
    # make a sample input and run prediction
    sample = {
        "Age": 30,
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
        "Vehicle Age": 2,
        "Credit Score": 700,
        "Insurance Duration": 1,
        "Policy Start Date": "2023-01-01",
        "Customer Feedback": "",
        "Smoking Status": "No",
        "Exercise Frequency": "Weekly",
        "Property Type": "Apartment"
    }
    pred = predict_single(sample, pipeline_path=model_path)
    assert isinstance(pred, float)
