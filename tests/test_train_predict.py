# tests/test_train_predict.py
import json
import os

import numpy as np
import pandas as pd

from src.predict import predict_single
from src.train import train_pipeline


def make_dummy_dataset(path):
    """Small synthetic dataset with the same columns (and messiness) as the real one."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "Age": rng.integers(18, 65, n).astype(float),
        "Gender": rng.choice(["Male", "Female"], n),
        "Annual Income": rng.integers(1000, 150000, n).astype(float),
        "Marital Status": rng.choice(["Single", "Married", "Divorced", None], n),
        "Number of Dependents": rng.integers(0, 5, n).astype(float),
        "Education Level": rng.choice(["High School", "Bachelor's", "Master's", "PhD"], n),
        "Occupation": rng.choice(["Employed", "Self-Employed", "Unemployed", None], n),
        "Health Score": rng.uniform(1, 60, n),
        "Location": rng.choice(["Urban", "Suburban", "Rural"], n),
        "Policy Type": rng.choice(["Basic", "Comprehensive", "Premium"], n),
        "Previous Claims": rng.poisson(1, n).astype(float),
        "Vehicle Age": rng.integers(0, 20, n),
        "Credit Score": rng.integers(300, 850, n).astype(float),
        "Insurance Duration": rng.integers(1, 10, n),
        "Premium Amount": rng.integers(20, 5000, n).astype(float),
        "Policy Start Date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str) + " 15:21:39.1",
        "Customer Feedback": rng.choice(["Poor", "Average", "Good", None], n),
        "Smoking Status": rng.choice(["Yes", "No"], n),
        "Exercise Frequency": rng.choice(["Daily", "Weekly", "Monthly", "Rarely"], n),
        "Property Type": rng.choice(["House", "Apartment", "Condo"], n),
    })
    df.loc[::17, "Annual Income"] = np.nan
    df.loc[::5, "Premium Amount"] = np.nan  # rows without a target are dropped
    df.to_csv(path, index=False)
    return path


SAMPLE = {
    "Age": 30, "Gender": "Male", "Annual Income": 50000, "Marital Status": "Single",
    "Number of Dependents": 0, "Education Level": "Bachelor's", "Occupation": "Employed",
    "Health Score": 25.0, "Location": "Urban", "Policy Type": "Comprehensive", "Previous Claims": 0,
    "Vehicle Age": 2, "Credit Score": 700, "Insurance Duration": 1,
    "Policy Start Date": "2023-01-01", "Customer Feedback": "Good", "Smoking Status": "No",
    "Exercise Frequency": "Weekly", "Property Type": "Apartment",
}


def test_training_and_prediction(tmp_path):
    csv = make_dummy_dataset(str(tmp_path / "dummy_insurance.csv"))
    outdir = str(tmp_path / "models")
    model_path, metrics = train_pipeline(
        csv_path=csv, output_dir=outdir, report_dir=str(tmp_path / "reports"),
        tune_size=0, n_iter=2, tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", log_models=False,
    )
    assert os.path.exists(model_path)
    assert {"rmsle", "rmse", "mae", "r2"} <= set(metrics)

    info = json.load(open(os.path.join(outdir, "model_info.json")))
    assert info["best_model"] == metrics["model_name"]

    pred = predict_single(SAMPLE, pipeline_path=model_path)
    assert isinstance(pred, float) and pred >= 0

    # Missing optional fields must not break prediction
    partial = {**SAMPLE, "Occupation": None, "Customer Feedback": None, "Credit Score": None}
    assert isinstance(predict_single(partial, pipeline_path=model_path), float)
