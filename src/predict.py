# src/predict.py
"""
Prediction helper: load the saved pipeline and predict for single samples or DataFrames.

The saved pipeline contains feature engineering, preprocessing and the model, so it accepts
raw records with the original dataset columns (missing values allowed).
"""
from functools import lru_cache
from typing import Any, Dict

import joblib
import pandas as pd

DEFAULT_PIPELINE_PATH = "models/pipeline.pkl"


@lru_cache(maxsize=4)
def load_pipeline(path: str = DEFAULT_PIPELINE_PATH):
    return joblib.load(path)


def predict_df(df: pd.DataFrame, pipeline_path: str = DEFAULT_PIPELINE_PATH) -> pd.Series:
    """Predict premiums for a DataFrame of raw records."""
    preds = load_pipeline(pipeline_path).predict(df)
    return pd.Series(preds, index=df.index, name="Predicted Premium").clip(lower=0)


def predict_single(sample: Dict[str, Any], pipeline_path: str = DEFAULT_PIPELINE_PATH) -> float:
    """sample: dict mapping feature -> value. Returns the predicted premium."""
    return float(predict_df(pd.DataFrame([sample]), pipeline_path).iloc[0])


if __name__ == "__main__":
    example = {
        "Age": 35, "Gender": "Male", "Annual Income": 50000, "Marital Status": "Single",
        "Number of Dependents": 0, "Education Level": "Bachelor's", "Occupation": "Employed",
        "Health Score": 25.0, "Location": "Urban", "Policy Type": "Comprehensive", "Previous Claims": 0,
        "Vehicle Age": 3, "Credit Score": 700, "Insurance Duration": 1,
        "Policy Start Date": "2023-01-01", "Customer Feedback": "Good", "Smoking Status": "No",
        "Exercise Frequency": "Weekly", "Property Type": "Apartment",
    }
    print(predict_single(example))
