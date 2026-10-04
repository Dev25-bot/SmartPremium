# quick_train.py
"""Fit a single fast model (Linear Regression pipeline) without tuning or MLflow - handy for a smoke test.

For the full workflow (all models, tuning, MLflow) use: python -m src.train
"""
import os

import joblib
from sklearn.linear_model import LinearRegression

from src.data import DEFAULT_DATA_PATH, load_data
from src.models import build_pipeline


def quick_train(data_path=DEFAULT_DATA_PATH, out_path="models/pipeline_quick.pkl"):
    print("Loading data from:", data_path)
    X, y = load_data(data_path)
    pipe = build_pipeline(LinearRegression())
    print("Fitting a Linear Regression pipeline...")
    pipe.fit(X, y)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(pipe, out_path)
    print("Saved pipeline to:", out_path)


if __name__ == "__main__":
    quick_train()
