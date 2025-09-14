# quick_train.py
import os
import joblib
from src.data import load_data
from src.preprocessing import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def quick_train(data_path="data/insurance.csv", out_path="models/pipeline.pkl"):
    print("Loading data from:", data_path)
    X, y = load_data(data_path)
    preprocessor, _, _ = build_preprocessor(X)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
    print("Fitting a LinearRegression pipeline on provided data (quick)...")
    pipe.fit(X, y)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(pipe, out_path)
    print("Saved pipeline to:", out_path)

if __name__ == "__main__":
    quick_train()
