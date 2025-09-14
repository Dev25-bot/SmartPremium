# src/train.py
"""
Training orchestration: builds pipeline, performs hyperparameter search,
logs to MLflow, and saves best pipeline (preprocessor + model).
"""
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_log_error
from scipy.stats import randint, uniform

from src.data import load_data
from src.preprocessing import build_preprocessor
from src.models import get_model_dict

MLFLOW_EXPERIMENT_NAME = "SmartPremium"

def rmsle(y_true, y_pred):
    # safe RMSLE: clip negatives
    y_pred_adj = np.maximum(0, y_pred)
    y_true_adj = np.maximum(0, y_true)
    return np.sqrt(mean_squared_log_error(y_true_adj, y_pred_adj))

def evaluate_model(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "rmsle": rmsle(y_true, y_pred)
    }

def train_pipeline(csv_path: str,
                   target_col: str = "Premium Amount",
                   test_size: float = 0.2,
                   random_state: int = 42,
                   output_dir: str = "models"):
    """
    Train models, run randomized search for best hyperparams on chosen models,
    pick best pipeline by RMSE and save it.
    """
    X, y = load_data(csv_path, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    models = get_model_dict()

    # parameter grids for RandomizedSearchCV
    param_distributions = {
        "random_forest": {
            "model__n_estimators": randint(50, 300),
            "model__max_depth": randint(3, 20),
            "model__min_samples_split": randint(2, 10),
        },
        "decision_tree": {
            "model__max_depth": randint(2, 20),
            "model__min_samples_split": randint(2, 10),
        },
        "xgboost": {
            "model__n_estimators": randint(50, 500),
            "model__max_depth": randint(2, 12),
            "model__learning_rate": uniform(0.01, 0.4),
        },
        "linear": {},  # no hyperparams here
    }

    os.makedirs(output_dir, exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_overall = {"rmse": float("inf")}
    best_pipeline_path = None

    for name, model in models.items():
        print(f"Training candidate: {name}")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        param_dist = param_distributions.get(name, {})
        if param_dist:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=20,
                cv=3,
                scoring="neg_root_mean_squared_error",
                verbose=1,
                n_jobs=-1,
                random_state=random_state,
            )
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
            params_used = search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_est = pipeline
            params_used = {}

        # Evaluate
        preds = best_est.predict(X_test)
        metrics = evaluate_model(y_test, preds)
        print(f"{name} metrics: {metrics}")

        # Log to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model_name": name, **{k: float(v) if hasattr(v, "item") else v for k,v in params_used.items()}})
            mlflow.log_metrics(metrics)
            # log model
            mlflow.sklearn.log_model(best_est, artifact_path="model")

        # If best, save pipeline pickle
        if metrics["rmse"] < best_overall["rmse"]:
            best_overall = metrics
            best_overall["model_name"] = name
            path = os.path.join(output_dir, "pipeline.pkl")
            joblib.dump(best_est, path)
            best_pipeline_path = path

    print("Best model:", best_overall)
    return best_pipeline_path, best_overall

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SmartPremium models")
    parser.add_argument("--data", type=str, default="data/insurance.csv", help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="Premium Amount", help="Name of target column")
    parser.add_argument("--out", type=str, default="models", help="Output directory for models")
    args = parser.parse_args()

    train_pipeline(csv_path=args.data, target_col=args.target, output_dir=args.out)
