# src/train.py
"""
Training orchestration.

For each candidate model (baseline, Linear Regression, Decision Tree, Random Forest, XGBoost):
  1. hyperparameter search with RandomizedSearchCV (3-fold CV, on a training subsample for speed)
  2. refit the best configuration on the full 80% training split
  3. evaluate on the 20% hold-out split: RMSLE, RMSE, MAE, R²
  4. log parameters, metrics and the fitted pipeline to MLflow
The pipeline with the lowest test RMSLE is saved to models/pipeline.pkl together with
models/model_info.json, and the comparison table to reports/model_comparison.csv.

Usage:
    python -m src.train                       # full run
    python -m src.train --tune-size 20000     # faster search
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments
"""
import json
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.data import DEFAULT_DATA_PATH, TARGET_COL, load_data
from src.models import PARAM_DISTRIBUTIONS, build_pipeline, get_model_dict

MLFLOW_EXPERIMENT_NAME = "SmartPremium"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def rmsle(y_true, y_pred):
    """Root mean squared logarithmic error (negative predictions are clipped to 0)."""
    return float(np.sqrt(mean_squared_log_error(np.maximum(0, y_true), np.maximum(0, y_pred))))


def evaluate_model(y_true, y_pred):
    return {
        "rmsle": rmsle(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _neg_rmsle(estimator, X, y):
    return -rmsle(y, estimator.predict(X))


def train_pipeline(csv_path: str = DEFAULT_DATA_PATH,
                   target_col: str = TARGET_COL,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   output_dir: str = "models",
                   report_dir: str = "reports",
                   tune_size: int = 40000,
                   n_iter: int = 12,
                   tracking_uri: str = MLFLOW_TRACKING_URI,
                   log_models: bool = True):
    """Train, tune, evaluate and log all candidate models; save the best pipeline."""
    X, y = load_data(csv_path, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}")

    # Hyperparameter search runs on a random subsample of the training split for speed
    if tune_size and tune_size < len(X_train):
        tune_idx = X_train.sample(tune_size, random_state=random_state).index
        X_tune, y_tune = X_train.loc[tune_idx], y_train.loc[tune_idx]
    else:
        X_tune, y_tune = X_train, y_train

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    results, best = [], None
    for name, model in get_model_dict().items():
        start = time.perf_counter()
        pipeline = build_pipeline(model)
        param_dist = PARAM_DISTRIBUTIONS.get(name, {})

        if param_dist:
            search = RandomizedSearchCV(pipeline, param_dist, n_iter=n_iter, cv=3, scoring=_neg_rmsle,
                                        random_state=random_state, n_jobs=1)
            search.fit(X_tune, y_tune)
            best_params = search.best_params_
            cv_rmsle = -search.best_score_
            final = clone(pipeline).set_params(**best_params)
        else:
            best_params, cv_rmsle, final = {}, None, pipeline

        final.fit(X_train, y_train)
        metrics = evaluate_model(y_test, final.predict(X_test))
        elapsed = time.perf_counter() - start
        print(f"{name:18} RMSLE {metrics['rmsle']:.4f}  RMSE {metrics['rmse']:.1f}  "
              f"MAE {metrics['mae']:.1f}  R2 {metrics['r2']:.4f}  ({elapsed:.0f}s)")

        clean_params = {k.replace("regressor__model__", ""): (v.item() if hasattr(v, "item") else v)
                        for k, v in best_params.items()}
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model_name": name, "train_rows": len(X_train), **clean_params})
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            if cv_rmsle is not None:
                mlflow.log_metric("cv_rmsle", cv_rmsle)
            if log_models:
                mlflow.sklearn.log_model(final, name="model", input_example=X_test.head(3))

        results.append({"model": name, **metrics, "cv_rmsle": cv_rmsle,
                        "train_seconds": round(elapsed, 1), "best_params": json.dumps(clean_params)})
        if best is None or metrics["rmsle"] < best[1]["rmsle"]:
            best = (name, metrics, final, clean_params)

    comparison = pd.DataFrame(results).sort_values("rmsle")
    comparison.to_csv(os.path.join(report_dir, "model_comparison.csv"), index=False)

    name, metrics, final, params = best
    pipeline_path = os.path.join(output_dir, "pipeline.pkl")
    joblib.dump(final, pipeline_path)
    info = {
        "best_model": name,
        "test_metrics": metrics,
        "best_params": params,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "target_quantiles": {str(q): float(y_train.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "comparison": comparison.drop(columns=["best_params"]).to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=float)

    print("\n" + comparison.drop(columns=["best_params"]).to_string(index=False))
    print(f"\nBest model: {name} -> saved to {pipeline_path}")
    return pipeline_path, {"model_name": name, **metrics}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SmartPremium models")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--target", default=TARGET_COL, help="Name of target column")
    parser.add_argument("--out", default="models", help="Output directory for the saved pipeline")
    parser.add_argument("--tune-size", type=int, default=40000, help="Rows used for hyperparameter search")
    parser.add_argument("--n-iter", type=int, default=12, help="Random-search iterations per model")
    args = parser.parse_args()

    train_pipeline(csv_path=args.data, target_col=args.target, output_dir=args.out,
                   tune_size=args.tune_size, n_iter=args.n_iter)
