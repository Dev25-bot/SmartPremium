"""
Model definitions, hyperparameter search spaces and the full ML pipeline.

Every model is wrapped as:
    TransformedTargetRegressor(log1p / expm1)
        Pipeline([FeatureEngineer, preprocessor, model])
Training on log(1 + premium) handles the skewed target and directly optimises RMSLE,
the main evaluation metric.
"""
import numpy as np
from scipy.stats import randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.preprocessing import FeatureEngineer, build_preprocessor

try:
    from xgboost import XGBRegressor
except Exception:  # optional dependency
    XGBRegressor = None


def get_model_dict():
    """Candidate models.

    The baseline ignores all features and always predicts the mean of log(1 + premium), which is the
    best possible constant prediction for RMSLE. A useful model must beat it.
    """
    models = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                         random_state=42, n_jobs=-1)
    return models


# RandomizedSearchCV spaces (keys refer to the step names inside build_pipeline)
PARAM_DISTRIBUTIONS = {
    "decision_tree": {
        "regressor__model__max_depth": randint(3, 15),
        "regressor__model__min_samples_leaf": randint(20, 500),
    },
    "random_forest": {
        "regressor__model__n_estimators": randint(50, 200),
        "regressor__model__max_depth": randint(4, 14),
        "regressor__model__min_samples_leaf": randint(20, 300),
        "regressor__model__max_features": uniform(0.3, 0.6),
    },
    "xgboost": {
        "regressor__model__n_estimators": randint(100, 600),
        "regressor__model__max_depth": randint(2, 8),
        "regressor__model__learning_rate": uniform(0.01, 0.15),
        "regressor__model__subsample": uniform(0.6, 0.4),
        "regressor__model__colsample_bytree": uniform(0.5, 0.5),
        "regressor__model__min_child_weight": randint(1, 50),
        "regressor__model__reg_lambda": uniform(0.5, 10),
    },
}


def build_pipeline(model) -> TransformedTargetRegressor:
    """Feature engineering -> preprocessing -> model, trained on log1p(target)."""
    preprocessor, _, _ = build_preprocessor()
    pipe = Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
