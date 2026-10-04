# src/preprocessing.py
"""
Feature engineering and preprocessing.

FeatureEngineer (custom transformer) cleans the raw columns and adds features:
  - Policy Start Date (text) -> policy_year, policy_month, policy_dayofweek, policy_age_days
  - Customer Feedback (Poor/Average/Good) -> ordinal feedback_score
  - log_annual_income to reduce the strong right skew of Annual Income
  - income_per_dependent
  - n_missing: number of missing fields in the row
  - Previous Claims capped at 5 to limit outliers

build_preprocessor() then imputes (median for numeric, mode for categorical),
scales numeric features and one-hot encodes categorical features.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FEEDBACK_SCORES = {"Poor": 0, "Average": 1, "Good": 2}
CLAIMS_CAP = 5


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean raw insurance records and derive model features (works on raw app input too)."""

    def fit(self, X: pd.DataFrame, y=None):
        dates = pd.to_datetime(X.get("Policy Start Date"), errors="coerce")
        # Reference date for "policy age": the latest start date seen in training
        self.reference_date_ = dates.max() if dates.notna().any() else pd.Timestamp("2024-08-15")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        n_missing = X.isna().sum(axis=1)

        if "Policy Start Date" in X:
            dates = pd.to_datetime(X.pop("Policy Start Date"), errors="coerce")
            X["policy_year"] = dates.dt.year
            X["policy_month"] = dates.dt.month
            X["policy_dayofweek"] = dates.dt.dayofweek
            X["policy_age_days"] = (self.reference_date_ - dates).dt.days

        if "Customer Feedback" in X:
            X["feedback_score"] = X.pop("Customer Feedback").map(FEEDBACK_SCORES)

        if "Annual Income" in X:
            income = pd.to_numeric(X["Annual Income"], errors="coerce")
            X["log_annual_income"] = np.log1p(income)
            if "Number of Dependents" in X:
                X["income_per_dependent"] = income / (pd.to_numeric(X["Number of Dependents"], errors="coerce") + 1)

        if "Previous Claims" in X:
            X["Previous Claims"] = pd.to_numeric(X["Previous Claims"], errors="coerce").clip(upper=CLAIMS_CAP)

        X["n_missing"] = n_missing
        return X


def _none_to_nan(X):
    """Treat None (e.g. empty app inputs) the same as NaN so imputers handle both."""
    X = pd.DataFrame(X).astype(object)
    return X.where(X.notna(), np.nan)


def _make_onehot_encoder():
    """OneHotEncoder compatible with old ('sparse') and new ('sparse_output') scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame = None) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer for numeric and categorical features.

    Columns are selected by dtype when the pipeline is fitted, so the same preprocessor works
    after FeatureEngineer adds columns. If X is given, the numeric and categorical column
    names are also returned (useful for inspection and tests).
    """
    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_include=["object", "category"])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("to_nan", FunctionTransformer(_none_to_nan, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_onehot_encoder()),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_selector),
        ("cat", cat_pipeline, categorical_selector),
    ], remainder="drop", verbose_feature_names_out=False)

    numeric_cols = numeric_selector(X) if X is not None else []
    cat_cols = categorical_selector(X) if X is not None else []
    return preprocessor, numeric_cols, cat_cols
