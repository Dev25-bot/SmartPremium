# tests/test_preprocessing.py
import numpy as np
import pandas as pd

from src.preprocessing import FeatureEngineer, build_preprocessor


def test_build_preprocessor_basic():
    df = pd.DataFrame({
        "Age": [25, np.nan, 45],
        "Gender": ["Male", "Female", None],
        "Annual Income": [100000, 200000, 150000],
    })
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    assert "Age" in num_cols
    assert "Annual Income" in num_cols
    assert "Gender" in cat_cols

    transformed = preprocessor.fit(df).transform(df)
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()  # missing values were imputed


def test_feature_engineer_derives_features():
    df = pd.DataFrame({
        "Annual Income": [1000.0, np.nan],
        "Number of Dependents": [1.0, 2.0],
        "Previous Claims": [9.0, 1.0],
        "Policy Start Date": ["2023-01-31 15:21:39.1", "not a date"],
        "Customer Feedback": ["Good", None],
    })
    out = FeatureEngineer().fit(df).transform(df)

    assert "Policy Start Date" not in out and "Customer Feedback" not in out
    assert out.loc[0, "policy_year"] == 2023 and out.loc[0, "policy_month"] == 1
    assert out.loc[0, "policy_age_days"] == 0
    assert np.isnan(out.loc[1, "policy_year"])          # bad dates become missing
    assert out.loc[0, "feedback_score"] == 2
    assert out.loc[0, "Previous Claims"] == 5           # outlier capped
    assert out.loc[0, "income_per_dependent"] == 500
    assert out.loc[1, "n_missing"] == 2
