# src/preprocessing.py
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def _make_onehot_encoder():
    """
    Construct OneHotEncoder in a way compatible with both older and newer scikit-learn versions.
    Newer versions expect 'sparse_output'; older ones expect 'sparse'.
    """
    try:
        # try the modern parameter name first
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # fallback for older scikit-learn versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer for numeric and categorical features.
    Returns (preprocessor, numeric_cols, cat_cols).
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_onehot_encoder()),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols),
    ], remainder="drop", verbose_feature_names_out=False)

    return preprocessor, numeric_cols, cat_cols
