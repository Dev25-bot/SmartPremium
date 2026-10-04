"""
Data loading utilities.

The dataset is the "Insurance Premium Prediction" dataset (278,860 rows, 20 columns).
Run `python -m src.download_data` to fetch it into data/insurance_premium_dataset.csv.
"""
from typing import Tuple

import pandas as pd

DEFAULT_DATA_PATH = "data/insurance_premium_dataset.csv"
TARGET_COL = "Premium Amount"


def load_data(csv_path: str = DEFAULT_DATA_PATH,
              target_col: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the dataset and return features (X) and target (y).

    Rows without a target value cannot be used for training, so they are dropped.
    """
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
