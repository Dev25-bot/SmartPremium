"""
Data loading utilities.
"""
import pandas as pd
from typing import Tuple

def load_data(csv_path: str, target_col: str = "Premium Amount") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from CSV and return features (X) and target (y).
    """
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y