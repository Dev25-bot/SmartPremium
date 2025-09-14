# tests/test_preprocessing.py
import pandas as pd
import numpy as np
from src.preprocessing import build_preprocessor

def test_build_preprocessor_basic():
    df = pd.DataFrame({
        "Age": [25, np.nan, 45],
        "Gender": ["Male", "Female", None],
        "Annual Income": [100000, 200000, 150000]
    })
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    assert "Age" in num_cols
    assert "Annual Income" in num_cols
    assert "Gender" in cat_cols

    # Fit transform should not crash
    transformer = preprocessor.fit(df)
    transformed = transformer.transform(df)
    assert transformed.shape[0] == 3
