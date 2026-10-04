# notebooks/eda_notebook.py
"""
Quick command-line EDA summary. The full, explained analysis is in notebooks/SmartPremium.ipynb.

Usage:  python notebooks/eda_notebook.py [path/to/data.csv]
"""
import sys

import numpy as np
import pandas as pd

CSV = "data/insurance_premium_dataset.csv"
TARGET = "Premium Amount"


def run_eda(csv_path=CSV):
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    print("\nMissing values (%):")
    print(df.isna().mean().mul(100).round(2).sort_values(ascending=False).to_string())
    print("\nNumeric summary:")
    print(df.describe().T.round(2).to_string())
    if TARGET in df:
        print(f"\nTarget skew: {df[TARGET].skew():.2f}  (log1p skew: {np.log1p(df[TARGET]).skew():.2f})")
        corr = df.select_dtypes("number").corr(method="spearman")[TARGET].drop(TARGET)
        print("\nSpearman correlation with the target:")
        print(corr.round(4).sort_values().to_string())


if __name__ == "__main__":
    run_eda(sys.argv[1] if len(sys.argv) > 1 else CSV)
