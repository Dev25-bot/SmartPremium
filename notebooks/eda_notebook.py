# notebooks/eda_notebook.py
"""
A lightweight, linear-script EDA you can open as a notebook or run as a script.
Replace 'data/insurance.csv' with your dataset path.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV = "data/insurance.csv"

def run_eda(csv_path=CSV):
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    display(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nSummary stats (numeric):")
    display(df.describe().T)

    # target distribution
    if "Premium Amount" in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df["Premium Amount"].dropna(), kde=True)
        plt.title("Premium Amount distribution")
        plt.show()

    # correlation heatmap for numeric
    numeric = df.select_dtypes(include=["number"])
    if not numeric.empty:
        plt.figure(figsize=(10,8))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f")
        plt.title("Numeric feature correlations")
        plt.show()

if __name__ == "__main__":
    run_eda()
