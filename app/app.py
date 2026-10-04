# app/app.py
"""
SmartPremium - Streamlit app for real-time insurance premium estimates.

Run from the project root:  streamlit run app/app.py
Requires a trained pipeline at models/pipeline.pkl (python -m src.train).
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_pipeline  # noqa: E402

MODEL_PATH = Path(os.environ.get("SMARTPREMIUM_MODEL_PATH", PROJECT_ROOT / "models" / "pipeline.pkl"))
INFO_PATH = MODEL_PATH.parent / "model_info.json"
UNKNOWN = "Unknown"
MODEL_LABELS = {
    "baseline_mean": "Baseline (constant)", "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree", "random_forest": "Random Forest", "xgboost": "XGBoost",
}
FEATURE_COLUMNS = [
    "Age", "Gender", "Annual Income", "Marital Status", "Number of Dependents", "Education Level",
    "Occupation", "Health Score", "Location", "Policy Type", "Previous Claims", "Vehicle Age",
    "Credit Score", "Insurance Duration", "Policy Start Date", "Customer Feedback", "Smoking Status",
    "Exercise Frequency", "Property Type",
]

st.set_page_config(page_title="SmartPremium", page_icon="💰", layout="wide")


@st.cache_resource
def get_pipeline():
    return load_pipeline(str(MODEL_PATH))


@st.cache_data
def get_model_info():
    return json.loads(INFO_PATH.read_text()) if INFO_PATH.exists() else None


def optional(value):
    """Map the 'Unknown' choice to a missing value; the pipeline imputes it."""
    return None if value == UNKNOWN else value


if not MODEL_PATH.exists():
    st.title("💰 SmartPremium")
    st.error(f"No trained model found at `{MODEL_PATH}`. Train one first: `python -m src.train`")
    st.stop()

pipeline = get_pipeline()
info = get_model_info()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    if info:
        st.write(f"**{MODEL_LABELS.get(info['best_model'], info['best_model'])}**")
        m = info["test_metrics"]
        c1, c2 = st.columns(2)
        c1.metric("RMSLE", f"{m['rmsle']:.3f}")
        c2.metric("R²", f"{m['r2']:.3f}")
        c1.metric("MAE", f"{m['mae']:,.0f}")
        c2.metric("RMSE", f"{m['rmse']:,.0f}")
        st.caption(f"Evaluated on {info['test_rows']:,} held-out policies.")
    st.divider()
    st.caption("Fields marked *optional* can be left as **Unknown** - the model fills them in "
               "the same way it handled missing values during training.")

st.title("💰 SmartPremium")
st.write("Estimate an insurance premium from customer and policy details.")

quote_tab, batch_tab, perf_tab = st.tabs(["Get a quote", "Batch predictions", "Model performance"])

# ----------------------------------------------------------------------------- single quote
with quote_tab:
    with st.form("quote"):
        st.subheader("Customer")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=18, max_value=64, value=35)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        marital = c3.selectbox("Marital status (optional)", [UNKNOWN, "Single", "Married", "Divorced"])
        education = c1.selectbox("Education level", ["High School", "Bachelor's", "Master's", "PhD"])
        occupation = c2.selectbox("Occupation (optional)", [UNKNOWN, "Employed", "Self-Employed", "Unemployed"])
        dependents = c3.number_input("Number of dependents", min_value=0, max_value=4, value=1)
        income = c1.number_input("Annual income", min_value=0, max_value=150000, value=32000, step=1000)
        credit = c2.number_input("Credit score", min_value=300, max_value=849, value=575)
        location = c3.selectbox("Location", ["Urban", "Suburban", "Rural"])
        property_type = c1.selectbox("Property type", ["House", "Apartment", "Condo"])

        st.subheader("Health & lifestyle")
        c1, c2, c3 = st.columns(3)
        health = c1.slider("Health score", min_value=0.0, max_value=94.0, value=26.5, step=0.5)
        smoking = c2.selectbox("Smoking status", ["No", "Yes"])
        exercise = c3.selectbox("Exercise frequency", ["Daily", "Weekly", "Monthly", "Rarely"])

        st.subheader("Policy")
        c1, c2, c3 = st.columns(3)
        policy_type = c1.selectbox("Policy type", ["Basic", "Comprehensive", "Premium"])
        duration = c2.number_input("Insurance duration (years)", min_value=1, max_value=9, value=5)
        start_date = c3.date_input("Policy start date", value=date(2023, 1, 1),
                                   min_value=date(2019, 1, 1), max_value=date(2025, 12, 31))
        claims = c1.number_input("Previous claims", min_value=0, max_value=9, value=1)
        vehicle_age = c2.number_input("Vehicle age (years)", min_value=0, max_value=19, value=10)
        feedback = c3.selectbox("Customer feedback (optional)", [UNKNOWN, "Poor", "Average", "Good"])

        submitted = st.form_submit_button("Estimate premium", type="primary")

    if submitted:
        sample = pd.DataFrame([{
            "Age": age, "Gender": gender, "Annual Income": income, "Marital Status": optional(marital),
            "Number of Dependents": dependents, "Education Level": education,
            "Occupation": optional(occupation), "Health Score": health, "Location": location,
            "Policy Type": policy_type, "Previous Claims": claims, "Vehicle Age": vehicle_age,
            "Credit Score": credit, "Insurance Duration": duration,
            "Policy Start Date": start_date.isoformat(), "Customer Feedback": optional(feedback),
            "Smoking Status": smoking, "Exercise Frequency": exercise, "Property Type": property_type,
        }])
        estimate = max(0.0, float(pipeline.predict(sample)[0]))
        st.metric("Estimated premium", f"{estimate:,.2f}")
        if info:
            q = info["target_quantiles"]
            st.caption(f"For reference, half of all premiums in the data fall between "
                       f"{q['0.25']:,.0f} and {q['0.75']:,.0f} (median {q['0.5']:,.0f}).")
        st.info("This is a model-based estimate for demonstration purposes. See *Model performance* "
                "for how accurate the model is on this dataset.")

# ----------------------------------------------------------------------------- batch
with batch_tab:
    st.write("Upload a CSV with the dataset's columns (missing values are fine) to price many policies at once.")
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
        else:
            out = df.copy()
            out["Predicted Premium"] = pipeline.predict(df[FEATURE_COLUMNS]).clip(min=0).round(2)
            st.dataframe(out.head(100), width="stretch")
            st.caption(f"{len(out):,} rows priced (showing the first 100).")
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")

# ----------------------------------------------------------------------------- performance
with perf_tab:
    if not info:
        st.info("Run `python -m src.train` to generate the model comparison.")
    else:
        comp = pd.DataFrame(info["comparison"])
        comp["model"] = comp["model"].map(lambda n: MODEL_LABELS.get(n, n))
        st.subheader("Model comparison (20% hold-out set)")
        st.dataframe(
            comp[["model", "rmsle", "rmse", "mae", "r2", "cv_rmsle", "train_seconds"]].rename(columns={
                "model": "Model", "rmsle": "RMSLE", "rmse": "RMSE", "mae": "MAE", "r2": "R²",
                "cv_rmsle": "CV RMSLE", "train_seconds": "Train time (s)"}).style.format(precision=4),
            width="stretch", hide_index=True,
        )
        base = next((r for r in info["comparison"] if r["model"] == "baseline_mean"), None)
        best = info["test_metrics"]
        if base:
            gain = 100 * (base["rmsle"] - best["rmsle"]) / base["rmsle"]
            st.markdown(
                f"**What this shows:** the best model improves RMSLE by only **{gain:.3f}%** over a baseline that "
                "ignores every input and always predicts the same typical premium. In this (synthetic) dataset the "
                "premium is essentially independent of every customer and policy feature, so no model can predict it "
                "better than the typical value. R² is negative because the models minimise RMSLE (log scale), which "
                "pulls predictions below the average premium. The notebook documents this analysis in detail."
            )
