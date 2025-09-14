# app/app.py
"""
Streamlit app for SmartPremium - robust import + prediction UI.

Notes:
- This file adds the project root to sys.path so `import src` works reliably
  when Streamlit runs the app.
- The app expects a trained sklearn Pipeline saved to models/pipeline.pkl.
  If it's missing, the UI will show instructions.
"""

import os
import sys
from pathlib import Path

# === Make sure project root is on sys.path so "src" can be imported ===
try:
    # app.py is in <repo>/app/app.py; parents[1] -> repo root
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except Exception:
    PROJECT_ROOT = Path.cwd()

PROJECT_ROOT = str(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === Imports (streamlit must be imported before any `st.*` usage) ===
import streamlit as st

# Import our local helper which loads the pipeline
# src/predict.py should contain load_pipeline() that uses joblib to load models/pipeline.pkl
try:
    from src.predict import load_pipeline
except Exception as e:
    # If import fails, show message in the app and stop further execution.
    st.title("SmartPremium — Insurance Premium Predictor")
    st.error(
        "Internal import error: couldn't import project modules from `src`.\n\n"
        "Make sure you are running Streamlit from the project root and that "
        "`src/__init__.py` exists. Exception:\n\n" + repr(e)
    )
    st.stop()

# === Configuration ===
MODEL_PATH = os.environ.get("SMARTPREMIUM_MODEL_PATH", "models/pipeline.pkl")

st.set_page_config(page_title="SmartPremium", page_icon="💰", layout="centered")
st.title("SmartPremium — Insurance Premium Predictor")
st.markdown("Enter customer & policy details below and click **Predict** to get an estimate.")

# === Try to load pipeline (if present) ===
pipeline = None
if not os.path.exists(MODEL_PATH):
    st.warning(
        f"Saved model not found at `{MODEL_PATH}`.\n\n"
        "Please run the training script to create the model (example):\n\n"
        "`python -m src.train --data data/insurance.csv --out models`"
    )
    # We don't `st.stop()` here because we still want users to see the input form,
    # or possibly upload a model later. But prediction will be disabled until model exists.
else:
    try:
        pipeline = load_pipeline(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load saved model pipeline at `{MODEL_PATH}`:\n\n{e}")
        pipeline = None

# === UI: Input form ===
with st.form("predict_form"):
    st.subheader("Customer & policy details")

    # Numeric inputs
    age = st.number_input("Age", min_value=16, max_value=120, value=30)
    annual_income = st.number_input("Annual Income (INR)", min_value=0, value=300000)
    number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=0)
    health_score = st.number_input("Health Score (0-100)", min_value=0, max_value=100, value=75)
    previous_claims = st.number_input("Previous Claims", min_value=0, value=0)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=2)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    insurance_duration = st.number_input("Insurance Duration (years)", min_value=0, value=1)

    # Categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])
    exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

    # Text / date-like fields
    policy_start_date = st.text_input("Policy Start Date (YYYY-MM-DD)", value="2024-01-01")
    customer_feedback = st.text_input("Customer Feedback (optional)", value="")

    submit_button = st.form_submit_button(label="Predict Premium")

# === Prediction logic ===
def build_sample_dict():
    """Return a single-row dict matching the dataset columns used in training."""
    return {
        "Age": age,
        "Gender": gender,
        "Annual Income": annual_income,
        "Marital Status": marital_status,
        "Number of Dependents": number_of_dependents,
        "Education Level": education_level,
        "Occupation": occupation,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Policy Start Date": policy_start_date,
        "Customer Feedback": customer_feedback,
        "Smoking Status": smoking_status,
        "Exercise Frequency": exercise_frequency,
        "Property Type": property_type,
    }

if submit_button:
    sample = build_sample_dict()

    if pipeline is None:
        st.error(
            "No trained model available to make predictions. "
            "Please train and save the model first (see message above)."
        )
    else:
        # Run prediction inside try/except to show friendly error messages
        try:
            import pandas as pd

            df_sample = pd.DataFrame([sample])
            # If the pipeline expects certain dtypes, DataFrame creation will use sensible defaults.
            pred = pipeline.predict(df_sample)
            # Many regressors return array-like; convert first element to float for display.
            if hasattr(pred, "__len__") and len(pred) > 0:
                est = float(pred[0])
                st.success(f"Estimated Premium: ₹{est:,.2f}")
                st.info("This is a model-based estimate for demonstration purposes.")
            else:
                st.error("Model returned an unexpected prediction format.")
        except Exception as e:
            st.error(
                "Prediction failed with an exception. This can happen if the model expects "
                "different feature names/types than provided.\n\n"
                f"Error: {e}"
            )

# === Footer / tips ===
st.markdown("---")
st.markdown(
    "Tips:\n"
    "- If prediction fails, check that the model pipeline (preprocessor + estimator) was "
    "trained using the same column names as the fields above.\n"
    "- To (re)train the model locally run: `python -m src.train --data data/insurance.csv --out models`."
)
