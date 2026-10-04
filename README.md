# SmartPremium: Predicting Insurance Costs with Machine Learning

End-to-end regression project: EDA → data cleaning & feature engineering → Linear Regression, Decision Tree,
Random Forest and XGBoost with hyperparameter tuning → MLflow experiment tracking → Streamlit app for real-time
premium estimates.

| | |
|---|---|
| Dataset | Insurance Premium Prediction – 278,860 policies, 20 columns, target `Premium Amount` |
| Data issues handled | missing values (up to 30% per column), text dates, text feedback, skewed income & target, claim outliers |
| Models | baseline, Linear Regression, Decision Tree, Random Forest, XGBoost (RandomizedSearchCV) |
| Metrics | RMSLE (primary), RMSE, MAE, R² on a 20% hold-out set |
| Tracking | MLflow (`sqlite:///mlflow.db`, one run per model with params, metrics and the fitted pipeline) |
| App | Streamlit: single quote, batch CSV predictions, model performance |

## Results (55,404 held-out policies)

| Model | RMSLE | RMSE | MAE | R² |
|---|---|---|---|---|
| **XGBoost** (deployed) | **1.2559** | 1002.4 | 674.0 | −0.206 |
| Random Forest | 1.2559 | 1002.4 | 674.0 | −0.206 |
| Baseline (same prediction for everyone) | 1.2560 | 1002.4 | 674.0 | −0.206 |
| Linear Regression | 1.2561 | 1002.5 | 674.1 | −0.207 |
| Decision Tree | 1.2566 | 1002.6 | 674.3 | −0.207 |

**Key finding: in this dataset the premium is not predictable from the features.**
Every feature's correlation with the premium is below 0.004, premiums are distributed identically across policy
types, smoking status, locations and every other category, mutual information is ~0, and an XGBoost model scores
*the same* on a randomly shuffled target as on the real one. The tuned models therefore match – but cannot beat – a
baseline that predicts the same value for every customer. The negative R² comes from optimising RMSLE (log scale),
which pulls predictions towards the geometric mean (~550) rather than the arithmetic mean (~966); a model trained on
the raw premium reaches R² ≈ 0. See `notebooks/SmartPremium.ipynb` for the full analysis.

## Project structure

```text
SmartPremium/
├── notebooks/
│   ├── SmartPremium.ipynb    # main deliverable: EDA, preprocessing, models, MLflow, evaluation, conclusions
│   └── eda_notebook.py       # quick command-line EDA summary
├── src/
│   ├── download_data.py      # fetches the dataset into data/insurance_premium_dataset.csv
│   ├── data.py               # loading (drops rows without a target)
│   ├── preprocessing.py      # FeatureEngineer + imputation / scaling / one-hot ColumnTransformer
│   ├── models.py             # candidate models, search spaces, full pipeline (log-target)
│   ├── train.py              # tuning, evaluation, MLflow logging, saves the best pipeline
│   └── predict.py            # load the pipeline and predict
├── app/app.py                # Streamlit app
├── models/
│   ├── pipeline.pkl          # deployed pipeline (feature engineering + preprocessing + XGBoost)
│   └── model_info.json       # metrics, best params, comparison
├── reports/model_comparison.csv
├── mlflow.db, mlruns/        # MLflow tracking store and logged models
├── tests/                    # pytest unit tests
├── quick_train.py            # fast single-model smoke test
├── Dockerfile                # container for the Streamlit app
└── .github/workflows/ci.yml  # GitHub Actions: tests + flake8
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.download_data          # ~10 MB download -> data/insurance_premium_dataset.csv
```

macOS note: XGBoost needs OpenMP (`brew install libomp`).

## Usage

```bash
python -m src.train                                  # train + tune all models, log to MLflow (~3 min)
mlflow ui --backend-store-uri sqlite:///mlflow.db    # compare runs at http://127.0.0.1:5000
streamlit run app/app.py                             # web app at http://localhost:8501
jupyter notebook notebooks/SmartPremium.ipynb        # the analysis notebook
pytest -q                                            # unit tests
```

## Pipeline

```text
TransformedTargetRegressor(log1p / expm1)          # trains on log premium -> optimises RMSLE
└── Pipeline
    ├── FeatureEngineer     policy_year / month / dayofweek / age_days from the text date,
    │                       feedback_score (Poor 0, Average 1, Good 2), log_annual_income,
    │                       income_per_dependent, n_missing, Previous Claims capped at 5
    ├── ColumnTransformer   numeric: median imputation + StandardScaler
    │                       categorical: mode imputation + one-hot encoding
    └── model               LinearRegression | DecisionTree | RandomForest | XGBoost
```

Hyperparameters are tuned with `RandomizedSearchCV` (12 candidates × 3-fold CV, RMSLE scoring) on a 40,000-row
training sample, then refitted on the full 80% training split and evaluated once on the 20% hold-out split.

## Deployment

- **Streamlit Community Cloud:** push this repo to GitHub → sign in at https://share.streamlit.io → *New app* →
  select the repo, branch `main`, main file `app/app.py`. The app only needs `models/pipeline.pkl` and
  `models/model_info.json`, which are committed.
- **Docker:** `docker build -t smartpremium . && docker run -p 8501:8501 smartpremium`

## Data

`data/insurance.csv` is a small 300-row placeholder file from early development (not used for training). The real dataset is the public
"Insurance Premium Prediction" dataset (synthetic, for educational use) referenced in the project brief.
