# SmartPremium

**SmartPremium: Predicting Insurance Costs with Machine Learning**

This repository contains a production-ready implementation for predicting insurance premiums using machine learning. It includes preprocessing, model training, MLflow integration, a Streamlit web app, unit tests, Dockerfile and CI workflow.

## Project structure
(see project tree in repository root)

## Quick start

1. Clone repo and place your dataset CSV into `data/insurance.csv`.
   - The dataset must include a column named `Premium Amount` (target).
2. Create virtual env and install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
