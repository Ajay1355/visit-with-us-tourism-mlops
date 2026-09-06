# Visit With Us — Tourism Package Purchase Prediction

## Project Objective
Predict whether a customer will purchase the Wellness Tourism Package (`ProdTaken`).

## MLOps Flow
Data Registration -> Data Preparation -> XGBoost Training/Tuning -> MLflow Tracking -> Model Artifact -> Streamlit

## Repository Structure
- `data/` — source tourism dataset
- `model_building/` — registration, preparation and training scripts
- `artifacts/` — metrics and experiment outputs
- `deployment/` — Streamlit application and serialized model
- `.github/workflows/` — GitHub Actions pipeline

## Model
XGBoost classifier with a scikit-learn preprocessing pipeline.

## Evaluation
F1-score is used as the hyperparameter-tuning objective because the target is imbalanced. Accuracy, precision, recall and ROC-AUC are also reported.

## Deployment
Deploy `deployment/app.py` using Streamlit Community Cloud with the repository root as the working directory.

## Security
No GitHub or ngrok tokens are stored in source code. Use repository/Colab secrets for credentials.
