# Insurance Fraud Detection

## Project Overview

This repository contains a small end-to-end project for detecting insurance claim fraud. It includes data preprocessing, exploratory data analysis (visualizations), model training and evaluation, and a saved model artifact for inference.

## Notebooks
- `notebooks/data_preprocessing.ipynb` — Load and clean the raw dataset, convert date columns, create feature-engineered columns (e.g., `claim_delay_days`, `vehicle_age`), drop unneeded columns, and save the cleaned data to `data/insurance_cleaned.csv`.
- `notebooks/visualization.ipynb` — Exploratory Data Analysis (EDA) including univariate and bivariate plots, histograms, and a correlation heatmap to understand relationships between features and the target.
- `notebooks/model_prediction.ipynb` — Encode categorical features, split data, scale where appropriate, train multiple classifiers, evaluate them (accuracy, precision, recall, F1), plot ROC/AUC, inspect feature importances, and save the final model.

## Data
- Raw/conversion: `data/insurance_data.csv` (if present)
- Cleaned: `data/insurance_cleaned.csv` — produced by the preprocessing notebook and consumed by modeling/visualization notebooks.

## Models & Results
- Models trained: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- Summary: During experiments in `notebooks/model_prediction.ipynb`, ensemble models (Random Forest and XGBoost) produced the strongest performance for this dataset. Exact per-model metrics (Accuracy, Precision, Recall, F1) are printed by the notebook as a `results_df` table — open and run the notebook to reproduce numeric results and visualizations (confusion matrix, ROC curve, feature importance plots).
- Saved artifact: `fraud_model.pkl` — the trained Random Forest model saved by the model notebook for later inference.

## Reproduce the analysis
1. Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run notebooks in this order (recommended):
   - `notebooks/data_preprocessing.ipynb` (produces `data/insurance_cleaned.csv`)
   - `notebooks/visualization.ipynb` (EDA)
   - `notebooks/model_prediction.ipynb` (train & evaluate models, saves `fraud_model.pkl`)

3. Open `notebooks/model_prediction.ipynb` and inspect the `results_df` table to see the recorded evaluation metrics for each model.

## Notes
- The model notebook may install XGBoost inline if it is not present. If you prefer, install `xgboost` ahead of time.
- The notebooks use `random_state=42` for reproducibility and `stratify=y` during train/test split to preserve class balance.

## License & Contributing
- Feel free to fork and improve this project. Add issues or pull requests with enhancements, additional preprocessing, hyperparameter tuning, or deployment examples.

---
