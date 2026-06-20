# Insurance Fraud Detection

## Project Overview

This is an end-to-end machine learning project for **detecting fraudulent insurance claims**. The project encompasses data preprocessing, exploratory data analysis, model training with multiple classifiers, performance evaluation, and a saved model artifact ready for deployment.

## Objectives

- **Clean and prepare** raw insurance data for analysis
- **Explore and visualize** relationships between features and fraud indicators
- **Train and compare** multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- **Evaluate models** using standard metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Identify important features** contributing to fraud prediction
- **Save the best model** for production inference

## Project Structure

```
insurance-fraud-detection/
├── data/
│   ├── insurance_data.csv          # Original converted dataset
│   └── insurance_cleaned.csv       # Cleaned dataset (output from preprocessing)
├── notebooks/
│   ├── data_preprocessing.ipynb    # Step 1: Data cleaning & feature engineering
│   ├── visualization.ipynb         # Step 2: EDA and data insights
│   └── model_prediction.ipynb      # Step 3: Model training & evaluation
├── fraud_model.pkl                 # Saved Random Forest model
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Notebooks Overview

### 1. Data Preprocessing (`notebooks/data_preprocessing.ipynb`)
**Purpose:** Load, clean, and engineer features from raw data.

**Key Steps:**
- Load data from Excel/CSV format
- Inspect dataset shape, data types, and missing values
- Handle missing values in `authorities_contacted` column (filled with 'None')
- Convert date columns (`policy_bind_date`, `incident_date`) to datetime format
- **Feature Engineering:**
  - `claim_delay_days` = days between policy binding and incident
  - `vehicle_age` = 2025 − vehicle manufacture year
- Remove low-variance columns: `policy_number`, `insured_zip`, `incident_location`
- Check for duplicates
- Export cleaned data to `data/insurance_cleaned.csv`

**Output:** `data/insurance_cleaned.csv` — the foundation for all subsequent analysis.

---

### 2. Visualization (`notebooks/visualization.ipynb`)
**Purpose:** Understand data patterns through exploratory data analysis (EDA).

**Analysis Performed:**
- **Univariate Analysis:**
  - Fraud distribution (count plot)
  - Incident severity distribution
  - Total claim amount distribution (histogram)
- **Bivariate Analysis:**
  - Incident severity vs. fraud reported
  - Incident type vs. fraud reported
  - Authorities contacted vs. fraud reported
- **Correlation Heatmap:** Identify relationships between numerical features
- **Key Insight:** Higher injury/property/vehicle claims correlate with higher total claim amounts

**Output:** Visual insights to guide feature selection and model design.

---

### 3. Model Prediction (`notebooks/model_prediction.ipynb`)
**Purpose:** Train, evaluate, and compare multiple machine learning classifiers.

**Preprocessing for Modeling:**
- Encode categorical variables using `LabelEncoder`
- Split data: 80% training, 20% testing (with stratification to preserve class balance)
- Feature scaling: Applied only to Logistic Regression using `StandardScaler`
  - Optimization-based models (LR) are sensitive to feature magnitude
  - Tree-based models (DT, RF, XGBoost) are scale-invariant

**Models Trained:**

1. **Logistic Regression**
   - Baseline linear classifier for binary classification
   - Uses scaled features (standardized to mean=0, std=1)
   - Max iterations: 10,000 for convergence

2. **Decision Tree**
   - Captures nonlinear relationships between features and target
   - Inherently scale-invariant

3. **Random Forest**
   - Ensemble of 200 decision trees
   - Combines multiple trees for improved robustness and performance

4. **XGBoost**
   - Gradient boosting algorithm
   - Often achieves state-of-the-art performance on classification tasks

**Evaluation Metrics:**
- **Accuracy** — Overall correctness
- **Precision** — True positives / (true positives + false positives)
- **Recall** — True positives / (true positives + false negatives)
- **F1-Score** — Harmonic mean of precision and recall

**Visualizations Generated:**
- **Confusion Matrix:** Shows TP, TN, FP, FN for Random Forest
- **ROC Curve & AUC:** Illustrates trade-off between TPR and FPR
- **Feature Importance Plot:** Top 15 features driving predictions in Random Forest

**Results Summary:**
During model evaluation, **Random Forest and XGBoost** demonstrated superior performance compared to Logistic Regression and Decision Tree. Run the notebook to see the exact `results_df` table with per-model metrics.

**Saved Artifact:**
- `fraud_model.pkl` — The trained Random Forest model, serialized using `joblib` for deployment and future inference.

---

## Data Description

- **Dataset:** Insurance claim records with various claim attributes and fraud indicators
- **Currency:** Claim amounts in Indian Rupees
- **Features:** Demographics (age), vehicle info (year, type), incident details (type, severity, date), claim amounts (injury, property, vehicle), policy info, and authorities contacted
- **Target:** `fraud_reported` — Binary label (0 = Not fraudulent, 1 = Fraudulent)
- **Size:** Thousands of records with ~30 features after preprocessing

---

## Quick Start

### Prerequisites
- Python 3.8 or later
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd insurance-fraud-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies typically include: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, openpyxl, joblib

3. **Run notebooks in order:**
   ```
   Notebook 1: notebooks/data_preprocessing.ipynb
   Notebook 2: notebooks/visualization.ipynb
   Notebook 3: notebooks/model_prediction.ipynb
   ```

4. **Review results:**
   - Open `notebooks/model_prediction.ipynb` and inspect the `results_df` table for model performance metrics
   - View confusion matrix, ROC curve, and feature importance plots generated by the notebook

### Model Performance
Exact metrics depend on the data split and model hyperparameters. Run the model notebook to generate a `results_df` DataFrame showing:
- Model names
- Accuracy, Precision, Recall, F1-Score for each model

---

## Key Findings

1. **Data Quality:** The dataset contains minimal missing values (only in `authorities_contacted`), making it suitable for machine learning without extensive imputation.

2. **Feature Relationships:** Claim amounts strongly correlate with one another; incident severity and type show meaningful associations with fraud.

3. **Model Comparison:** Ensemble methods (Random Forest, XGBoost) outperform linear and single-tree models, suggesting non-linear patterns in fraudulent claims.

4. **Feature Importance:** The Random Forest model identifies the most predictive features, which can guide future fraud detection rules and risk assessment strategies.

---

## Usage Notes

- **Reproducibility:** All notebooks use `random_state=42` and stratified splitting to ensure consistent results across runs.
- **XGBoost:** If not pre-installed, the model notebook will attempt to install it automatically. You can also install manually: `pip install xgboost`.
- **Feature Scaling:** Feature scaling is applied only to Logistic Regression. For tree-based models, original (unscaled) features are used.
- **Runtime:** The model training typically takes a few seconds to a minute, depending on your machine.

---

## Future Enhancements

- Hyperparameter tuning using Grid Search or Bayesian Optimization
- Cross-validation for more robust performance estimates
- Class imbalance handling (SMOTE, class weights)
- Additional feature engineering and selection techniques
- Model explainability (SHAP, permutation importance)
- Production deployment pipeline (Docker, FastAPI, or Flask)

---

## License

This project is open source. Feel free to fork, modify, and improve it. Contributions are welcome!

---

**Last Updated:** June 2026  
**Status:** Complete — Ready for model evaluation and deployment
