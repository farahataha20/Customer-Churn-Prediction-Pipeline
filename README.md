 # Customer Churn Prediction

A machine learning project for predicting whether a telecom customer is likely to churn. The project uses a **Random Forest Classifier** with a preprocessing pipeline designed to handle missing values, categorical variables, and class imbalance safely.

## Project Objective

The goal is to build a model that predicts the `Churn` status of customers and produces a CSV file containing predictions for the holdout test set.

## Dataset

The notebook uses the Cell2Cell telecom churn dataset. The training data contains customer-level usage, service, demographic, and account-related features.

- Target: `Churn`
- Identifier: `CustomerID`
- Training rows: 51,047
- Training columns: 58
- Target classes: `0 = No Churn`, `1 = Churn`

> The notebook keeps `CustomerID` only for identifying customers in the final output. It is not used as a model feature.

## Workflow

```text
Load Data
    ↓
Initial Inspection
    ↓
Separate Target and CustomerID
    ↓
Train / Validation Split (80% / 20%)
    ↓
Preprocessing
 ┌──────────────────────────────┐
 │ Numerical → Median Imputer   │
 │ Categorical → Mode Imputer   │
 │ Categorical → One-Hot Encode │
 └──────────────────────────────┘
    ↓
Random Forest (class_weight=balanced)
    ↓
Validation Evaluation
 ├── Classification Report
 ├── Confusion Matrix
 └── ROC-AUC / ROC Curve
    ↓
Retrain on Full Training Data
    ↓
Predict Holdout Test Set
    ↓
customer_churn_predictions.csv
```

## Key Improvements

### 1. Missing Values

The original workflow replaced missing values with `0`. The revised version uses a more appropriate strategy:

- Numerical columns → median imputation
- Categorical columns → most frequent value

Importantly, the imputation values are learned from the training data through the preprocessing pipeline.

### 2. Categorical Encoding

The project uses `OneHotEncoder(handle_unknown="ignore")` instead of repeatedly fitting a `LabelEncoder` to individual columns. This avoids inconsistent category mappings and prevents shape problems when validation/test data contains categories that were not present during training.

### 3. Data Preparation

- `Churn` is separated as the target variable.
- `CustomerID` is excluded from model features.
- The labeled training data is split into 80% training and 20% validation.
- `stratify=y` preserves the class distribution across the split.

### 4. Model

The main model is a `RandomForestClassifier` with:

```python
class_weight="balanced"
```

This gives additional importance to the minority churn class during training.

### 5. Evaluation

The validation set is evaluated using:

- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- ROC Curve

The notebook does not hard-code performance numbers because the actual metrics should come from the execution environment.

## Output Files

After running the notebook, it creates:

### `customer_churn_predictions.csv`

Contains:

| Column | Description |
|---|---|
| `CustomerID` | Customer identifier |
| `Churn` | Predicted churn class (`0` or `1`) |

### `customer_churn_predictions_with_probability.csv`

Optional output containing the predicted probability of churn:

| Column | Description |
|---|---|
| `CustomerID` | Customer identifier |
| `Churn` | Predicted class |
| `ChurnProbability` | Probability of churn |

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Kaggle Notebook

## How to Run

1. Open the notebook in Jupyter or Kaggle.
2. Make sure the dataset paths point to the available `cell2celltrain.csv` and `cell2cellholdout.csv` files.
3. Run the cells from top to bottom.
4. Review the validation metrics.
5. Check the generated CSV prediction files.

## Project Structure

```text
customer-churn-prediction/
│
├── customer_churn_prediction_modified.ipynb
├── customer_churn_predictions.csv
├── customer_churn_predictions_with_probability.csv
└── README.md
```

## Notes

The validation score is intended for local model evaluation. The holdout test set is used only for final inference, so its labels should not be used to tune the model.

## Future Improvements

- Hyperparameter tuning with cross-validation
- Compare Random Forest with Gradient Boosting / XGBoost
- Threshold tuning based on the business cost of false negatives vs. false positives
- Explain individual predictions using SHAP
- Add a reproducible `requirements.txt`
