# Module 2 — Analytics Pipeline

## Dataset

The Titanic dataset is loaded using Seaborn and saved locally as `titanic.csv`.

After the first successful download/load, the CSV can be reused offline.

## EDA

The analysis includes:

- Dataset shape
- Dataset information
- Descriptive statistics
- Missing-value analysis
- Missing-value treatment
- Age histogram
- Fare histogram
- Age boxplot
- Fare boxplot
- IQR outlier analysis
- Fare mean, median, mode and skewness
- Survival by sex
- Survival by passenger class
- Survival by sex and passenger class
- Six-column correlation matrix
- Multivariate visualizations
- Z-score exploratory analysis

## Classification

Models:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix

## Class Imbalance

Three approaches are compared:

1. Baseline
2. `class_weight="balanced"`
3. SMOTE

SMOTE is applied only to the training data through the modeling pipeline.

## Random Forest Optimization

GridSearchCV evaluates:

- `n_estimators`
- `max_depth`
- `max_features`

The optimized model also reports its OOB score.

## Regression

A Linear Regression model predicts Titanic fare.

Metrics:

- MAE
- RMSE
- R²
- Adjusted R²

A residual plot is used to inspect possible heteroscedasticity.

## Saved Model

The complete classification preprocessing and model pipeline is saved as:

`models/full_pipeline.joblib`

The saved pipeline is reloaded and used to generate predictions on raw test features.