from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_FILE = BASE_DIR / "titanic_cleaned.csv"

TARGET = "survived"

FEATURES = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

NUMERIC_FEATURES = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

CATEGORICAL_FEATURES = [
    "sex",
    "embarked"
]


def make_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )


def classification_metrics(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    name
):

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(f"\n{name} confusion matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()

    plt.title(
        f"{name} Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.close()


def main():

    print("Starting classification modeling...")

    df = pd.read_csv(DATA_FILE)

    # Keep only required fields.
    df = df[
        FEATURES + [TARGET]
    ].copy()

    X = df[FEATURES]
    y = df[TARGET]

    # IMPORTANT:
    # Stratification happens BEFORE preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain size:", X_train.shape)
    print("Test size :", X_test.shape)

    preprocessor = make_preprocessor()

    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42,
                max_depth=5
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
    }

    results = {}

    fitted_models = {}

    for name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor)
                ),
                (
                    "model",
                    estimator
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        metrics = classification_metrics(
            pipeline,
            X_test,
            y_test
        )

        results[name] = metrics
        fitted_models[name] = pipeline

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        for metric, value in metrics.items():
            print(
                f"{metric}: {value:.4f}"
            )

        plot_confusion_matrix(
            pipeline,
            X_test,
            y_test,
            name
        )

    # ---------------------------------------------------------
    # ROC curves
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 6))

    for name, model in fitted_models.items():

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name=name
        )

    plt.title(
        "ROC Curves - Classification Models"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "classification_roc_curves.png"
    )

    plt.close()

    # ---------------------------------------------------------
    # Decision Tree visualization
    # ---------------------------------------------------------

    tree_model = fitted_models[
        "Decision Tree"
    ]

    transformed_train = (
        tree_model
        .named_steps["preprocessor"]
        .transform(X_train)
    )

    feature_names = (
        tree_model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    plt.figure(figsize=(20, 10))

    plot_tree(
        tree_model.named_steps["model"],
        feature_names=feature_names,
        class_names=["Not Survived", "Survived"],
        filled=False,
        max_depth=3,
        fontsize=8
    )

    plt.title(
        "Decision Tree - First 3 Levels"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "decision_tree.png"
    )

    plt.close()

    # ---------------------------------------------------------
    # Imbalance experiments
    # ---------------------------------------------------------

    imbalance_results = []

    # Baseline
    baseline_model = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    baseline_model.fit(
        X_train,
        y_train
    )

    baseline_metrics = classification_metrics(
        baseline_model,
        X_test,
        y_test
    )

    imbalance_results.append({
        "method": "Baseline",
        **baseline_metrics
    })

    # Class weight balanced
    balanced_model = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    balanced_model.fit(
        X_train,
        y_train
    )

    balanced_metrics = classification_metrics(
        balanced_model,
        X_test,
        y_test
    )

    imbalance_results.append({
        "method": "class_weight=balanced",
        **balanced_metrics
    })

    # SMOTE
    smote_model = ImbPipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "smote",
                SMOTE(random_state=42)
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    smote_model.fit(
        X_train,
        y_train
    )

    smote_metrics = classification_metrics(
        smote_model,
        X_test,
        y_test
    )

    imbalance_results.append({
        "method": "SMOTE",
        **smote_metrics
    })

    imbalance_df = pd.DataFrame(
        imbalance_results
    )

    print("\n" + "=" * 70)
    print("CLASS IMBALANCE COMPARISON")
    print("=" * 70)

    print(
        imbalance_df.to_string(
            index=False
        )
    )

    imbalance_df.to_csv(
        OUTPUT_DIR /
        "imbalance_comparison.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Random Forest GridSearchCV
    # ---------------------------------------------------------

    rf_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    oob_score=True,
                    n_jobs=-1
                )
            )
        ]
    )

    param_grid = {
        "model__n_estimators": [
            100,
            200
        ],
        "model__max_depth": [
            None,
            5,
            10
        ],
        "model__max_features": [
            "sqrt",
            "log2"
        ]
    }

    grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

    best_rf = grid.best_estimator_

    best_rf_metrics = classification_metrics(
        best_rf,
        X_test,
        y_test
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST GRID SEARCH")
    print("=" * 70)

    print(
        "Best parameters:",
        grid.best_params_
    )

    print(
        "Best CV F1:",
        grid.best_score_
    )

    print(
        "Test metrics:",
        best_rf_metrics
    )

    # OOB score
    oob_score = (
        best_rf
        .named_steps["model"]
        .oob_score_
    )

    print(
        f"OOB score: {oob_score:.4f}"
    )

    # ---------------------------------------------------------
    # Regression: predict fare
    # ---------------------------------------------------------

    regression_features = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "survived"
    ]

    regression_target = "fare"

    regression_df = df[
        regression_features +
        [regression_target]
    ].dropna()

    X_reg = regression_df[
        regression_features
    ]

    y_reg = regression_df[
        regression_target
    ]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )

    regression_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),
                        (
                            "scaler",
                            StandardScaler()
                        )
                    ]
                ),
                regression_features
            )
        ]
    )

    regression_model = Pipeline(
        steps=[
            (
                "preprocessor",
                regression_preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    regression_model.fit(
        Xr_train,
        yr_train
    )

    yr_pred = regression_model.predict(
        Xr_test
    )

    mae = mean_absolute_error(
        yr_test,
        yr_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            yr_test,
            yr_pred
        )
    )

    r2 = r2_score(
        yr_test,
        yr_pred
    )

    n = len(yr_test)
    p = len(regression_features)

    if n > p + 1:
        adjusted_r2 = (
            1 -
            (1 - r2) *
            (n - 1) /
            (n - p - 1)
        )
    else:
        adjusted_r2 = np.nan

    print("\n" + "=" * 70)
    print("FARE REGRESSION")
    print("=" * 70)

    print(f"MAE       : {mae:.4f}")
    print(f"RMSE      : {rmse:.4f}")
    print(f"R2        : {r2:.4f}")
    print(
        f"Adjusted R2: {adjusted_r2:.4f}"
    )

    # Residual plot
    residuals = yr_test - yr_pred

    plt.figure(figsize=(8, 5))

    plt.scatter(
        yr_pred,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel(
        "Predicted Fare"
    )

    plt.ylabel(
        "Residual"
    )

    plt.title(
        "Fare Regression Residual Plot"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fare_residual_plot.png"
    )

    plt.close()

    # ---------------------------------------------------------
    # Final comparison tables
    # ---------------------------------------------------------

    classification_df = pd.DataFrame(
        results
    ).T.reset_index()

    classification_df = (
        classification_df
        .rename(columns={"index": "model"})
    )

    classification_df.to_csv(
        OUTPUT_DIR /
        "classification_results.csv",
        index=False
    )

    regression_results = pd.DataFrame([
        {
            "model": "Linear Regression",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2
        }
    ])

    regression_results.to_csv(
        OUTPUT_DIR /
        "regression_results.csv",
        index=False
    )

    # Combined final report
    final_report = {
        "best_random_forest_parameters":
            str(grid.best_params_),
        "random_forest_oob_score":
            oob_score,
        "regression_MAE":
            mae,
        "regression_RMSE":
            rmse,
        "regression_R2":
            r2,
        "regression_adjusted_R2":
            adjusted_r2
    }

    pd.DataFrame(
        [final_report]
    ).to_csv(
        OUTPUT_DIR /
        "final_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Save complete classification pipeline
    # ---------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "full_pipeline.joblib"
    )

    joblib.dump(
        best_rf,
        model_path
    )

    print(
        f"\nSaved complete pipeline to:"
        f"\n{model_path}"
    )

    # ---------------------------------------------------------
    # Reload pipeline and predict raw data
    # ---------------------------------------------------------

    loaded_pipeline = joblib.load(
        model_path
    )

    sample_raw = X_test.head(5)

    sample_predictions = (
        loaded_pipeline
        .predict(sample_raw)
    )

    print("\nReloaded pipeline predictions:")

    print(sample_predictions)

    print(
        "\nClassification and regression "
        "modeling completed successfully."
    )


if __name__ == "__main__":
    main()
