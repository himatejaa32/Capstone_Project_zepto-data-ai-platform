from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

RAW_FILE = BASE_DIR / "titanic.csv"
CLEAN_FILE = BASE_DIR / "titanic_cleaned.csv"


def load_titanic():

    # Assignment requires seaborn dataset loading.
    df = sns.load_dataset("titanic")

    # Save immediately so later runs can work offline.
    df.to_csv(RAW_FILE, index=False)

    print(f"Saved raw Titanic data to: {RAW_FILE}")

    return df


def missing_value_report(df):

    print("\n" + "=" * 70)
    print("MISSING VALUE REPORT")
    print("=" * 70)

    missing = df.isna().mean().mul(100).sort_values(ascending=False)

    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": missing
    })

    print(report[report["missing_count"] > 0])


def apply_missing_value_strategy(df):

    df = df.copy()

    # Very high missingness: deck is approximately 77%.
    # It is not useful enough to keep in this baseline project.
    if "deck" in df.columns:
        df = df.drop(columns=["deck"])

    # Drop rows for columns with <5% missing values.
    low_missing_columns = [
        column
        for column in df.columns
        if 0 < df[column].isna().mean() < 0.05
    ]

    if low_missing_columns:
        df = df.dropna(subset=low_missing_columns)

    # Impute columns between 5% and 30%.
    medium_missing_columns = [
        column
        for column in df.columns
        if 0.05 <= df[column].isna().mean() <= 0.30
    ]

    for column in medium_missing_columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(
                df[column].median()
            )
        else:
            mode = df[column].mode()

            if not mode.empty:
                df[column] = df[column].fillna(
                    mode.iloc[0]
                )

    return df


def create_eda_charts(df):

    # 1. Age histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="age",
        bins=30,
        kde=True
    )
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "01_age_histogram.png"
    )
    plt.close()

    # 2. Fare histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="fare",
        bins=30,
        kde=True
    )
    plt.title("Fare Distribution")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "02_fare_histogram.png"
    )
    plt.close()

    # 3. Age boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="age"
    )
    plt.title("Age Boxplot")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "03_age_boxplot.png"
    )
    plt.close()

    # 4. Fare boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="fare"
    )
    plt.title("Fare Boxplot")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "04_fare_boxplot.png"
    )
    plt.close()


def calculate_iqr_outliers(df, column):

    series = df[column].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = (
        (series < lower) |
        (series > upper)
    ).sum()

    return {
        "column": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": count
    }


def fare_statistics(df):

    fare = df["fare"].dropna()

    print("\n" + "=" * 70)
    print("FARE STATISTICS")
    print("=" * 70)

    print(f"Mean   : {fare.mean():.4f}")
    print(f"Median : {fare.median():.4f}")
    print(f"Mode   : {fare.mode().iloc[0]:.4f}")
    print(f"Skewness: {fare.skew():.4f}")


def survival_analysis(df):

    print("\n" + "=" * 70)
    print("SURVIVAL BY SEX")
    print("=" * 70)

    print(
        df.groupby("sex")["survived"]
        .mean()
        .to_string()
    )

    print("\n" + "=" * 70)
    print("SURVIVAL BY PCLASS")
    print("=" * 70)

    print(
        df.groupby("pclass")["survived"]
        .mean()
        .to_string()
    )

    print("\n" + "=" * 70)
    print("SURVIVAL BY SEX + PCLASS")
    print("=" * 70)

    print(
        df.groupby(
            ["sex", "pclass"]
        )["survived"]
        .mean()
        .to_string()
    )


def correlation_analysis(df):

    columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]

    correlation = df[columns].corr()

    print("\n" + "=" * 70)
    print("EXACT SIX-COLUMN CORRELATION MATRIX")
    print("=" * 70)

    print(correlation)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Titanic Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "05_correlation_heatmap.png"
    )

    plt.close()


def multivariate_charts(df):

    # 1
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="sex",
        y="survived",
        hue="pclass"
    )

    plt.title(
        "Survival Rate by Sex and Passenger Class"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "06_survival_sex_pclass.png"
    )

    plt.close()

    # 2
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="survived",
        y="fare",
        hue="sex"
    )

    plt.title(
        "Fare Distribution by Survival and Sex"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "07_fare_survival_sex.png"
    )

    plt.close()

    # 3
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="pclass",
        hue="survived"
    )

    plt.title(
        "Passenger Class and Survival"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "08_pclass_survival.png"
    )

    plt.close()

    # 4
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="age",
        y="fare",
        hue="survived",
        alpha=0.7
    )

    plt.title(
        "Age vs Fare by Survival"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "09_age_fare_survival.png"
    )

    plt.close()


def zscore_check(df):

    print("\n" + "=" * 70)
    print("Z-SCORE EXPLORATORY CHECK")
    print("=" * 70)

    for column in ["age", "fare"]:

        values = df[column].dropna()

        z_scores = (
            values - values.mean()
        ) / values.std(ddof=0)

        extreme_count = (
            np.abs(z_scores) > 3
        ).sum()

        print(
            f"{column}: "
            f"{extreme_count} observations "
            f"with |z| > 3"
        )


def main():

    print("Starting Titanic EDA...")

    df = load_titanic()

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset information:")
    print(df.info())

    print("\nDescriptive statistics:")
    print(df.describe(include="all").transpose())

    missing_value_report(df)

    create_eda_charts(df)

    print("\nIQR OUTLIER COUNTS")

    for column in ["age", "fare"]:

        result = calculate_iqr_outliers(
            df,
            column
        )

        print(result)

    fare_statistics(df)

    survival_analysis(df)

    correlation_analysis(df)

    multivariate_charts(df)

    zscore_check(df)

    cleaned = apply_missing_value_strategy(df)

    cleaned.to_csv(
        CLEAN_FILE,
        index=False
    )

    print(
        f"\nCleaned dataset saved to: {CLEAN_FILE}"
    )

    print(
        f"Original shape: {df.shape}"
    )

    print(
        f"Cleaned shape: {cleaned.shape}"
    )

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()