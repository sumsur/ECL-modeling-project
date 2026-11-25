from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "pd_model"
    / "pd_modeling_dataset.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "pd_feature_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading PD modeling dataset...", flush=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# 3. DEFINE FEATURES
# --------------------------------------------------

numeric_features = [
    "fico",
    "ltv",
    "cltv",
    "dti",
    "orig_upb",
    "orig_int_rt",
    "orig_loan_term",
    "cnt_borr",
    "cnt_units",
    "mi_pct"
]

categorical_features = [
    "flag_fthb",
    "occpy_sts",
    "channel",
    "st",
    "prop_type",
    "loan_purpose",
    "seller_name"
]

# Keep only columns available in the dataset.
numeric_features = [
    c for c in numeric_features if c in df.columns
]

categorical_features = [
    c for c in categorical_features if c in df.columns
]


# --------------------------------------------------
# 4. CONVERT NUMERIC FEATURES
# --------------------------------------------------

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------
# 5. NUMERIC DESCRIPTIVE STATISTICS
# --------------------------------------------------

print("\nCalculating numeric statistics...", flush=True)

numeric_summary = df[numeric_features].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

numeric_summary["missing_pct"] = (
    df[numeric_features].isna().mean() * 100
)

numeric_summary.to_csv(
    OUTPUT_DIR / "numeric_feature_summary.csv"
)

print("\nNumeric feature summary:")
print(numeric_summary.round(2).to_string())


# --------------------------------------------------
# 6. DEFAULT RATE BY NUMERIC BINS
# --------------------------------------------------

print("\nCalculating default rates by numeric groups...", flush=True)

bin_results = []

# Quantile-based groups allow comparison across features.
for col in numeric_features:

    valid = df[[col, "target_12m"]].dropna().copy()

    if valid[col].nunique() < 2:
        continue

    try:
        valid["group"] = pd.qcut(
            valid[col],
            q=5,
            duplicates="drop"
        )
    except ValueError:
        continue

    grouped = (
        valid.groupby("group", observed=True)
        .agg(
            observations=("target_12m", "size"),
            defaults=("target_12m", "sum"),
            default_rate=("target_12m", "mean"),
            mean_feature_value=(col, "mean")
        )
        .reset_index()
    )

    grouped["feature"] = col
    grouped["group"] = grouped["group"].astype(str)

    bin_results.append(grouped)

if bin_results:
    numeric_default_analysis = pd.concat(
        bin_results,
        ignore_index=True
    )

    numeric_default_analysis.to_csv(
        OUTPUT_DIR / "numeric_default_rates.csv",
        index=False
    )

    print("\nNumeric default-rate analysis:")
    print(numeric_default_analysis.to_string(index=False))


# --------------------------------------------------
# 7. CATEGORICAL DEFAULT RATES
# --------------------------------------------------

print("\nCalculating categorical default rates...", flush=True)

categorical_results = []

for col in categorical_features:

    temp = df[[col, "target_12m"]].copy()

    temp[col] = (
        temp[col]
        .fillna("<MISSING>")
        .astype(str)
    )

    grouped = (
        temp.groupby(col)
        .agg(
            observations=("target_12m", "size"),
            defaults=("target_12m", "sum"),
            default_rate=("target_12m", "mean")
        )
        .reset_index()
    )

    grouped["feature"] = col

    # Avoid interpreting very small groups as reliable.
    grouped["low_sample_flag"] = (
        grouped["observations"] < 100
    )

    categorical_results.append(grouped)

if categorical_results:
    categorical_default_analysis = pd.concat(
        categorical_results,
        ignore_index=True
    )

    categorical_default_analysis.to_csv(
        OUTPUT_DIR / "categorical_default_rates.csv",
        index=False
    )

    print("\nCategorical default-rate analysis:")
    print(categorical_default_analysis.to_string(index=False))


# --------------------------------------------------
# 8. CORRELATION MATRIX
# --------------------------------------------------

print("\nCalculating correlations...", flush=True)

correlation_features = numeric_features + ["target_12m"]

correlation_matrix = df[
    correlation_features
].corr(method="spearman")

correlation_matrix.to_csv(
    OUTPUT_DIR / "spearman_correlation_matrix.csv"
)

print("\nSpearman correlations with target:")

print(
    correlation_matrix["target_12m"]
    .sort_values(ascending=False)
    .round(4)
)


# --------------------------------------------------
# 9. LOAN-LEVEL CHECK
# --------------------------------------------------

print("\nChecking repeated observations...", flush=True)

loan_counts = (
    df.groupby("id_loan")
    .agg(
        observations=("target_12m", "size"),
        positive_observations=("target_12m", "sum"),
        ever_positive=("target_12m", "max")
    )
)

print("\nLoan-level observation summary:")
print(loan_counts.describe().round(2))

print(
    "\nLoans with at least one positive 12-month target:",
    int(loan_counts["ever_positive"].sum())
)


# --------------------------------------------------
# 10. SAVE CLEAN ANALYSIS DATA
# --------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "pd_feature_analysis_base.csv",
    index=False
)

print("\nAll results saved to:")
print(OUTPUT_DIR)

print("\nScript 22 completed successfully.")

