import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "pd_cleaning"
    / "pd_modeling_dataset_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "pd_cleaning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# 1. Default rate by numerical feature
numeric_features = [
    "fico",
    "ltv",
    "cltv",
    "dti",
    "orig_upb",
    "orig_int_rt",
    "orig_loan_term"
]

results = []

for col in numeric_features:
    temp = df[[col, "target_12m"]].dropna().copy()

    temp["group"] = pd.qcut(
        temp[col],
        q=5,
        duplicates="drop"
    )

    summary = (
        temp.groupby("group", observed=False)
        .agg(
            observations=("target_12m", "size"),
            positive_targets=("target_12m", "sum"),
            target_rate=("target_12m", "mean")
        )
        .reset_index()
    )

    summary["feature"] = col
    results.append(summary)

    print(f"\n===== TARGET RATE BY {col.upper()} =====")
    print(summary.to_string(index=False))

# 2. Spearman correlation
corr_features = numeric_features + ["target_12m"]

correlations = (
    df[corr_features]
    .corr(method="spearman")["target_12m"]
    .sort_values(ascending=False)
)

print("\n===== SPEARMAN CORRELATIONS =====")
print(correlations)

# 3. Missing values
missing = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename("missing_pct")
    .reset_index()
)

missing.columns = ["variable", "missing_pct"]

print("\n===== MISSING VALUES =====")
print(missing.head(15).to_string(index=False))

# 4. Save results
all_results = pd.concat(results, ignore_index=True)

all_results.to_csv(
    OUTPUT_DIR / "feature_analysis_clean.csv",
    index=False
)

correlations.to_csv(
    OUTPUT_DIR / "spearman_correlations_clean.csv"
)

missing.to_csv(
    OUTPUT_DIR / "missing_values_clean.csv",
    index=False
)

print("\nClean feature analysis completed.")

