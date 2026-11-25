from pathlib import Path
import pandas as pd

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

OUTPUT_DIR = PROJECT_DIR / "outputs" / "pd_cleaning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading modeling dataset...", flush=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. CHECK SPECIAL VALUES
# --------------------------------------------------

special_checks = {
    "fico": [9999],
    "cltv": [999]
}

print("\nSpecial value counts:")

for col, values in special_checks.items():

    for value in values:

        count = (df[col] == value).sum()

        print(f"{col} = {value}: {count:,}")

# --------------------------------------------------
# 4. CONVERT CONFIRMED SENTINELS TO MISSING
# --------------------------------------------------

# These two codes must be confirmed against the
# dataset's Freddie Mac data dictionary before
# treating them as missing values.

for col, values in special_checks.items():

    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

    df[col] = df[col].replace(values, pd.NA)

# --------------------------------------------------
# 5. CHECK OTHER NUMERIC VARIABLES
# --------------------------------------------------

numeric_cols = [
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

print("\nUpdated descriptive statistics:")

print(
    df[numeric_cols]
    .describe(
        percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
    )
    .T
    .round(2)
    .to_string()
)

# --------------------------------------------------
# 6. MISSING VALUE REPORT
# --------------------------------------------------

missing_report = pd.DataFrame({
    "variable": df.columns,
    "missing_count": [
        df[col].isna().sum() for col in df.columns
    ],
    "missing_percentage": [
        df[col].isna().mean() * 100
        for col in df.columns
    ]
})

missing_report = missing_report.sort_values(
    "missing_percentage",
    ascending=False
)

missing_report.to_csv(
    OUTPUT_DIR / "missing_values_after_cleaning.csv",
    index=False
)

# --------------------------------------------------
# 7. SAVE CLEAN DATASET
# --------------------------------------------------

OUTPUT_FILE = OUTPUT_DIR / "pd_modeling_dataset_clean.csv"

df.to_csv(OUTPUT_FILE, index=False)

print("\nClean dataset saved:")
print(OUTPUT_FILE)

print("\nMissing value report saved:")
print(OUTPUT_DIR / "missing_values_after_cleaning.csv")

print("\nScript 23 completed successfully.")

