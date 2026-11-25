from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data_raw" / "freddie_mac_sample"

TARGET_FILE = (
    PROJECT_DIR
    / "outputs"
    / "default_analysis"
    / "pd_12m_target_feasibility.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "pd_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "pd_modeling_dataset.csv"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading origination data...", flush=True)

orig = pd.read_csv(
    DATA_DIR / "origination.csv",
    low_memory=False
)

print("Loading 12-month PD target data...", flush=True)

target = pd.read_csv(
    TARGET_FILE,
    low_memory=False
)

print(f"Origination records: {len(orig):,}")
print(f"Target observations: {len(target):,}")


# --------------------------------------------------
# 3. CLEAN LOAN IDENTIFIERS
# --------------------------------------------------

orig["id_loan"] = (
    orig["id_loan"].astype("string").str.strip()
)

target["id_loan"] = (
    target["id_loan"].astype("string").str.strip()
)

if orig["id_loan"].duplicated().any():
    raise ValueError(
        "Origination data contains duplicate loan IDs."
    )

if target["id_loan"].isna().any():
    raise ValueError(
        "Target data contains missing loan IDs."
    )


# --------------------------------------------------
# 4. CHECK TARGET
# --------------------------------------------------

target["target_12m"] = pd.to_numeric(
    target["target_12m"],
    errors="coerce"
)

if target["target_12m"].isna().any():
    raise ValueError(
        "Missing values detected in target_12m."
    )

if not target["target_12m"].isin([0, 1]).all():
    raise ValueError(
        "Target must contain only 0 and 1."
    )


# --------------------------------------------------
# 5. MERGE ORIGINATION AND TARGET
# --------------------------------------------------

print("\nMerging datasets...", flush=True)

dataset = target.merge(
    orig,
    on="id_loan",
    how="left",
    validate="many_to_one",
    indicator=True,
    suffixes=("", "_orig")
)

# Check whether all target records found an origination record.
merge_counts = dataset["_merge"].value_counts()

print("\nMerge results:")
print(merge_counts)

unmatched = (dataset["_merge"] != "both").sum()

if unmatched > 0:
    raise ValueError(
        f"{unmatched} target observations have no origination match."
    )

dataset = dataset.drop(columns=["_merge"])


# --------------------------------------------------
# 6. CHECK DATASET STRUCTURE
# --------------------------------------------------

print("\nMerged dataset shape:")
print(dataset.shape)

print("\nUnique loans:")
print(dataset["id_loan"].nunique())

print("\nTarget distribution:")
print(dataset["target_12m"].value_counts())

print("\nTarget rate:")
print(f"{dataset['target_12m'].mean():.2%}")


# --------------------------------------------------
# 7. CHECK MISSING VALUES
# --------------------------------------------------

missing = (
    dataset.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

missing_report = missing.reset_index()
missing_report.columns = [
    "variable",
    "missing_percentage"
]

missing_file = OUTPUT_DIR / "missing_values_report.csv"

missing_report.to_csv(
    missing_file,
    index=False
)

print("\nVariables with highest missingness:")
print(missing_report.head(15).to_string(index=False))


# --------------------------------------------------
# 8. CHECK CONSTANT VARIABLES
# --------------------------------------------------

constant_columns = [
    col for col in dataset.columns
    if dataset[col].nunique(dropna=False) <= 1
]

print("\nConstant variables:")
print(constant_columns)


# --------------------------------------------------
# 9. SAVE MODELING DATASET
# --------------------------------------------------

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDataset successfully saved:")
print(OUTPUT_FILE)

print("\nMissing values report saved:")
print(missing_file)

print("\nScript 21 completed successfully.")

