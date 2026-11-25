import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# =====================================================
# 1. PATHS AND SETTINGS
# =====================================================

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "pd_cleaning"
    / "pd_modeling_dataset_clean.csv"
)

PERFORMANCE_FILE = (
    BASE_DIR
    / "data_raw"
    / "freddie_mac_sample"
    / "performance.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "pd_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "logistic_pd_model_final.joblib"

PORTFOLIO_FILE = (
    OUTPUT_DIR / "portfolio_pd_snapshot_202109.csv"
)

SNAPSHOT_PERIOD = "202109"


# =====================================================
# 2. LOAD MODELING DATASET
# =====================================================

print("Loading origination and target dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)


# =====================================================
# 3. DEFINE FEATURES AND TARGET
# =====================================================

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
    "loan_purpose"
]

features = numeric_features + categorical_features

target = "target_12m"

required_columns = (
    features
    + [target, "id_loan", "period"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing modeling columns: {missing_columns}"
    )

df["period"] = (
    df["period"]
    .astype(str)
    .str.strip()
)

if df["id_loan"].isna().any():
    raise ValueError("Missing loan IDs in modeling dataset.")

if df[target].isna().any():
    raise ValueError("Missing target values in modeling dataset.")

df[target] = df[target].astype(int)

if not df[target].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1.")

for col in numeric_features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

for col in categorical_features:
    df[col] = (
        df[col]
        .fillna("MISSING")
        .astype(str)
    )

X = df[features].copy()
y = df[target].copy()
groups = df["id_loan"]


# =====================================================
# 4. PREPROCESSING
# =====================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# =====================================================
# 5. TRAIN FINAL PD MODEL
# =====================================================

print("\nTraining final PD model...")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    ))
])

# Equal total weight per loan

loan_counts = groups.value_counts()

sample_weights = (
    1.0 / groups.map(loan_counts)
).to_numpy()

sample_weights = (
    sample_weights / sample_weights.mean()
)

model.fit(
    X,
    y,
    model__sample_weight=sample_weights
)

joblib.dump(model, MODEL_FILE)

print("Final model trained and saved.")


# =====================================================
# 6. SEPTEMBER 2021 PORTFOLIO
# =====================================================

print("\nPreparing September 2021 portfolio...")

snapshot = df[
    df["period"] == SNAPSHOT_PERIOD
].copy()

if snapshot.empty:
    raise ValueError(
        "No September 2021 observations found."
    )

if snapshot["id_loan"].duplicated().any():
    raise ValueError(
        "Duplicate loans in September 2021 snapshot."
    )

snapshot["pd_12m"] = model.predict_proba(
    snapshot[features]
)[:, 1]

print("Loans in snapshot:", len(snapshot))


# =====================================================
# 7. LOAD PERFORMANCE DATA
# =====================================================

print("\nLoading monthly performance data...")

performance = pd.read_csv(
    PERFORMANCE_FILE,
    low_memory=False
)

performance.columns = performance.columns.str.strip()

# Use only verified columns from the existing script.
# delq_sts is intentionally excluded.

performance_columns = [
    "id_loan",
    "period",
    "curr_act_upb",
    "loan_age",
    "mths_remng"
]

missing_performance = [
    col for col in performance_columns
    if col not in performance.columns
]

if missing_performance:
    raise ValueError(
        f"Missing columns in performance.csv: {missing_performance}"
    )

performance = performance[
    performance_columns
].copy()

performance["period"] = (
    performance["period"]
    .astype(str)
    .str.strip()
)

performance_snapshot = performance[
    performance["period"] == SNAPSHOT_PERIOD
].copy()

if performance_snapshot.empty:
    raise ValueError(
        "No September 2021 performance records found."
    )

if performance_snapshot["id_loan"].duplicated().any():
    raise ValueError(
        "Duplicate loan IDs in performance snapshot."
    )

for col in ["curr_act_upb", "loan_age", "mths_remng"]:
    performance_snapshot[col] = pd.to_numeric(
        performance_snapshot[col],
        errors="coerce"
    )


# =====================================================
# 8. MERGE PORTFOLIO AND PERFORMANCE
# =====================================================

print("\nMerging portfolio and performance data...")

# Performance data is the source of current UPB,
# loan age and remaining maturity.
# Remove possible overlapping columns from snapshot
# to prevent pandas from creating _x and _y suffixes.

performance_value_columns = [
    "curr_act_upb",
    "loan_age",
    "mths_remng"
]

snapshot_for_merge = snapshot.drop(
    columns=[
        col for col in performance_value_columns
        if col in snapshot.columns
    ],
    errors="ignore"
)

portfolio = snapshot_for_merge.merge(
    performance_snapshot,
    on=["id_loan", "period"],
    how="left",
    validate="one_to_one",
    indicator=True
)

matched = (portfolio["_merge"] == "both").sum()
unmatched = (portfolio["_merge"] == "left_only").sum()

print("\n===== MERGE CHECK =====")
print("Portfolio loans:", len(portfolio))
print("Matched loans:", matched)
print("Unmatched loans:", unmatched)

if unmatched > 0:
    raise ValueError(
        f"{unmatched} portfolio loans have no matching "
        "performance record. No values will be fabricated."
    )

if portfolio["curr_act_upb"].isna().any():
    raise ValueError(
        "Missing current UPB values. ECL portfolio cannot "
        "be finalized with incomplete exposure data."
    )

portfolio = portfolio.drop(columns=["_merge"])


# =====================================================
# 9. FINAL PORTFOLIO STRUCTURE
# =====================================================

output_columns = [
    "id_loan",
    "period",
    "pd_12m",
    "curr_act_upb",
    "loan_age",
    "mths_remng",
    "fico",
    "ltv",
    "cltv",
    "dti",
    "orig_upb",
    "orig_int_rt"
]

missing_output = [
    col for col in output_columns
    if col not in portfolio.columns
]

if missing_output:
    raise ValueError(
        f"Missing final portfolio columns: {missing_output}"
    )

portfolio = portfolio[
    output_columns
].copy()

if len(portfolio) != len(snapshot):
    raise ValueError(
        "Portfolio record count changed during processing."
    )

if portfolio["id_loan"].duplicated().any():
    raise ValueError(
        "Duplicate loan IDs in final portfolio."
    )

if not portfolio["pd_12m"].between(0, 1).all():
    raise ValueError("Invalid PD values detected.")

if (portfolio["curr_act_upb"] < 0).any():
    raise ValueError("Negative UPB values detected.")


# =====================================================
# 10. PORTFOLIO SUMMARY
# =====================================================

print("\n===== PORTFOLIO PD SUMMARY =====")

print("Snapshot date: September 2021")
print("Number of loans:", len(portfolio))

total_upb = portfolio["curr_act_upb"].sum()

print(
    "Total outstanding balance:",
    round(total_upb, 2)
)

print(
    "Average PD:",
    round(portfolio["pd_12m"].mean(), 6)
)

weighted_average_pd = np.average(
    portfolio["pd_12m"],
    weights=portfolio["curr_act_upb"]
)

print(
    "Balance-weighted average PD:",
    round(weighted_average_pd, 6)
)

print(
    "Minimum PD:",
    round(portfolio["pd_12m"].min(), 6)
)

print(
    "Maximum PD:",
    round(portfolio["pd_12m"].max(), 6)
)

print(
    "Missing UPB:",
    portfolio["curr_act_upb"].isna().sum()
)


# =====================================================
# 11. SAVE OUTPUTS
# =====================================================

portfolio.to_csv(
    PORTFOLIO_FILE,
    index=False
)

print("\n===== SAVED FILES =====")

print("Model:", MODEL_FILE)
print("Portfolio:", PORTFOLIO_FILE)

print("\nScript 26 completed successfully.")



