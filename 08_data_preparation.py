import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = Path(r"C:\Python\ECL modeling\data_raw.csv")

OUTPUT_DIR = BASE_DIR / "outputs" / "data_preparation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Status"
TEST_SIZE = 0.30
RANDOM_STATE = 42
SMOOTHING = 0.5
MAX_BINS = 5

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("PD MODEL - DATA PREPARATION")
print("=" * 65)

print("Original dataset:", df.shape)

# Exclude EQUI from primary development sample
df["credit_type"] = (
    df["credit_type"].astype(str).str.strip().str.upper()
)

df = df[df["credit_type"] != "EQUI"].copy()

print("Dataset without EQUI:", df.shape)

# ============================================================
# 3. DEFINE FEATURES
# ============================================================

numerical_features = [
    "dtir1",
    "LTV",
    "income",
    "property_value",
    "loan_amount",
    "Credit_Score"
]

categorical_features = [
    "Neg_ammortization",
    "loan_type",
    "loan_purpose",
    "Credit_Worthiness",
    "business_or_commercial",
    "loan_limit",
    "age",
    "submission_of_application",
    "Region"
]

features = numerical_features + categorical_features

# Exclude previously identified problematic variables
excluded_features = [
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges",
    "credit_type",
    "ID",
    "year"
]

X = df[features].copy()
y = df[TARGET].astype(int).copy()

# ============================================================
# 4. BASIC DATA CLEANING
# ============================================================

# Normalize missing values in categorical variables
for col in categorical_features:
    X[col] = (
        X[col]
        .astype("object")
        .where(X[col].notna(), "MISSING")
        .astype(str)
        .str.strip()
    )

# Convert numerical columns to numeric
for col in numerical_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Remove impossible / extreme LTV values.
# This is a conservative data-quality rule:
# LTV above 200% is excluded from this initial model.
# Keep a record of excluded observations.
invalid_ltv = X["LTV"] > 200

print("\nObservations with LTV above 200%:", invalid_ltv.sum())

X = X.loc[~invalid_ltv].copy()
y = y.loc[X.index].copy()

# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTRAIN / TEST SPLIT")
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nTraining default rate:", round(y_train.mean() * 100, 2), "%")
print("Testing default rate:", round(y_test.mean() * 100, 2), "%")

# ============================================================
# 6. WOЕ BINNING AND TRANSFORMATION
# ============================================================

woe_mappings = {}
bin_definitions = {}
woe_summary = []

def calculate_woe(data, target, feature, bin_series, smoothing=0.5):

    temp = pd.DataFrame({
        "bin": bin_series,
        "target": target
    })

    grouped = temp.groupby("bin", observed=False)["target"].agg(
        ["count", "sum"]
    )

    grouped["non_default"] = grouped["count"] - grouped["sum"]

    total_defaults = grouped["sum"].sum()
    total_non_defaults = grouped["non_default"].sum()

    n_bins = len(grouped)

    grouped["default_dist"] = (
        grouped["sum"] + smoothing
    ) / (total_defaults + smoothing * n_bins)

    grouped["non_default_dist"] = (
        grouped["non_default"] + smoothing
    ) / (total_non_defaults + smoothing * n_bins)

    grouped["WOE"] = np.log(
        grouped["non_default_dist"] / grouped["default_dist"]
    )

    grouped["IV_component"] = (
        grouped["non_default_dist"] - grouped["default_dist"]
    ) * grouped["WOE"]

    grouped["Feature"] = feature

    return grouped


# Numerical features: quantile binning based on TRAIN only
for col in numerical_features:

    train_values = X_train[col]
    median_value = train_values.median()

    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

    # Fit quantile boundaries on training data only
    _, bin_edges = pd.qcut(
        X_train[col],
        q=MAX_BINS,
        retbins=True,
        duplicates="drop"
    )

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    bin_definitions[col] = bin_edges

    train_bins = pd.cut(
        X_train[col],
        bins=bin_edges,
        include_lowest=True
    )

    test_bins = pd.cut(
        X_test[col],
        bins=bin_edges,
        include_lowest=True
    )

    mapping = calculate_woe(
        X_train,
        y_train,
        col,
        train_bins,
        SMOOTHING
    )

    woe_map = mapping["WOE"].to_dict()
    woe_mappings[col] = woe_map

    X_train[col] = train_bins.map(woe_map).astype(float)
    X_test[col] = test_bins.map(woe_map).astype(float)

    # Unseen test bins: assign neutral WOE
    X_train[col] = X_train[col].fillna(0)
    X_test[col] = X_test[col].fillna(0)

    mapping = mapping.reset_index()
    mapping.to_csv(
        OUTPUT_DIR / f"woe_{col}.csv",
        index=False
    )

    woe_summary.append({
        "Feature": col,
        "IV": mapping["IV_component"].sum(),
        "Number_of_Bins": len(mapping)
    })


# Categorical features: fit mapping on TRAIN only
for col in categorical_features:

    train_bins = X_train[col].fillna("MISSING").astype(str)
    test_bins = X_test[col].fillna("MISSING").astype(str)

    mapping = calculate_woe(
        X_train,
        y_train,
        col,
        train_bins,
        SMOOTHING
    )

    woe_map = mapping["WOE"].to_dict()
    woe_mappings[col] = woe_map

    X_train[col] = train_bins.map(woe_map).astype(float)

    # Unknown categories in test receive neutral WOE
    X_test[col] = (
        test_bins.map(woe_map)
        .fillna(0)
        .astype(float)
    )

    mapping = mapping.reset_index()
    mapping.to_csv(
        OUTPUT_DIR / f"woe_{col}.csv",
        index=False
    )

    woe_summary.append({
        "Feature": col,
        "IV": mapping["IV_component"].sum(),
        "Number_of_Bins": len(mapping)
    })

# ============================================================
# 7. SAVE WOE SUMMARY
# ============================================================

woe_summary_df = pd.DataFrame(woe_summary)

woe_summary_df = woe_summary_df.sort_values(
    "IV",
    ascending=False
)

woe_summary_df.to_csv(
    OUTPUT_DIR / "woe_summary.csv",
    index=False
)

print("\nWOE SUMMARY")
print(woe_summary_df.to_string(index=False))

# ============================================================
# 8. SAVE TRAINING AND TEST DATASETS
# ============================================================

train_final = X_train.copy()
train_final[TARGET] = y_train

test_final = X_test.copy()
test_final[TARGET] = y_test

train_final.to_csv(
    OUTPUT_DIR / "train_woe.csv",
    index=False
)

test_final.to_csv(
    OUTPUT_DIR / "test_woe.csv",
    index=False
)

# ============================================================
# 9. SAVE FEATURE LIST
# ============================================================

pd.DataFrame({
    "Feature": features,
    "Type": (
        ["Numerical"] * len(numerical_features)
        + ["Categorical"] * len(categorical_features)
    )
}).to_csv(
    OUTPUT_DIR / "feature_list.csv",
    index=False
)

# ============================================================
# 10. FINAL CHECKS
# ============================================================

print("\nFINAL CHECKS")

print("Missing values in training:", X_train.isna().sum().sum())
print("Missing values in testing:", X_test.isna().sum().sum())

print("Training shape:", train_final.shape)
print("Testing shape:", test_final.shape)

print("\nFiles saved in:", OUTPUT_DIR)

print("\nDATA PREPARATION COMPLETED")