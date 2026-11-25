from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "outputs" / "data_preparation"
OUTPUT_DIR = BASE_DIR / "outputs" / "region_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

TARGET = "Status"

BASE_FEATURES = [
    "dtir1",
    "LTV",
    "income",
    "property_value",
    "loan_amount",
    "Credit_Score",
    "Neg_ammortization",
    "loan_type",
    "loan_purpose",
    "Credit_Worthiness",
    "business_or_commercial",
    "loan_limit",
    "age",
    "submission_of_application"
]

FEATURES = BASE_FEATURES + ["Region"]


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(DATA_DIR / "train_woe.csv")
test = pd.read_csv(DATA_DIR / "test_woe.csv")

mapping_files = list(DATA_DIR.glob("*Region*.csv"))

if not mapping_files:
    mapping_files = [
        file for file in DATA_DIR.glob("*.csv")
        if "region" in file.name.lower()
    ]

if len(mapping_files) != 1:
    raise ValueError(
        f"Expected one Region mapping file, found: "
        f"{[f.name for f in mapping_files]}"
    )

mapping = pd.read_csv(mapping_files[0])

required_columns = {
    "bin", "count", "sum", "WOE"
}

if not required_columns.issubset(mapping.columns):
    raise ValueError(
        f"Unexpected mapping columns: {mapping.columns.tolist()}"
    )


# ============================================================
# FIT LOGISTIC REGRESSION
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs"
)

model.fit(X_train, y_train)

region_coefficient = model.coef_[0][FEATURES.index("Region")]


# ============================================================
# REGION ANALYSIS
# ============================================================

mapping["Default_Rate"] = (
    mapping["sum"] / mapping["count"]
)

mapping["Region_Coefficient"] = region_coefficient

mapping["Log_Odds_Contribution"] = (
    region_coefficient * mapping["WOE"]
)

mapping["Odds_Multiplier_vs_WOE_Zero"] = np.exp(
    mapping["Log_Odds_Contribution"]
)

# Sort from highest to lowest observed default rate
mapping = mapping.sort_values(
    "Default_Rate",
    ascending=False
)


# ============================================================
# OUTPUT
# ============================================================

output_columns = [
    "bin",
    "count",
    "Default_Rate",
    "WOE",
    "Region_Coefficient",
    "Log_Odds_Contribution",
    "Odds_Multiplier_vs_WOE_Zero"
]

results = mapping[output_columns].copy()

print("\n" + "=" * 75)
print("REGIONAL PD MODEL ANALYSIS")
print("=" * 75)

print(
    "\nRegion coefficient:",
    round(region_coefficient, 6)
)

print("\nREGIONAL RESULTS")

print(
    results.to_string(
        index=False,
        formatters={
            "Default_Rate": "{:.4%}".format,
            "WOE": "{:.6f}".format,
            "Region_Coefficient": "{:.6f}".format,
            "Log_Odds_Contribution": "{:.6f}".format,
            "Odds_Multiplier_vs_WOE_Zero": "{:.4f}".format
        }
    )
)

results.to_csv(
    OUTPUT_DIR / "regional_pd_analysis.csv",
    index=False
)

print("\nResults saved in:", OUTPUT_DIR)
print("\nREGIONAL ANALYSIS COMPLETED")

