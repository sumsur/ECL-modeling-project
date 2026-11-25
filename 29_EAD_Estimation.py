import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "pd_model"
    / "portfolio_pd_snapshot_202109.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "ead"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ead_portfolio_202109.csv"
SUMMARY_FILE = OUTPUT_DIR / "ead_summary_202109.csv"
SEGMENT_FILE = OUTPUT_DIR / "ead_by_remaining_term.csv"

# ============================================================
# LOAD PORTFOLIO
# ============================================================

print("\n===== LOADING PORTFOLIO =====")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Portfolio observations: {len(df):,}")
print(f"Unique loans: {df['id_loan'].nunique():,}")

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "id_loan",
    "curr_act_upb",
    "loan_age",
    "mths_remng",
    "pd_12m"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ============================================================
# DATA QUALITY CHECKS
# ============================================================

print("\n===== DATA QUALITY CHECKS =====")

if df["id_loan"].duplicated().any():
    duplicates = df.loc[
        df["id_loan"].duplicated(keep=False),
        "id_loan"
    ].nunique()

    raise ValueError(
        f"Portfolio contains {duplicates} duplicated loan IDs."
    )

df["curr_act_upb"] = pd.to_numeric(
    df["curr_act_upb"],
    errors="coerce"
)

df["pd_12m"] = pd.to_numeric(
    df["pd_12m"],
    errors="coerce"
)

df["loan_age"] = pd.to_numeric(
    df["loan_age"],
    errors="coerce"
)

df["mths_remng"] = pd.to_numeric(
    df["mths_remng"],
    errors="coerce"
)

print("Missing current UPB:", df["curr_act_upb"].isna().sum())
print("Non-positive current UPB:", (df["curr_act_upb"] <= 0).sum())
print("Missing PD:", df["pd_12m"].isna().sum())

if df["curr_act_upb"].isna().any():
    raise ValueError("Missing current UPB values detected.")

if (df["curr_act_upb"] < 0).any():
    raise ValueError("Negative current UPB values detected.")

if df["pd_12m"].isna().any():
    raise ValueError("Missing PD values detected.")

if not df["pd_12m"].between(0, 1).all():
    raise ValueError("PD values must be between 0 and 1.")

# ============================================================
# EAD ESTIMATION
# ============================================================

print("\n===== EAD ESTIMATION =====")

# Current outstanding principal is used as the EAD proxy.
df["ead"] = df["curr_act_upb"]

df["ead_method"] = (
    "Current outstanding principal (curr_act_upb)"
)

# ============================================================
# PORTFOLIO SUMMARY
# ============================================================

total_ead = df["ead"].sum()

summary = pd.DataFrame({
    "Metric": [
        "Reporting period",
        "Number of loans",
        "Total EAD",
        "Average EAD",
        "Median EAD",
        "Minimum EAD",
        "Maximum EAD",
        "Total PD-weighted EAD",
        "EAD-weighted average PD"
    ],
    "Value": [
        "September 2021",
        len(df),
        total_ead,
        df["ead"].mean(),
        df["ead"].median(),
        df["ead"].min(),
        df["ead"].max(),
        (df["ead"] * df["pd_12m"]).sum(),
        (df["ead"] * df["pd_12m"]).sum() / total_ead
    ]
})

print(summary.to_string(index=False))

# ============================================================
# EAD BY REMAINING MATURITY
# ============================================================

df["remaining_term_group"] = pd.cut(
    df["mths_remng"],
    bins=[-1, 60, 120, 180, 240, 300, 360, np.inf],
    labels=[
        "0-60 months",
        "61-120 months",
        "121-180 months",
        "181-240 months",
        "241-300 months",
        "301-360 months",
        "Over 360 months"
    ]
)

term_summary = (
    df.groupby(
        "remaining_term_group",
        observed=False
    )
    .agg(
        number_of_loans=("id_loan", "count"),
        total_ead=("ead", "sum"),
        average_ead=("ead", "mean"),
        average_pd=("pd_12m", "mean")
    )
    .reset_index()
)

term_summary["ead_share"] = (
    term_summary["total_ead"] / total_ead
)

print("\n===== EAD BY REMAINING TERM =====")
print(term_summary.to_string(index=False))

# ============================================================
# EXPORT RESULTS
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)
term_summary.to_csv(SEGMENT_FILE, index=False)

print("\n===== OUTPUT FILES =====")
print(OUTPUT_FILE)
print(SUMMARY_FILE)
print(SEGMENT_FILE)

print("\nEAD ESTIMATION COMPLETED.")

