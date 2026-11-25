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
    / "ead"
    / "ead_portfolio_202109.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "lifetime_pd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOAN_OUTPUT = OUTPUT_DIR / "lifetime_pd_portfolio_202109.csv"
CURVE_OUTPUT = OUTPUT_DIR / "lifetime_pd_annual_curve_202109.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "lifetime_pd_summary_202109.csv"

# ============================================================
# LOAD DATA
# ============================================================

print("\n===== LOADING EAD PORTFOLIO =====")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Portfolio observations: {len(df):,}")
print(f"Unique loans: {df['id_loan'].nunique():,}")

# ============================================================
# VALIDATE INPUTS
# ============================================================

required_columns = [
    "id_loan",
    "ead",
    "pd_12m",
    "mths_remng"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in ["ead", "pd_12m", "mths_remng"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[required_columns].isna().any().any():
    raise ValueError("Missing values detected in required fields.")

if not df["pd_12m"].between(0, 1).all():
    raise ValueError("PD values must be between 0 and 1.")

if (df["ead"] <= 0).any():
    raise ValueError("EAD must be positive.")

if (df["mths_remng"] <= 0).any():
    raise ValueError("Remaining maturity must be positive.")

if (df["mths_remng"] % 1 != 0).any():
    raise ValueError("Remaining maturity must be expressed in whole months.")

df["mths_remng"] = df["mths_remng"].astype(int)

# ============================================================
# LIFETIME PD CALCULATION
# ============================================================

print("\n===== LIFETIME PD CALCULATION =====")

# Constant monthly hazard implied by the 12-month PD.
# Survival over T months = (1 - PD_12m) ** (T / 12).

df["lifetime_pd"] = (
    1 - (1 - df["pd_12m"]) ** (df["mths_remng"] / 12)
)

df["lifetime_pd"] = df["lifetime_pd"].clip(0, 1)

df["lifetime_pd_method"] = (
    "Constant hazard extrapolation from 12-month PD"
)

# ============================================================
# BUILD ANNUAL MARGINAL PD CURVE
# ============================================================

print("\n===== BUILDING ANNUAL PD CURVE =====")

curve_parts = []

max_years = int(np.ceil(df["mths_remng"].max() / 12))

for year in range(1, max_years + 1):

    elapsed_before = (year - 1) * 12

    months_in_period = (
        df["mths_remng"] - elapsed_before
    ).clip(lower=0, upper=12)

    active = months_in_period > 0

    if not active.any():
        continue

    part = df.loc[
        active,
        ["id_loan", "ead", "pd_12m", "mths_remng"]
    ].copy()

    part["year"] = year
    part["months_in_period"] = months_in_period.loc[active]

    # Survival probability at the beginning of the period.
    part["survival_start"] = (
        (1 - part["pd_12m"]) ** (elapsed_before / 12)
    )

    # Conditional default probability during this period.
    part["conditional_pd"] = (
        1
        - (1 - part["pd_12m"]) **
        (part["months_in_period"] / 12)
    )

    # Marginal probability of default during this period.
    part["marginal_pd"] = (
        part["survival_start"] * part["conditional_pd"]
    )

    elapsed_end = (
        elapsed_before + part["months_in_period"]
    )

    part["cumulative_pd"] = (
        1 - (1 - part["pd_12m"]) ** (elapsed_end / 12)
    )

    curve_parts.append(part)

curve = pd.concat(curve_parts, ignore_index=True)

# ============================================================
# RECONCILIATION
# ============================================================

print("\n===== PD CURVE RECONCILIATION =====")

marginal_sum = (
    curve.groupby("id_loan")["marginal_pd"].sum()
)

reconciliation = df.set_index("id_loan")["lifetime_pd"]

difference = (
    marginal_sum - reconciliation
).abs()

print("Maximum absolute difference:", difference.max())

if difference.max() > 1e-10:
    raise ValueError(
        "Annual marginal PDs do not reconcile to lifetime PD."
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
        "Average 12-month PD",
        "EAD-weighted 12-month PD",
        "Average lifetime PD",
        "EAD-weighted lifetime PD",
        "Minimum lifetime PD",
        "Maximum lifetime PD",
        "Maximum remaining maturity (months)",
        "PD projection method"
    ],
    "Value": [
        "September 2021",
        len(df),
        total_ead,
        df["pd_12m"].mean(),
        (df["ead"] * df["pd_12m"]).sum() / total_ead,
        df["lifetime_pd"].mean(),
        (df["ead"] * df["lifetime_pd"]).sum() / total_ead,
        df["lifetime_pd"].min(),
        df["lifetime_pd"].max(),
        df["mths_remng"].max(),
        "Constant hazard extrapolation"
    ]
})

print("\n===== LIFETIME PD SUMMARY =====")
print(summary.to_string(index=False))

# ============================================================
# ANNUAL PORTFOLIO PD PROFILE
# ============================================================

annual_summary = (
    curve.groupby("year")
    .apply(
        lambda x: pd.Series({
            "active_loans": x["id_loan"].nunique(),
            "ead_at_start": x.drop_duplicates("id_loan")["ead"].sum(),
            "total_marginal_pd_weighted_ead": (
                x["marginal_pd"] * x["ead"]
            ).sum(),
            "ead_weighted_marginal_pd": (
                (x["marginal_pd"] * x["ead"]).sum()
                / x.drop_duplicates("id_loan")["ead"].sum()
            )
        }),
        include_groups=False
    )
    .reset_index()
)

print("\n===== ANNUAL MARGINAL PD PROFILE =====")
print(annual_summary.to_string(index=False))

# ============================================================
# EXPORT RESULTS
# ============================================================

df.to_csv(LOAN_OUTPUT, index=False)
curve.to_csv(CURVE_OUTPUT, index=False)
summary.to_csv(SUMMARY_OUTPUT, index=False)

annual_summary.to_csv(
    OUTPUT_DIR / "annual_pd_portfolio_summary.csv",
    index=False
)

print("\n===== OUTPUT FILES =====")
print(LOAN_OUTPUT)
print(CURVE_OUTPUT)
print(SUMMARY_OUTPUT)
print(OUTPUT_DIR / "annual_pd_portfolio_summary.csv")

print("\nLIFETIME PD ESTIMATION COMPLETED.")

