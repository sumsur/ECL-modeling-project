import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "lifetime_pd"
    / "lifetime_pd_portfolio_202109.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "ecl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOAN_OUTPUT = OUTPUT_DIR / "ecl_loan_level_202109.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "ecl_scenario_summary_202109.csv"
ANNUAL_OUTPUT = OUTPUT_DIR / "ecl_annual_breakdown_202109.csv"

LGD_SCENARIOS = {
    "Optimistic": 0.20,
    "Base": 0.35,
    "Adverse": 0.50
}

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n===== LOADING LIFETIME PD PORTFOLIO =====")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Portfolio observations: {len(df):,}")
print(f"Unique loans: {df['id_loan'].nunique():,}")

# ============================================================
# 3. VALIDATE INPUT DATA
# ============================================================

required_columns = [
    "id_loan",
    "ead",
    "pd_12m",
    "mths_remng",
    "orig_int_rt"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

for col in ["ead", "pd_12m", "mths_remng", "orig_int_rt"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[required_columns].isna().any().any():
    raise ValueError("Missing values in required ECL inputs.")

if df["id_loan"].duplicated().any():
    raise ValueError("Duplicate loan IDs detected.")

if not df["pd_12m"].between(0, 1).all():
    raise ValueError("PD must be between 0 and 1.")

if (df["ead"] <= 0).any():
    raise ValueError("EAD must be positive.")

if (df["mths_remng"] <= 0).any():
    raise ValueError("Remaining maturity must be positive.")

if (df["orig_int_rt"] <= -100).any():
    raise ValueError("Invalid interest rate.")

df["mths_remng"] = df["mths_remng"].astype(int)

# Original interest rate is stored as a percentage.
df["discount_rate"] = df["orig_int_rt"] / 100

print("\nInput validation completed.")

# ============================================================
# 4. ECL CALCULATION
# ============================================================

print("\n===== ECL CALCULATION =====")

loan_results = []
annual_results = []

for row in df.itertuples(index=False):

    loan_id = row.id_loan
    initial_ead = float(row.ead)
    pd_12m = float(row.pd_12m)
    maturity = int(row.mths_remng)
    annual_rate = float(row.discount_rate)

    # --------------------------------------------------------
    # Monthly default probability
    # --------------------------------------------------------

    monthly_hazard = (
        1 - (1 - pd_12m) ** (1 / 12)
    )

    months = np.arange(1, maturity + 1)

    survival_probability = (
        (1 - monthly_hazard) ** (months - 1)
    )

    marginal_pd = (
        survival_probability * monthly_hazard
    )

    # --------------------------------------------------------
    # Amortizing EAD projection
    # --------------------------------------------------------

    monthly_rate = annual_rate / 12

    if abs(monthly_rate) < 1e-12:

        monthly_payment = initial_ead / maturity

        projected_ead = (
            initial_ead - monthly_payment * (months - 1)
        )

    else:

        monthly_payment = (
            initial_ead * monthly_rate
            / (1 - (1 + monthly_rate) ** (-maturity))
        )

        projected_ead = (
            initial_ead * (1 + monthly_rate) ** (months - 1)
            - monthly_payment
            * (
                (1 + monthly_rate) ** (months - 1) - 1
            )
            / monthly_rate
        )

    projected_ead = np.maximum(projected_ead, 0)

    # --------------------------------------------------------
    # Discounting
    # --------------------------------------------------------

    # Mid-year convention: expected loss is discounted
    # from the approximate middle of each monthly period.

    discount_factor = (
        (1 + annual_rate) ** ((months - 0.5) / 12)
    )

    discounted_loss_exposure = (
        marginal_pd * projected_ead / discount_factor
    )

    # --------------------------------------------------------
    # Loan-level results
    # --------------------------------------------------------

    first_12_months = months <= 12

    result = {
        "id_loan": loan_id,
        "initial_ead": initial_ead,
        "pd_12m": pd_12m,
        "lifetime_pd": 1 - (1 - pd_12m) ** (maturity / 12),
        "remaining_maturity_months": maturity,
        "original_interest_rate": row.orig_int_rt,
        "monthly_hazard": monthly_hazard,
        "discount_rate_proxy": annual_rate,
        "projected_ead_at_maturity": projected_ead[-1],
        "lifetime_marginal_pd_sum": marginal_pd.sum()
    }

    for scenario, lgd in LGD_SCENARIOS.items():

        ecl_12m = (
            discounted_loss_exposure[first_12_months].sum()
            * lgd
        )

        ecl_lifetime = (
            discounted_loss_exposure.sum()
            * lgd
        )

        result[f"lgd_{scenario.lower()}"] = lgd
        result[f"ecl_12m_{scenario.lower()}"] = ecl_12m
        result[f"ecl_lifetime_{scenario.lower()}"] = ecl_lifetime

    loan_results.append(result)

    # --------------------------------------------------------
    # Annual breakdown
    # --------------------------------------------------------

    years = ((months - 1) // 12) + 1

    for year in np.unique(years):

        mask = years == year

        annual_row = {
            "id_loan": loan_id,
            "year": int(year),
            "marginal_pd": marginal_pd[mask].sum(),
            "ead_weighted_marginal_pd": (
                marginal_pd[mask] * projected_ead[mask]
            ).sum(),
            "average_projected_ead": projected_ead[mask].mean()
        }

        for scenario, lgd in LGD_SCENARIOS.items():

            annual_row[
                f"ecl_{scenario.lower()}"
            ] = (
                discounted_loss_exposure[mask].sum() * lgd
            )

        annual_results.append(annual_row)

# ============================================================
# 5. CREATE RESULT DATAFRAMES
# ============================================================

loan_df = pd.DataFrame(loan_results)
annual_df = pd.DataFrame(annual_results)

total_ead = loan_df["initial_ead"].sum()

# ============================================================
# 6. PORTFOLIO ECL SUMMARY
# ============================================================

summary_rows = []

for scenario, lgd in LGD_SCENARIOS.items():

    ecl_12m = loan_df[
        f"ecl_12m_{scenario.lower()}"
    ].sum()

    ecl_lifetime = loan_df[
        f"ecl_lifetime_{scenario.lower()}"
    ].sum()

    summary_rows.append({
        "Reporting_period": "September 2021",
        "Scenario": scenario,
        "LGD": lgd,
        "Portfolio_EAD": total_ead,
        "ECL_12m": ecl_12m,
        "ECL_lifetime": ecl_lifetime,
        "ECL_12m_ratio": ecl_12m / total_ead,
        "ECL_lifetime_ratio": ecl_lifetime / total_ead
    })

summary_df = pd.DataFrame(summary_rows)

# ============================================================
# 7. ANNUAL PORTFOLIO BREAKDOWN
# ============================================================

annual_summary = (
    annual_df
    .groupby("year")
    .agg(
        active_loans=("id_loan", "nunique"),
        marginal_pd=("marginal_pd", "mean"),
        total_ead_weighted_marginal_pd=(
            "ead_weighted_marginal_pd", "sum"
        ),
        average_projected_ead=(
            "average_projected_ead", "mean"
        ),
        ecl_optimistic=("ecl_optimistic", "sum"),
        ecl_base=("ecl_base", "sum"),
        ecl_adverse=("ecl_adverse", "sum")
    )
    .reset_index()
)

# ============================================================
# 8. VALIDATION CHECKS
# ============================================================

print("\n===== ECL VALIDATION =====")

if not (
    loan_df["ecl_12m_base"]
    <= loan_df["ecl_lifetime_base"] + 1e-8
).all():

    raise ValueError(
        "12-month ECL exceeds lifetime ECL for at least one loan."
    )

if not (
    loan_df["ecl_lifetime_optimistic"]
    <= loan_df["ecl_lifetime_base"]
).all():

    raise ValueError("LGD scenario ordering is incorrect.")

if not (
    loan_df["ecl_lifetime_base"]
    <= loan_df["ecl_lifetime_adverse"]
).all():

    raise ValueError("LGD scenario ordering is incorrect.")

if not np.allclose(
    loan_df["lifetime_marginal_pd_sum"],
    loan_df["lifetime_pd"],
    atol=1e-10
):

    raise ValueError(
        "Marginal PD does not reconcile to lifetime PD."
    )

print("12-month vs lifetime ECL: PASSED")
print("LGD scenario ordering: PASSED")
print("Lifetime PD reconciliation: PASSED")

# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n===== PORTFOLIO ECL RESULTS =====")

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:,.6f}"
    )
)

print("\n===== ANNUAL ECL PROFILE =====")

print(
    annual_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:,.4f}"
    )
)

# ============================================================
# 10. EXPORT RESULTS
# ============================================================

loan_df.to_csv(LOAN_OUTPUT, index=False)

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

annual_summary.to_csv(
    ANNUAL_OUTPUT,
    index=False
)

print("\n===== OUTPUT FILES =====")
print(LOAN_OUTPUT)
print(SUMMARY_OUTPUT)
print(ANNUAL_OUTPUT)

print("\nECL CALCULATION COMPLETED SUCCESSFULLY.")

