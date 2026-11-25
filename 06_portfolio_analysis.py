import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. PATHS AND OUTPUT FOLDER
# ============================================================

DATA_PATH = Path("C:\Python\ECL modeling\data_raw.csv")

OUTPUT_DIR = Path("outputs/portfolio_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("PORTFOLIO RISK ANALYSIS")
print("=" * 60)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. DATA PREPARATION
# ============================================================

required_columns = [
    "Status",
    "loan_amount",
    "credit_type",
    "loan_type",
    "loan_purpose",
    "Region",
    "Gender",
    "occupancy_type",
    "Neg_ammortization",
    "interest_only",
    "age",
    "Credit_Score",
    "income",
    "dtir1"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

# Convert numeric columns
df["Status"] = pd.to_numeric(df["Status"], errors="coerce")
df["loan_amount"] = pd.to_numeric(
    df["loan_amount"], errors="coerce"
)

# Keep observations with valid binary target
df = df[df["Status"].isin([0, 1])].copy()

# Create exposure-related helper columns
df["Default_Loan_Amount"] = np.where(
    df["Status"] == 1,
    df["loan_amount"].fillna(0),
    0
)

df["Non_Default_Loan_Amount"] = np.where(
    df["Status"] == 0,
    df["loan_amount"].fillna(0),
    0
)

print(f"Observations after cleaning: {len(df):,}")


# ============================================================
# 3. DEFINE PORTFOLIO SAMPLES
# ============================================================

samples = {
    "Full_Portfolio": df.copy(),
    "Without_EQUI": df[
        df["credit_type"] != "EQUI"
    ].copy()
}


# ============================================================
# 4. PORTFOLIO OVERVIEW
# ============================================================

def portfolio_overview(data, sample_name):

    total_loans = len(data)

    defaults = int((data["Status"] == 1).sum())

    non_defaults = int((data["Status"] == 0).sum())

    total_loan_amount = data["loan_amount"].sum()

    default_loan_amount = data[
        "Default_Loan_Amount"
    ].sum()

    default_rate = (
        defaults / total_loans * 100
        if total_loans > 0 else 0
    )

    default_amount_share = (
        default_loan_amount / total_loan_amount * 100
        if total_loan_amount > 0 else 0
    )

    return {
        "Sample": sample_name,
        "Number_of_Loans": total_loans,
        "Defaults": defaults,
        "Non_Defaults": non_defaults,
        "Default_Rate": default_rate,
        "Total_Loan_Amount": total_loan_amount,
        "Default_Loan_Amount": default_loan_amount,
        "Default_Loan_Amount_Share": default_amount_share
    }


overview_results = []

for sample_name, sample_data in samples.items():

    overview_results.append(
        portfolio_overview(sample_data, sample_name)
    )

overview_df = pd.DataFrame(overview_results)

overview_df.to_csv(
    OUTPUT_DIR / "portfolio_overview.csv",
    index=False
)

print("\nPORTFOLIO OVERVIEW")
print(overview_df.round(2).to_string(index=False))


# ============================================================
# 5. EXPOSURE BY DEFAULT STATUS
# ============================================================

for sample_name, sample_data in samples.items():

    exposure = (
        sample_data
        .groupby("Status")
        .agg(
            Number_of_Loans=("Status", "size"),
            Total_Loan_Amount=("loan_amount", "sum")
        )
        .reset_index()
    )

    exposure["Status_Label"] = exposure["Status"].map({
        0: "Non-Default",
        1: "Default"
    })

    exposure.to_csv(
        OUTPUT_DIR / f"{sample_name}_exposure_by_status.csv",
        index=False
    )


# ============================================================
# 6. SEGMENT ANALYSIS FUNCTION
# ============================================================

def segment_analysis(data, variable, sample_name, analysis_type):

    temp = data.copy()

    # Remove missing group values only for grouping.
    # Missing values are retained as a separate category.
    temp[variable] = temp[variable].fillna("Missing").astype(str)

    # Aggregate portfolio information by segment
    summary = (
        temp.groupby(variable, dropna=False)
        .agg(
            Number_of_Loans=("Status", "size"),
            Defaults=("Status", "sum"),
            Total_Loan_Amount=("loan_amount", "sum"),
            Default_Loan_Amount=("Default_Loan_Amount", "sum")
        )
        .reset_index()
    )

    # Calculate risk and portfolio metrics
    summary["Default_Rate"] = (
        summary["Defaults"] /
        summary["Number_of_Loans"] * 100
    )

    total_loans = len(temp)

    total_loan_amount = temp["loan_amount"].sum()

    summary["Portfolio_Share"] = (
        summary["Number_of_Loans"] /
        total_loans * 100
    )

    summary["Default_Loan_Amount_Share"] = (
        summary["Default_Loan_Amount"] /
        total_loan_amount * 100
        if total_loan_amount > 0 else 0
    )

    # Save results
    file_name = (
        f"{sample_name}_{analysis_type}_{variable}.csv"
    )

    summary.to_csv(
        OUTPUT_DIR / file_name,
        index=False
    )

    # Plot default rate
    plt.figure(figsize=(10, 6))

    plt.bar(
        summary[variable].astype(str),
        summary["Default_Rate"]
    )

    plt.title(
        f"Default Rate by {variable} - {sample_name}"
    )

    plt.xlabel(variable)
    plt.ylabel("Default Rate (%)")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    chart_name = (
        f"{sample_name}_{analysis_type}_{variable}.png"
    )

    plt.savefig(
        OUTPUT_DIR / chart_name,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return summary


# ============================================================
# 7. CATEGORICAL SEGMENT ANALYSIS
# ============================================================

categorical_variables = [
    "loan_type",
    "loan_purpose",
    "Region",
    "Gender",
    "occupancy_type",
    "Neg_ammortization",
    "interest_only",
    "age"
]

for sample_name, sample_data in samples.items():

    print(f"\nCATEGORICAL ANALYSIS: {sample_name}")

    for variable in categorical_variables:

        result = segment_analysis(
            sample_data,
            variable,
            sample_name,
            "categorical"
        )

        print(f"\nDefault rate by {variable}:")
        print(
            result[
                [variable, "Number_of_Loans", "Defaults", "Default_Rate"]
            ].round(2).to_string(index=False)
        )


# ============================================================
# 8. NUMERICAL SEGMENT ANALYSIS
# ============================================================

numerical_variables = [
    "Credit_Score",
    "income",
    "loan_amount",
    "dtir1"
]

for sample_name, sample_data in samples.items():

    print(f"\nNUMERICAL ANALYSIS: {sample_name}")

    for variable in numerical_variables:

        temp = sample_data.copy()

        temp[variable] = pd.to_numeric(
            temp[variable],
            errors="coerce"
        )

        # Create five quantile-based groups
        temp["Risk_Group"] = pd.qcut(
            temp[variable],
            q=5,
            duplicates="drop"
        )

        # Convert intervals to strings and retain missing values
        temp["Risk_Group"] = (
            temp["Risk_Group"]
            .astype(str)
            .replace("nan", "Missing")
        )

        result = segment_analysis(
            temp,
            "Risk_Group",
            sample_name,
            f"numerical_{variable}"
        )

        print(f"\nDefault rate by {variable}:")
        print(
            result[
                [
                    "Risk_Group",
                    "Number_of_Loans",
                    "Defaults",
                    "Default_Rate",
                    "Default_Loan_Amount_Share"
                ]
            ].round(2).to_string(index=False)
        )


# ============================================================
# 9. PORTFOLIO DEFAULT RATE COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    overview_df["Sample"],
    overview_df["Default_Rate"]
)

plt.title("Portfolio Default Rate Comparison")
plt.xlabel("Portfolio Sample")
plt.ylabel("Default Rate (%)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "portfolio_default_rate_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PORTFOLIO ANALYSIS COMPLETED")
print("=" * 60)

print(f"All results saved in: {OUTPUT_DIR.resolve()}")

print("\nGenerated files include:")
print("- Portfolio overview")
print("- Exposure by default status")
print("- Categorical segment analysis")
print("- Numerical segment analysis")
print("- Default rate charts")
print("- Full portfolio vs Without EQUI comparison")

