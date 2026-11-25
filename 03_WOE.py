import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. TARGET VARIABLE
# --------------------------------------------------

file_path = Path("C:\Python\ECL modeling\data_raw.csv")
df = pd.read_csv(file_path)

target = pd.to_numeric(df["Status"], errors="coerce")

print("Target distribution:")
print(target.value_counts(dropna=False))

# Assumption: 1 = Default, 0 = Non-default
if set(target.dropna().unique()) != {0, 1}:
    raise ValueError(
        "Status must be coded as 0 and 1. "
        "Check target values before continuing."
    )

# Remove rows with missing target
data = df.loc[target.notna()].copy()
data["Default"] = target.loc[target.notna()].astype(int)

# --------------------------------------------------
# 2. DEFINE VARIABLES
# --------------------------------------------------

numeric_vars = [
    "Credit_Score",
    "LTV",
    "income",
    "loan_amount",
    "term",
    "dtir1",
    "property_value"
]

categorical_vars = [
    "loan_type",
    "loan_purpose",
    "credit_type",
    "Region",
    "Gender",
    "age",
    "occupancy_type",
    "Neg_ammortization",
    "interest_only"
]

# --------------------------------------------------
# 3. WOE AND IV FUNCTION
# --------------------------------------------------

def calculate_woe_iv(data, variable, numeric=False):

    temp = data[[variable, "Default"]].copy()

    if numeric:

        values = pd.to_numeric(
            temp[variable], errors="coerce"
        )

        non_missing = values.notna()

        temp["Group"] = "Missing"

        if non_missing.sum() > 0:

            temp.loc[non_missing, "Group"] = (
                pd.qcut(
                    values[non_missing],
                    q=5,
                    duplicates="drop"
                ).astype(str).values
            )

    else:

        temp["Group"] = (
            temp[variable]
            .fillna("Missing")
            .astype(str)
        )

    # Count good and bad loans
    grouped = temp.groupby("Group", observed=True)["Default"]

    result = grouped.agg(
        Number_of_Loans="count",
        Defaults="sum"
    ).reset_index()

    result["Non_Defaults"] = (
        result["Number_of_Loans"] - result["Defaults"]
    )

    total_good = result["Non_Defaults"].sum()
    total_bad = result["Defaults"].sum()

    n_groups = len(result)

    # Smoothing to avoid division by zero
    alpha = 0.5

    result["Good_Distribution"] = (
        result["Non_Defaults"] + alpha
    ) / (total_good + alpha * n_groups)

    result["Bad_Distribution"] = (
        result["Defaults"] + alpha
    ) / (total_bad + alpha * n_groups)

    # WOE
    result["WOE"] = np.log(
        result["Good_Distribution"] /
        result["Bad_Distribution"]
    )

    # IV contribution
    result["IV_Contribution"] = (
        result["Good_Distribution"] -
        result["Bad_Distribution"]
    ) * result["WOE"]

    total_iv = result["IV_Contribution"].sum()

    return result, total_iv


# --------------------------------------------------
# 4. CALCULATE IV FOR ALL VARIABLES
# --------------------------------------------------

iv_results = []
woe_tables = {}

for variable in numeric_vars:

    if variable in data.columns:

        table, iv = calculate_woe_iv(
            data, variable, numeric=True
        )

        woe_tables[variable] = table

        iv_results.append({
            "Variable": variable,
            "IV": iv
        })


for variable in categorical_vars:

    if variable in data.columns:

        table, iv = calculate_woe_iv(
            data, variable, numeric=False
        )

        woe_tables[variable] = table

        iv_results.append({
            "Variable": variable,
            "IV": iv
        })


# --------------------------------------------------
# 5. IV SUMMARY
# --------------------------------------------------

iv_summary = pd.DataFrame(iv_results)

iv_summary = iv_summary.sort_values(
    "IV", ascending=False
).reset_index(drop=True)


def iv_interpretation(iv):

    if iv < 0.02:
        return "Not predictive"
    elif iv < 0.10:
        return "Weak"
    elif iv < 0.30:
        return "Medium"
    elif iv < 0.50:
        return "Strong"
    else:
        return "Suspicious / Very strong"


iv_summary["Interpretation"] = (
    iv_summary["IV"].apply(iv_interpretation)
)

print("\n========== WOE / IV SUMMARY ==========")
print(iv_summary.to_string(index=False))


# --------------------------------------------------
# 6. EXPORT RESULTS
# --------------------------------------------------

iv_summary.to_csv(
    "woe_iv_summary.csv",
    index=False
)

for variable, table in woe_tables.items():

    table.to_csv(
        f"woe_iv_detail_{variable}.csv",
        index=False
    )

print("\nWOE / IV ANALYSIS COMPLETED")

