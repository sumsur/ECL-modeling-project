import pandas as pd
from pathlib import Path

# Load data
df = pd.read_csv("C:\Python\ECL modeling\data_raw.csv")

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

# Numerical variables
numeric_vars = [
    "Credit_Score",
    "LTV",
    "income",
    "loan_amount",
    "term",
    "dtir1",
    "property_value"
]

# Categorical variables
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

# 1. Default rate by numerical variables
for var in numeric_vars:

    temp = df[[var, "Status"]].dropna().copy()

    temp["Group"] = pd.qcut(
        temp[var],
        q=5,
        duplicates="drop"
    )

    result = temp.groupby(
        "Group",
        observed=False
    ).agg(
        Number_of_Loans=("Status", "count"),
        Default_Rate=("Status", "mean")
    )

    result["Default_Rate"] *= 100

    print(f"\n===== DEFAULT RATE BY {var} =====")
    print(result)

    result.to_csv(
        OUTPUT / f"default_by_{var}.csv"
    )


# 2. Default rate by categorical variables
for var in categorical_vars:

    result = df.groupby(
        var,
        dropna=False,
        observed=False
    ).agg(
        Number_of_Loans=("Status", "count"),
        Default_Rate=("Status", "mean")
    )

    result["Default_Rate"] *= 100

    result = result.sort_values(
        "Default_Rate",
        ascending=False
    )

    print(f"\n===== DEFAULT RATE BY {var} =====")
    print(result)

    result.to_csv(
        OUTPUT / f"default_by_{var}.csv"
    )


print("\nDEFAULT RISK ANALYSIS COMPLETED")
