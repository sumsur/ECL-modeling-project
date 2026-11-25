import pandas as pd

df = pd.read_csv("C:\Python\ECL modeling\data_raw.csv")

df = df[
    df["credit_type"].astype(str).str.strip().str.upper() != "EQUI"
].copy()

check_variables = [
    "Credit_Worthiness",
    "approv_in_adv",
    "loan_limit",
    "submission_of_application",
    "business_or_commercial",
    "open_credit",
    "Secured_by",
    "Security_Type",
    "co-applicant_credit_type"
]

for col in check_variables:

    print(f"\n{'='*60}")
    print(f"VARIABLE: {col}")
    print("="*60)

    print(
        pd.crosstab(
            df[col],
            df["Status"],
            normalize="index"
        ).round(4) * 100
    )