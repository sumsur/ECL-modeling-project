import pandas as pd

df = pd.read_csv("C:\Python\ECL modeling\data_raw.csv")

df_model = df[
    df["credit_type"].astype(str).str.strip().str.upper() != "EQUI"
].copy()

variables = [
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges"
]

for col in variables:
    print(f"\n{'='*60}")
    print(f"VARIABLE: {col}")
    print("="*60)

    df_model[f"{col}_missing"] = df_model[col].isna()

    print(
        df_model.groupby("Status")[f"{col}_missing"]
        .agg(["count", "sum", "mean"])
    )

    print("\nDefault rate by missingness:")
    print(
        df_model.groupby(f"{col}_missing")["Status"]
        .agg(["count", "mean"])
    )
