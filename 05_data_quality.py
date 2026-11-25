import pandas as pd

df = pd.read_csv("C:\Python\ECL modeling\data_raw.csv")

# 1. Investigate credit_type
print("\n===== CREDIT TYPE VS STATUS =====")

print(pd.crosstab(
    df["credit_type"],
    df["Status"],
    margins=True
))

print("\n===== DEFAULT RATE BY CREDIT TYPE =====")

print(
    df.groupby("credit_type", dropna=False)["Status"]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
)

# Inspect EQUI observations
print("\n===== EQUI STATUS DISTRIBUTION =====")

equivalent = df[df["credit_type"] == "EQUI"]

print(equivalent["Status"].value_counts(dropna=False))
print(equivalent[["credit_type", "Credit_Score",
                  "loan_amount", "Status"]].head(15))


# 2. Investigate LTV
print("\n===== LTV DESCRIPTIVE STATISTICS =====")

print(df["LTV"].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50,
                 0.75, 0.95, 0.99]
))

print("\n===== EXTREME LTV OBSERVATIONS =====")

print(
    df.nlargest(15, "LTV")[
        ["loan_amount", "property_value", "LTV", "Status"]
    ]
)


# 3. Investigate property_value
print("\n===== PROPERTY VALUE STATISTICS =====")

print(df["property_value"].describe())

print("\n===== ZERO OR NEGATIVE PROPERTY VALUES =====")

print(
    df[df["property_value"] <= 0][
        ["loan_amount", "property_value", "LTV", "Status"]
    ].head(20)
)


# 4. Investigate Credit Score
print("\n===== CREDIT SCORE BY STATUS =====")

print(
    df.groupby("Status")["Credit_Score"]
    .describe()
)


# 5. Investigate missing values by Status
print("\n===== MISSING VALUES BY STATUS =====")

for col in ["Credit_Score", "LTV", "income",
            "property_value", "dtir1", "credit_type"]:

    print(f"\n{col}")

    print(
        df.groupby("Status")[col]
        .apply(lambda x: x.isna().mean() * 100)
    )


print("\nDATA QUALITY INVESTIGATION COMPLETED")

