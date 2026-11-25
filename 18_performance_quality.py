from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_DIR = Path("data_raw/freddie_mac_sample")

orig = pd.read_csv(DATA_DIR / "origination.csv")
perf = pd.read_csv(DATA_DIR / "performance.csv")

# Standardize delinquency status to avoid mixed-type sorting errors
perf["delq_sts"] = (
    perf["delq_sts"]
    .astype("string")
    .fillna("<MISSING>")
)

print("=" * 70)
print("FREDDIE MAC - PERFORMANCE DATA QUALITY ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 2. PERIOD RANGE
# --------------------------------------------------

print("\n1. PERIOD RANGE")

print("First period:", perf["period"].min())
print("Last period:", perf["period"].max())
print("Number of unique periods:", perf["period"].nunique())

print("\nAll periods:")
print(sorted(perf["period"].unique()))


# --------------------------------------------------
# 3. DELINQUENCY STATUS DISTRIBUTION
# --------------------------------------------------

print("\n2. DELINQUENCY STATUS DISTRIBUTION")

delq_distribution = (
    perf["delq_sts"]
    .value_counts(dropna=False)
    .sort_index()
)

print(delq_distribution.to_string())

print("\nPercentage distribution:")

print(
    (perf["delq_sts"].value_counts(normalize=True, dropna=False) * 100)
    .sort_index()
    .round(2)
    .to_string()
)


# --------------------------------------------------
# 4. NUMBER OF MONTHLY RECORDS PER LOAN
# --------------------------------------------------

print("\n3. NUMBER OF MONTHLY RECORDS PER LOAN")

records = perf.groupby("id_loan").size()

print("\nDescriptive statistics:")
print(records.describe())

print("\nLoans by number of monthly records:")
print(records.value_counts().sort_index().to_string())


# --------------------------------------------------
# 5. DUPLICATE LOAN-MONTH RECORDS
# --------------------------------------------------

print("\n4. DUPLICATE LOAN-MONTH RECORDS")

duplicates = perf.duplicated(
    subset=["id_loan", "period"]
).sum()

print("Duplicate loan-period rows:", duplicates)


# --------------------------------------------------
# 6. LOAN ID MATCHING
# --------------------------------------------------

print("\n5. LOAN ID MATCHING")

orig_ids = set(orig["id_loan"])
perf_ids = set(perf["id_loan"])

print("Origination loans:", len(orig_ids))
print("Performance loans:", len(perf_ids))
print("Matched loans:", len(orig_ids & perf_ids))
print("Origination loans without performance:", len(orig_ids - perf_ids))
print("Performance loans without origination:", len(perf_ids - orig_ids))


# --------------------------------------------------
# 7. DELINQUENCY STATUS BY PERIOD
# --------------------------------------------------

print("\n6. DELINQUENCY STATUS BY PERIOD")

delq_by_period = pd.crosstab(
    perf["period"],
    perf["delq_sts"]
)

print(delq_by_period.to_string())


# --------------------------------------------------
# 8. NON-ZERO DELINQUENCY RECORDS
# --------------------------------------------------

print("\n7. SAMPLE OF NON-ZERO DELINQUENCY RECORDS")

nonzero = perf[
    ~perf["delq_sts"].isin(["0", "<MISSING>"])
]

print(
    nonzero[
        [
            "id_loan",
            "period",
            "delq_sts",
            "loan_age",
            "curr_act_upb"
        ]
    ].head(20).to_string(index=False)
)


# --------------------------------------------------
# 9. MISSING VALUES IN KEY VARIABLES
# --------------------------------------------------

print("\n8. MISSING VALUES IN KEY VARIABLES")

key_variables = [
    "id_loan",
    "period",
    "delq_sts",
    "loan_age",
    "curr_act_upb",
    "eltv"
]

print(
    perf[key_variables]
    .isnull()
    .sum()
    .to_string()
)


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)


