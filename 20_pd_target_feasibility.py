from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data_raw" / "freddie_mac_sample"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "default_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "performance.csv"
OUTPUT_FILE = OUTPUT_DIR / "pd_12m_target_feasibility.csv"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading performance data...", flush=True)

perf = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
    usecols=["id_loan", "period", "delq_sts"]
)

print(f"Loaded {len(perf):,} observations.", flush=True)


# --------------------------------------------------
# 3. CLEAN DATA
# --------------------------------------------------

perf["id_loan"] = perf["id_loan"].astype("string").str.strip()

perf["delq_sts"] = (
    perf["delq_sts"]
    .astype("string")
    .str.strip()
)

perf["period"] = pd.to_numeric(
    perf["period"], errors="coerce"
)

perf = perf.dropna(
    subset=["id_loan", "period"]
).copy()

# Convert YYYYMM into a monthly integer index.
year = (perf["period"] // 100).astype(int)
month = (perf["period"] % 100).astype(int)

perf["month_index"] = year * 12 + month

# Numeric delinquency status.
delq_numeric = pd.to_numeric(
    perf["delq_sts"], errors="coerce"
)

# Development default proxy:
# 90+ days delinquent or REO Acquisition.
perf["default_flag"] = (
    (delq_numeric >= 3) |
    (perf["delq_sts"] == "RA")
).astype("int8")


# --------------------------------------------------
# 4. SORT AND CHECK
# --------------------------------------------------

perf = perf.sort_values(
    ["id_loan", "month_index"]
).reset_index(drop=True)

if perf.duplicated(["id_loan", "month_index"]).any():
    raise ValueError(
        "Duplicate loan-month observations detected."
    )

print("Data cleaning completed.", flush=True)


# --------------------------------------------------
# 5. IDENTIFY FIRST DEFAULT MONTH
# --------------------------------------------------

first_default = (
    perf.loc[perf["default_flag"] == 1]
    .groupby("id_loan")["month_index"]
    .min()
)

perf["first_default_month"] = (
    perf["id_loan"].map(first_default)
)

# Last observation month in the dataset.
last_month = int(perf["month_index"].max())

# The last eligible observation month is 12 months
# before the end of the observed performance period.
cutoff_month = last_month - 12

print(
    f"Last performance month index: {last_month}",
    flush=True
)

print(
    f"Observation cutoff index: {cutoff_month}",
    flush=True
)


# --------------------------------------------------
# 6. CREATE ELIGIBLE OBSERVATIONS
# --------------------------------------------------

eligible = perf.loc[
    (perf["month_index"] <= cutoff_month) &
    (
        perf["first_default_month"].isna() |
        (perf["first_default_month"] > perf["month_index"])
    )
].copy()

print(
    f"Eligible observations: {len(eligible):,}",
    flush=True
)


# --------------------------------------------------
# 7. IDENTIFY DEFAULTS IN THE NEXT 12 MONTHS
# --------------------------------------------------

# Use a MultiIndex to perform efficient lookups.
# This avoids slow row-by-row processing.

status_index = pd.MultiIndex.from_frame(
    perf[["id_loan", "month_index"]]
)

default_values = pd.Series(
    perf["default_flag"].to_numpy(),
    index=status_index
)

observed_values = pd.Series(
    1,
    index=status_index,
    dtype="int8"
)

loan_ids = eligible["id_loan"].to_numpy()
observation_months = eligible["month_index"].to_numpy()

future_default = pd.Series(
    False, index=eligible.index
)

future_months_observed = pd.Series(
    0, index=eligible.index, dtype="int8"
)

print("Checking future 12-month windows...", flush=True)

for offset in range(1, 13):

    future_index = pd.MultiIndex.from_arrays(
        [
            loan_ids,
            observation_months + offset
        ],
        names=["id_loan", "month_index"]
    )

    future_status = default_values.reindex(future_index)

    future_observed = observed_values.reindex(future_index)

    future_default |= (
        future_status.fillna(0).to_numpy() == 1
    )

    future_months_observed += (
        future_observed.notna().to_numpy().astype("int8")
    )

    if offset in [4, 8, 12]:
        print(
            f"Processed {offset}/12 future months.",
            flush=True
        )


# --------------------------------------------------
# 8. DEFINE 12-MONTH PD TARGET
# --------------------------------------------------

# Positive target:
# At least one default event in the next 12 months.
#
# Negative target:
# All 12 future months observed and no default.
#
# Incomplete observations without a future default
# are excluded.

positive_target = future_default

negative_target = (
    (~future_default) &
    (future_months_observed == 12)
)

eligible["target_12m"] = pd.NA

eligible.loc[
    positive_target, "target_12m"
] = 1

eligible.loc[
    negative_target, "target_12m"
] = 0

final_data = eligible.loc[
    eligible["target_12m"].notna()
].copy()

final_data["target_12m"] = (
    final_data["target_12m"].astype("int8")
)


# --------------------------------------------------
# 9. SUMMARY
# --------------------------------------------------

print("\n========== 12-MONTH PD TARGET SUMMARY ==========")

print(f"Eligible observations: {len(eligible):,}")

print(f"Final observations: {len(final_data):,}")

print("\nTarget distribution:")

print(
    final_data["target_12m"].value_counts().sort_index()
)

print("\nTarget rate:")

print(
    f"{final_data['target_12m'].mean():.2%}"
)

print("\nUnique loans:")

print(final_data["id_loan"].nunique())

print("\nUnique loans with positive target:")

print(
    final_data.loc[
        final_data["target_12m"] == 1,
        "id_loan"
    ].nunique()
)


# --------------------------------------------------
# 10. MONTHLY SUMMARY
# --------------------------------------------------

monthly_summary = (
    final_data
    .groupby("month_index")
    .agg(
        observations=("target_12m", "size"),
        defaults=("target_12m", "sum"),
        target_rate=("target_12m", "mean"),
        unique_loans=("id_loan", "nunique")
    )
    .reset_index()
)

monthly_summary["period"] = (
    monthly_summary["month_index"]
    .apply(
        lambda x: f"{(x - 1) // 12:04d}{(x - 1) % 12 + 1:02d}"
    )
)

print("\nMonthly target summary:")

print(
    monthly_summary[
        [
            "period",
            "observations",
            "defaults",
            "target_rate",
            "unique_loans"
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 11. SAVE RESULTS
# --------------------------------------------------

final_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")

print(OUTPUT_FILE)

print("\nScript completed successfully.")


