from pathlib import Path
import pandas as pd

DATA_DIR = Path("data_raw/freddie_mac_sample")
OUT_DIR = Path("outputs/default_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

perf = pd.read_csv(DATA_DIR / "performance.csv")

# Convert delinquency status into a consistent format
perf["delq_sts"] = perf["delq_sts"].astype("string").str.strip()

# Numeric delinquency status
perf["delq_numeric"] = pd.to_numeric(
    perf["delq_sts"],
    errors="coerce"
)

# Development default definition: 90+ DPD or REO
perf["default_flag"] = (
    (perf["delq_numeric"] >= 3) |
    (perf["delq_sts"] == "RA")
).astype(int)

perf = perf.sort_values(["id_loan", "period"])

print("=" * 65)
print("DEFAULT EVENT ANALYSIS")
print("=" * 65)

# 1. Overall default observations
print("\n1. DEFAULT OBSERVATIONS")
print("Total monthly default observations:", perf["default_flag"].sum())

# 2. Unique loans ever in default
default_loans = perf.loc[
    perf["default_flag"] == 1, "id_loan"
].nunique()

print("Unique loans ever in default:", default_loans)

print(
    "Share of all loans ever in default:",
    round(default_loans / perf["id_loan"].nunique() * 100, 2),
    "%"
)

# 3. First default date per loan
first_default = (
    perf.loc[perf["default_flag"] == 1]
    .groupby("id_loan")
    .agg(
        first_default_period=("period", "min"),
        default_observations=("default_flag", "sum")
    )
    .reset_index()
)

print("\n2. FIRST DEFAULT PERIOD DISTRIBUTION")
print(
    first_default["first_default_period"]
    .value_counts()
    .sort_index()
    .to_string()
)

# 4. Default status by period
print("\n3. DEFAULT OBSERVATIONS BY PERIOD")
print(
    perf.groupby("period")["default_flag"]
    .agg(["sum", "count", "mean"])
    .to_string()
)

# 5. Loans returning to current status after default
default_ids = set(first_default["id_loan"])

after_default = perf[
    perf["id_loan"].isin(default_ids)
].copy()

after_default["had_default_before"] = (
    after_default.groupby("id_loan")["default_flag"]
    .transform(lambda x: x.cumsum().shift(fill_value=0) > 0)
)

cured = after_default[
    (after_default["had_default_before"]) &
    (after_default["delq_sts"] == "0")
]["id_loan"].nunique()

print("\n4. LOANS RETURNING TO CURRENT AFTER DEFAULT")
print("Loans with a later current-status observation:", cured)

# 6. Save results
first_default.to_csv(
    OUT_DIR / "first_default_by_loan.csv",
    index=False
)

perf.to_csv(
    OUT_DIR / "performance_with_default_flag.csv",
    index=False
)

print("\nFiles saved to:", OUT_DIR)
print("\nANALYSIS COMPLETED")

