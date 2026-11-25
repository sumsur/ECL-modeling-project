import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_FILE = (
    BASE_DIR
    / "data_raw"
    / "freddie_mac_sample"
    / "performance.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "lgd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("\n===== LOADING PERFORMANCE DATA =====")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Rows: {len(df):,}")
print(f"Unique loans: {df['id_loan'].nunique():,}")

# ============================================================
# ZERO BALANCE CODE DISTRIBUTION
# ============================================================

print("\n===== ZERO BALANCE CODE DISTRIBUTION =====")

print(
    df["cd_zero_bal"]
    .value_counts(dropna=False)
    .sort_index()
)

# ============================================================
# IDENTIFY RELEVANT LOSS EVENTS
# ============================================================

loss_codes = [2, 3, 9, 15]

loss_events = df[df["cd_zero_bal"].isin(loss_codes)].copy()

print("\n===== LOSS EVENT SUMMARY =====")

print(f"Rows with relevant codes: {len(loss_events):,}")
print(f"Unique loans: {loss_events['id_loan'].nunique():,}")

# ============================================================
# MISSINGNESS IN LOSS VARIABLES
# ============================================================

loss_columns = [
    "zb_removal_upb",
    "dlq_acrd_int",
    "net_sale_proceeds",
    "mi_recoveries",
    "non_mi_recoveries",
    "expenses",
    "actual_loss",
    "dt_dfct_setlmt"
]

print("\n===== LOSS VARIABLE MISSINGNESS =====")

print(
    loss_events[loss_columns]
    .isna()
    .sum()
    .to_string()
)

# ============================================================
# ACTUAL LOSS RECONSTRUCTION
# ============================================================

loss_events["calculated_loss"] = (
    loss_events["zb_removal_upb"]
    + loss_events["dlq_acrd_int"]
    - loss_events["net_sale_proceeds"]
    - loss_events["mi_recoveries"]
    - loss_events["non_mi_recoveries"]
    - loss_events["expenses"]
)

loss_events["loss_difference"] = (
    loss_events["actual_loss"]
    - loss_events["calculated_loss"]
)

complete_mask = loss_events[
    [
        "zb_removal_upb",
        "dlq_acrd_int",
        "net_sale_proceeds",
        "mi_recoveries",
        "non_mi_recoveries",
        "expenses",
        "actual_loss"
    ]
].notna().all(axis=1)

reconciled = loss_events.loc[complete_mask].copy()

print("\n===== ACTUAL LOSS RECONCILIATION =====")

print(f"Complete observations: {len(reconciled):,}")

if len(reconciled) > 0:
    print(
        "Mean absolute difference:",
        round(reconciled["loss_difference"].abs().mean(), 4)
    )

    print(
        "Maximum absolute difference:",
        round(reconciled["loss_difference"].abs().max(), 4)
    )

    print(
        "Correlation:",
        round(
            reconciled["actual_loss"].corr(
                reconciled["calculated_loss"]
            ),
            6
        )
    )

# ============================================================
# ACTUAL LOSS AND EXPOSURE SUMMARY
# ============================================================

valid_lgd = loss_events[
    ["actual_loss", "zb_removal_upb"]
].notna().all(axis=1)

valid_lgd &= loss_events["zb_removal_upb"] > 0

lgd_data = loss_events.loc[valid_lgd].copy()

lgd_data["realized_lgd"] = (
    lgd_data["actual_loss"]
    / lgd_data["zb_removal_upb"]
)

print("\n===== REALIZED LGD SUMMARY =====")

if len(lgd_data) > 0:
    print(lgd_data["realized_lgd"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ))

    print(
        "\nNegative LGD observations:",
        (lgd_data["realized_lgd"] < 0).sum()
    )

    print(
        "LGD above 100%:",
        (lgd_data["realized_lgd"] > 1).sum()
    )

# ============================================================
# BREAKDOWN BY ZERO BALANCE CODE
# ============================================================

print("\n===== LGD BY ZERO BALANCE CODE =====")

if len(lgd_data) > 0:
    summary = (
        lgd_data
        .groupby("cd_zero_bal")
        .agg(
            observations=("id_loan", "count"),
            unique_loans=("id_loan", "nunique"),
            avg_loss=("actual_loss", "mean"),
            avg_exposure=("zb_removal_upb", "mean"),
            avg_lgd=("realized_lgd", "mean"),
            median_lgd=("realized_lgd", "median"),
            exposure_weighted_lgd=(
                "actual_loss",
                lambda x: x.sum()
            )
        )
    )

    exposure = (
        lgd_data.groupby("cd_zero_bal")["zb_removal_upb"].sum()
    )

    summary["exposure_weighted_lgd"] = (
        summary["exposure_weighted_lgd"] / exposure
    )

    print(summary.to_string())

    summary.to_csv(
        OUTPUT_DIR / "lgd_by_zero_balance_code.csv"
    )

# ============================================================
# EXPORT EVENT-LEVEL DATA
# ============================================================

loss_events.to_csv(
    OUTPUT_DIR / "lgd_loss_event_audit.csv",
    index=False
)

print("\n===== OUTPUT FILES =====")
print(OUTPUT_DIR / "lgd_loss_event_audit.csv")
print(OUTPUT_DIR / "lgd_by_zero_balance_code.csv")

print("\nLGD DATA AUDIT COMPLETED.")

