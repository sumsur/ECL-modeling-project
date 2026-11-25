from pathlib import Path
import shutil

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGES_DIR = BASE_DIR / "README_images"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def find_file(filename):
    matches = list(OUTPUTS_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} inside {OUTPUTS_DIR}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple files named {filename} found: {matches}"
        )

    return matches[0]


# ============================================================
# 2. COPY EXISTING PD MODEL CHARTS
# ============================================================

existing_charts = [
    "pd_roc_curve.png",
    "pd_precision_recall.png",
    "pd_calibration.png"
]

for filename in existing_charts:
    matches = list(OUTPUTS_DIR.rglob(filename))

    if matches:
        shutil.copy2(
            matches[0],
            IMAGES_DIR / filename
        )
        print(f"Copied: {filename}")
    else:
        print(f"WARNING: {filename} not found")


# ============================================================
# 3. LIFETIME PD PROJECTION
# ============================================================

lifetime_file = find_file(
    "lifetime_pd_annual_curve_202109.csv"
)

df = pd.read_csv(lifetime_file)

required_columns = {
    "id_loan",
    "ead",
    "year",
    "cumulative_pd"
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required_columns - set(df.columns)}"
    )

df["year"] = pd.to_numeric(df["year"])
df["ead"] = pd.to_numeric(df["ead"])
df["cumulative_pd"] = pd.to_numeric(df["cumulative_pd"])

# One EAD value per loan
ead_by_loan = df.groupby("id_loan")["ead"].first()

# Loan-level cumulative PD by year
pd_matrix = df.pivot(
    index="id_loan",
    columns="year",
    values="cumulative_pd"
)

# Retain the final cumulative PD after a loan's
# remaining contractual maturity has been reached
max_year = int(df["year"].max())

pd_matrix = pd_matrix.reindex(
    columns=range(1, max_year + 1)
)

pd_matrix = pd_matrix.sort_index(axis=1).ffill(axis=1)

# EAD-weighted cumulative PD using initial portfolio EAD
weighted_pd = (
    pd_matrix.mul(ead_by_loan, axis=0).sum(axis=0)
    / ead_by_loan.sum()
)

lifetime_chart = pd.DataFrame({
    "Year": weighted_pd.index,
    "Cumulative_PD": weighted_pd.values
})

plt.figure(figsize=(10, 6))

plt.plot(
    lifetime_chart["Year"],
    lifetime_chart["Cumulative_PD"],
    marker="o",
    linewidth=2
)

plt.title(
    "EAD-Weighted Cumulative Lifetime PD",
    fontsize=14
)

plt.xlabel("Projection Year")
plt.ylabel("Cumulative Probability of Default")

plt.gca().yaxis.set_major_formatter(
    PercentFormatter(1)
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "lifetime_pd_projection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Created: lifetime_pd_projection.png")


# ============================================================
# 4. ECL SCENARIO COMPARISON
# ============================================================

scenario_file = find_file(
    "ecl_scenario_summary_202109.csv"
)

ecl = pd.read_csv(scenario_file)

scenario_order = [
    "Optimistic",
    "Base",
    "Adverse"
]

ecl["Scenario"] = pd.Categorical(
    ecl["Scenario"],
    categories=scenario_order,
    ordered=True
)

ecl = ecl.sort_values("Scenario")

ecl["ECL_12m_m"] = ecl["ECL_12m"] / 1_000_000
ecl["ECL_lifetime_m"] = ecl["ECL_lifetime"] / 1_000_000

ax = ecl.set_index("Scenario")[
    ["ECL_12m_m", "ECL_lifetime_m"]
].plot(
    kind="bar",
    figsize=(10, 6),
    width=0.75
)

ax.set_title(
    "ECL Sensitivity Analysis by LGD Scenario",
    fontsize=14
)

ax.set_xlabel("LGD Scenario")
ax.set_ylabel("Expected Credit Loss (Millions)")

ax.legend([
    "12-Month ECL",
    "Lifetime ECL"
])

ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "ecl_scenario_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Created: ecl_scenario_analysis.png")


# ============================================================
# 5. ANNUAL ECL PROFILE
# ============================================================

annual_file = find_file(
    "ecl_annual_breakdown_202109.csv"
)

annual = pd.read_csv(annual_file)

annual = annual.sort_values("year")

scenario_columns = {
    "ecl_optimistic": "Optimistic",
    "ecl_base": "Base",
    "ecl_adverse": "Adverse"
}

plt.figure(figsize=(11, 6))

for column, label in scenario_columns.items():

    if column not in annual.columns:
        raise ValueError(f"Missing column: {column}")

    plt.plot(
        annual["year"],
        annual[column] / 1_000_000,
        marker="o",
        linewidth=2,
        label=label
    )

plt.title(
    "Annual Expected Credit Loss Projection",
    fontsize=14
)

plt.xlabel("Projection Year")
plt.ylabel("Annual ECL (Millions)")

plt.legend(title="LGD Scenario")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    IMAGES_DIR / "annual_ecl_profile.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Created: annual_ecl_profile.png")


# ============================================================
# 6. FINAL CHECK
# ============================================================

print("\n===== README CHARTS =====")

for file in sorted(IMAGES_DIR.glob("*.png")):
    print(f"{file.name} | {file.stat().st_size:,} bytes")

print("\nAll available charts saved successfully.")

