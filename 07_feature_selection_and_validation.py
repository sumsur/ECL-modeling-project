import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND DATA LOADING
# ============================================================

DATA_PATH = Path("C:\Python\ECL modeling\data_raw.csv")

OUTPUT_DIR = Path("outputs/feature_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("FEATURE SELECTION AND DATA QUALITY VALIDATION")
print("=" * 70)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DATA PREPARATION
# ============================================================

df["Status"] = pd.to_numeric(df["Status"], errors="coerce")
df = df[df["Status"].isin([0, 1])].copy()

df["loan_amount"] = pd.to_numeric(
    df["loan_amount"], errors="coerce"
)

# Primary analytical sample excludes EQUI
df_model = df[
    df["credit_type"].astype(str).str.upper() != "EQUI"
].copy()

print("\nFull portfolio:", len(df))
print("Portfolio without EQUI:", len(df_model))

print(
    "Default rate without EQUI:",
    round(df_model["Status"].mean() * 100, 2), "%"
)


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

missing_summary = pd.DataFrame({
    "Missing_Count": df_model.isna().sum(),
    "Missing_Percentage": df_model.isna().mean() * 100
})

missing_summary = missing_summary.sort_values(
    "Missing_Percentage",
    ascending=False
)

missing_summary.to_csv(
    OUTPUT_DIR / "missing_value_summary.csv"
)

print("\nMISSING VALUE SUMMARY")
print(missing_summary.round(2).to_string())


# ============================================================
# 4. MISSINGNESS BY DEFAULT STATUS
# ============================================================

missing_by_status = []

variables_for_missing = [
    "Credit_Score",
    "income",
    "property_value",
    "LTV",
    "dtir1",
    "loan_amount"
]

for variable in variables_for_missing:

    if variable not in df_model.columns:
        continue

    for status in [0, 1]:

        subset = df_model[df_model["Status"] == status]

        missing_count = subset[variable].isna().sum()

        missing_percentage = (
            subset[variable].isna().mean() * 100
        )

        missing_by_status.append({
            "Variable": variable,
            "Status": status,
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_percentage
        })

missing_status_df = pd.DataFrame(missing_by_status)

missing_status_df.to_csv(
    OUTPUT_DIR / "missingness_by_default_status.csv",
    index=False
)

print("\nMISSINGNESS BY STATUS")
print(missing_status_df.round(2).to_string(index=False))


# ============================================================
# 5. EQUI AND MISSINGNESS INVESTIGATION
# ============================================================

equI_analysis = []

for category in df["credit_type"].dropna().unique():

    subset = df[df["credit_type"] == category]

    equI_analysis.append({
        "Credit_Type": category,
        "Number_of_Loans": len(subset),
        "Defaults": int(subset["Status"].sum()),
        "Default_Rate": subset["Status"].mean() * 100,
        "Missing_DTIR": subset["dtir1"].isna().sum(),
        "Missing_DTIR_Percentage": (
            subset["dtir1"].isna().mean() * 100
        ),
        "Missing_LTV": subset["LTV"].isna().sum(),
        "Missing_LTV_Percentage": (
            subset["LTV"].isna().mean() * 100
        )
    })

equI_df = pd.DataFrame(equI_analysis)

equI_df.to_csv(
    OUTPUT_DIR / "credit_type_quality_analysis.csv",
    index=False
)

print("\nCREDIT TYPE INVESTIGATION")
print(equI_df.round(2).to_string(index=False))


# ============================================================
# 6. NUMERICAL OUTLIER ANALYSIS
# ============================================================

numerical_variables = [
    "Credit_Score",
    "income",
    "loan_amount",
    "property_value",
    "LTV",
    "dtir1"
]

outlier_results = []

for variable in numerical_variables:

    if variable not in df_model.columns:
        continue

    values = pd.to_numeric(
        df_model[variable],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = values[
        (values < lower_bound) |
        (values > upper_bound)
    ]

    outlier_results.append({
        "Variable": variable,
        "Count": len(values),
        "Minimum": values.min(),
        "Q1": q1,
        "Median": values.median(),
        "Q3": q3,
        "Maximum": values.max(),
        "Lower_IQR_Bound": lower_bound,
        "Upper_IQR_Bound": upper_bound,
        "Outlier_Count": len(outliers),
        "Outlier_Percentage": (
            len(outliers) / len(values) * 100
        )
    })

outlier_df = pd.DataFrame(outlier_results)

outlier_df.to_csv(
    OUTPUT_DIR / "numerical_outlier_analysis.csv",
    index=False
)

print("\nOUTLIER ANALYSIS")
print(outlier_df.round(2).to_string(index=False))


# ============================================================
# 7. STATISTICAL TESTS - NUMERICAL VARIABLES
# ============================================================

numerical_test_results = []

for variable in numerical_variables:

    if variable not in df_model.columns:
        continue

    temp = df_model[
        ["Status", variable]
    ].copy()

    temp[variable] = pd.to_numeric(
        temp[variable],
        errors="coerce"
    )

    non_default = temp[
        (temp["Status"] == 0) &
        temp[variable].notna()
    ][variable]

    default = temp[
        (temp["Status"] == 1) &
        temp[variable].notna()
    ][variable]

    if len(non_default) == 0 or len(default) == 0:
        continue

    statistic, p_value = mannwhitneyu(
        non_default,
        default,
        alternative="two-sided"
    )

    numerical_test_results.append({
        "Variable": variable,
        "Non_Default_Median": non_default.median(),
        "Default_Median": default.median(),
        "Non_Default_Count": len(non_default),
        "Default_Count": len(default),
        "Mann_Whitney_U": statistic,
        "P_Value": p_value,
        "Statistically_Significant": p_value < 0.05
    })

numerical_tests_df = pd.DataFrame(
    numerical_test_results
)

numerical_tests_df.to_csv(
    OUTPUT_DIR / "numerical_statistical_tests.csv",
    index=False
)

print("\nNUMERICAL STATISTICAL TESTS")
print(numerical_tests_df.round(5).to_string(index=False))


# ============================================================
# 8. CHI-SQUARE TESTS - CATEGORICAL VARIABLES
# ============================================================

categorical_variables = [
    "loan_type",
    "loan_purpose",
    "Region",
    "Gender",
    "occupancy_type",
    "Neg_ammortization",
    "interest_only",
    "age"
]

categorical_test_results = []

for variable in categorical_variables:

    if variable not in df_model.columns:
        continue

    temp = df_model[
        [variable, "Status"]
    ].copy()

    temp[variable] = temp[variable].fillna("Missing")

    contingency_table = pd.crosstab(
        temp[variable],
        temp["Status"]
    )

    if contingency_table.shape[0] < 2:
        continue

    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table
    )

    n = contingency_table.to_numpy().sum()

    min_dimension = min(
        contingency_table.shape[0] - 1,
        contingency_table.shape[1] - 1
    )

    cramers_v = np.sqrt(
        chi2 / (n * min_dimension)
    ) if min_dimension > 0 else np.nan

    categorical_test_results.append({
        "Variable": variable,
        "Chi2_Statistic": chi2,
        "Degrees_of_Freedom": dof,
        "P_Value": p_value,
        "Cramers_V": cramers_v,
        "Statistically_Significant": p_value < 0.05
    })

categorical_tests_df = pd.DataFrame(
    categorical_test_results
)

categorical_tests_df.to_csv(
    OUTPUT_DIR / "categorical_statistical_tests.csv",
    index=False
)

print("\nCATEGORICAL STATISTICAL TESTS")
print(categorical_tests_df.round(5).to_string(index=False))


# ============================================================
# 9. CORRELATION ANALYSIS
# ============================================================

correlation_variables = [
    "Credit_Score",
    "income",
    "loan_amount",
    "property_value",
    "LTV",
    "dtir1"
]

available_variables = [
    col for col in correlation_variables
    if col in df_model.columns
]

correlation_data = df_model[
    available_variables
].apply(pd.to_numeric, errors="coerce")

correlation_matrix = correlation_data.corr(
    method="spearman"
)

correlation_matrix.to_csv(
    OUTPUT_DIR / "spearman_correlation_matrix.csv"
)

plt.figure(figsize=(10, 8))

plt.imshow(
    correlation_matrix,
    aspect="auto",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Spearman correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.index)),
    correlation_matrix.index
)

plt.title("Spearman Correlation Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "spearman_correlation_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. MUTUAL INFORMATION
# ============================================================

mi_variables = [
    "Credit_Score",
    "income",
    "loan_amount",
    "property_value",
    "LTV",
    "dtir1"
]

mi_data = df_model[
    mi_variables
].copy()

mi_data = mi_data.apply(
    pd.to_numeric,
    errors="coerce"
)

# Add missingness indicators
for col in mi_variables:
    mi_data[col + "_missing"] = (
        mi_data[col].isna().astype(int)
    )

# Median imputation for this exploratory calculation only
mi_data = mi_data.fillna(
    mi_data.median()
)

mi_scores = mutual_info_classif(
    mi_data,
    df_model["Status"].astype(int),
    random_state=42
)

mi_results = pd.DataFrame({
    "Variable": mi_data.columns,
    "Mutual_Information": mi_scores
})

mi_results = mi_results.sort_values(
    "Mutual_Information",
    ascending=False
)

mi_results.to_csv(
    OUTPUT_DIR / "mutual_information_results.csv",
    index=False
)

print("\nMUTUAL INFORMATION")
print(mi_results.round(5).to_string(index=False))


# ============================================================
# 11. MISSINGNESS VISUALIZATION
# ============================================================

missing_plot = (
    missing_status_df
    .pivot(
        index="Variable",
        columns="Status",
        values="Missing_Percentage"
    )
)

missing_plot.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Missing Values by Default Status")
plt.xlabel("Variable")
plt.ylabel("Missing Values (%)")
plt.xticks(rotation=45, ha="right")
plt.legend(["Non-Default", "Default"])
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "missingness_by_status.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION COMPLETED")
print("=" * 70)

print(f"Results saved in: {OUTPUT_DIR.resolve()}")

print("\nGenerated files:")
print("- missing_value_summary.csv")
print("- missingness_by_default_status.csv")
print("- credit_type_quality_analysis.csv")
print("- numerical_outlier_analysis.csv")
print("- numerical_statistical_tests.csv")
print("- categorical_statistical_tests.csv")
print("- spearman_correlation_matrix.csv")
print("- mutual_information_results.csv")
print("- spearman_correlation_matrix.png")
print("- missingness_by_status.png")

print("\nNext step: Review feature quality before PD modeling.")
