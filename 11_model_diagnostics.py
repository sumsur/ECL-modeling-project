# ============================================================
# 11 - COMPREHENSIVE PD MODEL DIAGNOSTICS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_DIR = BASE_DIR / "outputs" / "data_preparation"
OUTPUT_DIR = BASE_DIR / "outputs" / "model_diagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Status"
BOOTSTRAP_ITERATIONS = 50
RANDOM_STATE = 42


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

train = pd.read_csv(INPUT_DIR / "train_woe.csv")

X = train.drop(columns=[TARGET])
y = train[TARGET]

features = X.columns.tolist()

print("=" * 70)
print("COMPREHENSIVE PD MODEL DIAGNOSTICS")
print("=" * 70)

print("Observations:", len(X))
print("Number of predictors:", len(features))


# ------------------------------------------------------------
# 3. FIT BASELINE LOGISTIC REGRESSION
# ------------------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs"
)

model.fit(X, y)

base_coefficients = pd.Series(
    model.coef_[0],
    index=features
)


# ------------------------------------------------------------
# 4. CORRELATION MATRIX
# ------------------------------------------------------------

correlation_matrix = X.corr()

correlation_matrix.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

# Extract highly correlated pairs

correlation_pairs = []

for i in range(len(features)):
    for j in range(i + 1, len(features)):

        corr = correlation_matrix.iloc[i, j]

        if abs(corr) >= 0.50:

            correlation_pairs.append({
                "Feature_1": features[i],
                "Feature_2": features[j],
                "Correlation": corr
            })

correlation_df = pd.DataFrame(
    correlation_pairs,
    columns=["Feature_1", "Feature_2", "Correlation"]
)

correlation_df.to_csv(
    OUTPUT_DIR / "high_correlations.csv",
    index=False
)


# ------------------------------------------------------------
# 5. VARIANCE INFLATION FACTOR (VIF)
# ------------------------------------------------------------

# VIF calculated from the inverse correlation matrix.
# No additional package is required.

corr_values = correlation_matrix.to_numpy()

vif_results = []

try:

    inverse_corr = np.linalg.inv(corr_values)

    vif_values = np.diag(inverse_corr)

    for feature, vif in zip(features, vif_values):

        vif_results.append({
            "Feature": feature,
            "VIF": vif
        })

except np.linalg.LinAlgError:

    print("\nWARNING: Correlation matrix is singular.")
    print("VIF could not be calculated using matrix inversion.")

    for feature in features:
        vif_results.append({
            "Feature": feature,
            "VIF": np.nan
        })

vif_df = pd.DataFrame(vif_results)

vif_df = vif_df.sort_values(
    "VIF",
    ascending=False
)

vif_df.to_csv(
    OUTPUT_DIR / "vif_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# 6. STANDARDIZED COEFFICIENTS
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=features,
    index=X.index
)

standardized_model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs"
)

standardized_model.fit(X_scaled, y)

standardized_coefficients = pd.Series(
    standardized_model.coef_[0],
    index=features
)

coefficient_analysis = pd.DataFrame({
    "Feature": features,
    "Original_Coefficient": base_coefficients.values,
    "Standardized_Coefficient": standardized_coefficients.values,
    "Original_Odds_Ratio": np.exp(base_coefficients.values),
    "Absolute_Standardized_Coefficient":
        np.abs(standardized_coefficients.values)
})

coefficient_analysis = coefficient_analysis.sort_values(
    "Absolute_Standardized_Coefficient",
    ascending=False
)

coefficient_analysis.to_csv(
    OUTPUT_DIR / "coefficient_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# 7. WOE SIGN ANALYSIS
# ------------------------------------------------------------

sign_analysis = pd.DataFrame({
    "Feature": features,
    "Coefficient": base_coefficients.values
})

# WOE = ln(non-default distribution / default distribution)
# Therefore, a negative coefficient is the expected direction
# when predicting default.

sign_analysis["Expected_Sign"] = "Negative"

sign_analysis["Observed_Sign"] = np.where(
    sign_analysis["Coefficient"] > 0,
    "Positive",
    "Negative"
)

sign_analysis["Sign_Consistent"] = (
    sign_analysis["Observed_Sign"] ==
    sign_analysis["Expected_Sign"]
)

sign_analysis.to_csv(
    OUTPUT_DIR / "woe_sign_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# 8. BOOTSTRAP COEFFICIENT STABILITY
# ------------------------------------------------------------

bootstrap_coefficients = []

rng = np.random.default_rng(RANDOM_STATE)

for iteration in range(BOOTSTRAP_ITERATIONS):

    indices = rng.choice(
        len(X),
        size=len(X),
        replace=True
    )

    X_boot = X.iloc[indices]
    y_boot = y.iloc[indices]

    bootstrap_model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    bootstrap_model.fit(X_boot, y_boot)

    bootstrap_coefficients.append(
        bootstrap_model.coef_[0]
    )

bootstrap_df = pd.DataFrame(
    bootstrap_coefficients,
    columns=features
)

bootstrap_summary = pd.DataFrame({
    "Feature": features,
    "Mean_Coefficient": bootstrap_df.mean().values,
    "Std_Coefficient": bootstrap_df.std().values,
    "Min_Coefficient": bootstrap_df.min().values,
    "Max_Coefficient": bootstrap_df.max().values,
    "Positive_Sign_Frequency":
        (bootstrap_df > 0).mean().values,
    "Negative_Sign_Frequency":
        (bootstrap_df < 0).mean().values
})

bootstrap_summary.to_csv(
    OUTPUT_DIR / "bootstrap_stability.csv",
    index=False
)


# ------------------------------------------------------------
# 9. DIAGNOSTIC SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HIGH CORRELATIONS")
print("=" * 70)

if correlation_df.empty:
    print("No predictor pairs with absolute correlation >= 0.50.")
else:
    print(correlation_df.to_string(index=False))


print("\n" + "=" * 70)
print("VIF ANALYSIS")
print("=" * 70)

print(vif_df.to_string(index=False))


print("\n" + "=" * 70)
print("STANDARDIZED COEFFICIENTS")
print("=" * 70)

print(coefficient_analysis.to_string(index=False))


print("\n" + "=" * 70)
print("WOE SIGN ANALYSIS")
print("=" * 70)

print(sign_analysis.to_string(index=False))


print("\n" + "=" * 70)
print("BOOTSTRAP COEFFICIENT STABILITY")
print("=" * 70)

print(bootstrap_summary.to_string(index=False))


# ------------------------------------------------------------
# 10. CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(12, 10))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(features)),
    features,
    rotation=90
)

plt.yticks(
    range(len(features)),
    features
)

plt.title("PD Model - Predictor Correlation Matrix")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 11. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETED")
print("=" * 70)

print("All results saved to:")
print(OUTPUT_DIR)

