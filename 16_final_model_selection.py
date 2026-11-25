from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    roc_curve
)


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "outputs" / "data_preparation"
OUTPUT_DIR = BASE_DIR / "outputs" / "final_PD_model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Status"

BASE_FEATURES = [
    "dtir1",
    "LTV",
    "income",
    "property_value",
    "loan_amount",
    "Credit_Score",
    "Neg_ammortization",
    "loan_type",
    "loan_purpose",
    "Credit_Worthiness",
    "business_or_commercial",
    "loan_limit",
    "age",
    "submission_of_application"
]

REGION_FEATURES = BASE_FEATURES + ["Region"]


# ============================================================
# 2. LOAD DATA
# ============================================================

train = pd.read_csv(DATA_DIR / "train_woe.csv")
test = pd.read_csv(DATA_DIR / "test_woe.csv")

y_train = train[TARGET]
y_test = test[TARGET]


# ============================================================
# 3. METRICS FUNCTION
# ============================================================

def calculate_ks(y_true, probabilities):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities
    )

    return np.max(tpr - fpr)


def evaluate_model(name, features):

    X_train = train[features]
    X_test = test[features]

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    train_pd = model.predict_proba(X_train)[:, 1]
    test_pd = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_pd)
    test_auc = roc_auc_score(y_test, test_pd)

    metrics = {
        "Model": name,
        "Number_of_Features": len(features),
        "Train_AUC": train_auc,
        "Test_AUC": test_auc,
        "Test_Gini": 2 * test_auc - 1,
        "Test_KS": calculate_ks(y_test, test_pd),
        "Test_Brier": brier_score_loss(y_test, test_pd),
        "AUC_Gap": train_auc - test_auc
    }

    return model, metrics, train_pd, test_pd


# ============================================================
# 4. COMPARE MODEL SPECIFICATIONS
# ============================================================

baseline_model, baseline_metrics, _, _ = evaluate_model(
    "Baseline - without Region",
    BASE_FEATURES
)

region_model, region_metrics, train_pd, test_pd = evaluate_model(
    "Final candidate - with Region",
    REGION_FEATURES
)

comparison = pd.DataFrame([
    baseline_metrics,
    region_metrics
])

print("\n" + "=" * 75)
print("FINAL PD MODEL - SPECIFICATION COMPARISON")
print("=" * 75)

print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_DIR / "final_model_comparison.csv",
    index=False
)


# ============================================================
# 5. FINAL MODEL SELECTION
# ============================================================

# Select the model with Region based on the previously observed
# improvement in Test AUC and Brier score.
# This is a provisional selection, not proof of statistical superiority.

final_model = region_model
final_features = REGION_FEATURES

print("\nSelected model: Logistic Regression with Region")
print("Number of features:", len(final_features))


# ============================================================
# 6. CALIBRATION ANALYSIS
# ============================================================

calibration_data = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted_PD": test_pd
})

calibration_data["Decile"] = pd.qcut(
    calibration_data["Predicted_PD"],
    q=10,
    labels=False,
    duplicates="drop"
) + 1

calibration = (
    calibration_data
    .groupby("Decile")
    .agg(
        Number_of_Observations=("Actual", "count"),
        Mean_Predicted_PD=("Predicted_PD", "mean"),
        Observed_Default_Rate=("Actual", "mean")
    )
    .reset_index()
)

print("\nCALIBRATION TABLE")
print(calibration.to_string(index=False))

calibration.to_csv(
    OUTPUT_DIR / "final_calibration.csv",
    index=False
)


# ============================================================
# 7. COEFFICIENTS AND ODDS RATIOS
# ============================================================

coefficients = pd.DataFrame({
    "Feature": final_features,
    "Coefficient": final_model.coef_[0]
})

coefficients["Odds_Ratio"] = np.exp(
    coefficients["Coefficient"]
)

coefficients = coefficients.sort_values(
    "Coefficient",
    ascending=False
)

print("\nFINAL MODEL COEFFICIENTS")
print(coefficients.to_string(index=False))

coefficients.to_csv(
    OUTPUT_DIR / "final_coefficients.csv",
    index=False
)


# ============================================================
# 8. SAVE TEST PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "Actual_Default": y_test.to_numpy(),
    "Predicted_PD": test_pd
})

predictions.to_csv(
    OUTPUT_DIR / "final_test_predictions.csv",
    index=False
)


# ============================================================
# 9. SAVE MODEL
# ============================================================

joblib.dump(
    final_model,
    OUTPUT_DIR / "final_PD_logistic_model.joblib"
)

metadata = {
    "model_type": "Logistic Regression",
    "selected_specification": "With Region",
    "features": final_features,
    "target": TARGET,
    "train_observations": int(len(train)),
    "test_observations": int(len(test)),
    "train_default_rate": float(y_train.mean()),
    "test_default_rate": float(y_test.mean()),
    "test_auc": float(region_metrics["Test_AUC"]),
    "test_gini": float(region_metrics["Test_Gini"]),
    "test_ks": float(region_metrics["Test_KS"]),
    "test_brier": float(region_metrics["Test_Brier"]),
    "auc_gap": float(region_metrics["AUC_Gap"]),
    "input_format": "WOE-transformed features"
}

with open(
    OUTPUT_DIR / "final_model_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)


# ============================================================
# 10. FINAL REPORT
# ============================================================

report = f"""
FINAL PD MODEL - PROJECT SUMMARY

Model:
Logistic Regression with Region

Observations:
Training: {len(train)}
Testing: {len(test)}

Default rates:
Training: {y_train.mean():.4%}
Testing: {y_test.mean():.4%}

Performance:
Train AUC: {region_metrics['Train_AUC']:.4f}
Test AUC: {region_metrics['Test_AUC']:.4f}
Test Gini: {region_metrics['Test_Gini']:.4f}
Test KS: {region_metrics['Test_KS']:.4f}
Test Brier: {region_metrics['Test_Brier']:.4f}
AUC Gap: {region_metrics['AUC_Gap']:.4f}

Features:
{', '.join(final_features)}

Important limitations:
- Dataset contains observations from 2019 only.
- Model estimates default probability for the available dataset.
- This is not a complete IFRS 9 lifetime PD model.
- No macroeconomic scenario adjustment is included.
- LGD and EAD are not included.
- Model validation is based on a random train/test split,
  not an out-of-time sample.
- Model input must be WOE-transformed.

"""

with open(
    OUTPUT_DIR / "final_model_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\n" + "=" * 75)
print("FINAL PD MODEL COMPLETED")
print("=" * 75)

print("\nAll outputs saved in:")
print(OUTPUT_DIR)

