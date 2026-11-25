# ============================================================
# 09 - LOGISTIC REGRESSION PD MODEL
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_DIR = BASE_DIR / "outputs" / "data_preparation"
OUTPUT_DIR = BASE_DIR / "outputs" / "logistic_regression"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

train = pd.read_csv(INPUT_DIR / "train_woe.csv")
test = pd.read_csv(INPUT_DIR / "test_woe.csv")

TARGET = "Status"

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]

print("=" * 65)
print("LOGISTIC REGRESSION - PD MODEL")
print("=" * 65)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ------------------------------------------------------------
# 3. TRAIN LOGISTIC REGRESSION
# ------------------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs"
)

model.fit(X_train, y_train)


# ------------------------------------------------------------
# 4. PREDICT PROBABILITIES
# ------------------------------------------------------------

y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

y_test_pred = model.predict(X_test)


# ------------------------------------------------------------
# 5. MODEL PERFORMANCE
# ------------------------------------------------------------

train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

train_gini = 2 * train_auc - 1
test_gini = 2 * test_auc - 1

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

ks_statistic = np.max(tpr - fpr)

brier = brier_score_loss(y_test, y_test_proba)

print("\nMODEL PERFORMANCE")
print("-" * 65)

print(f"Train ROC-AUC: {train_auc:.4f}")
print(f"Test ROC-AUC: {test_auc:.4f}")

print(f"Train Gini: {train_gini:.4f}")
print(f"Test Gini: {test_gini:.4f}")

print(f"Test KS: {ks_statistic:.4f}")
print(f"Test Brier: {brier:.4f}")


# ------------------------------------------------------------
# 6. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_test_pred)

print("\nCONFUSION MATRIX")
print("-" * 65)
print(cm)

print("\nCLASSIFICATION REPORT")
print("-" * 65)
print(classification_report(y_test, y_test_pred))


# ------------------------------------------------------------
# 7. ROC CURVE
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PD Model")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 8. CALIBRATION CURVE
# ------------------------------------------------------------

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test,
    y_test_proba,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(8, 6))

plt.plot(
    mean_predicted_value,
    fraction_of_positives,
    marker="o",
    label="Logistic Regression"
)

plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

plt.xlabel("Mean Predicted PD")
plt.ylabel("Observed Default Rate")
plt.title("Calibration Curve - PD Model")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 9. MODEL COEFFICIENTS
# ------------------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})

coefficients["Odds_Ratio"] = np.exp(coefficients["Coefficient"])

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\nMODEL COEFFICIENTS")
print("-" * 65)
print(coefficients.to_string(index=False))

coefficients.to_csv(
    OUTPUT_DIR / "model_coefficients.csv",
    index=False
)


# ------------------------------------------------------------
# 10. SAVE TEST PREDICTIONS
# ------------------------------------------------------------

predictions = pd.DataFrame({
    "Actual_Status": y_test,
    "Predicted_PD": y_test_proba,
    "Predicted_Status": y_test_pred
})

predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

print("\nPredictions saved successfully")
print(OUTPUT_DIR / "trst_predictions.csv")

# ------------------------------------------------------------
# 11. SAVE MODEL PERFORMANCE
# ------------------------------------------------------------

performance = pd.DataFrame({
    "Metric": [
        "Train ROC-AUC",
        "Test ROC-AUC",
        "Train Gini",
        "Test Gini",
        "Test KS",
        "Test Brier Score"
    ],
    "Value": [
        train_auc,
        test_auc,
        train_gini,
        test_gini,
        ks_statistic,
        brier
    ]
})

performance.to_csv(
    OUTPUT_DIR / "model_performance.csv",
    index=False
)


print("\nFILES SAVED TO:")
print(OUTPUT_DIR)

print("\nLOGISTIC REGRESSION MODEL COMPLETED")

