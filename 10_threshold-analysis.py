
# ============================================================
# 10 - CLASSIFICATION THRESHOLD ANALYSIS
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "logistic_regression"
    / "test_predictions.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "logistic_regression"

# ------------------------------------------------------------
# 2. LOAD PREDICTIONS
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

y_true = df["Actual_Status"]
y_proba = df["Predicted_PD"]

# ------------------------------------------------------------
# 3. TEST DIFFERENT THRESHOLDS
# ------------------------------------------------------------

thresholds = np.arange(0.05, 0.51, 0.05)

results = []

for threshold in thresholds:

    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": round(threshold, 2),
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1_Score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "Predicted_Default_Rate": y_pred.mean()
    })

threshold_results = pd.DataFrame(results)

# ------------------------------------------------------------
# 4. DISPLAY RESULTS
# ------------------------------------------------------------

print("=" * 65)
print("CLASSIFICATION THRESHOLD ANALYSIS")
print("=" * 65)

print(threshold_results.to_string(index=False))

# ------------------------------------------------------------
# 5. SAVE RESULTS
# ------------------------------------------------------------

threshold_results.to_csv(
    OUTPUT_DIR / "threshold_analysis.csv",
    index=False
)

print("\nResults saved to:")
print(OUTPUT_DIR / "threshold_analysis.csv")

