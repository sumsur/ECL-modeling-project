# ============================================================
# 12 - PD MODEL COMPARISON
# Four Logistic Regression Specifications
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_DIR = BASE_DIR / "outputs" / "data_preparation"

OUTPUT_DIR = BASE_DIR / "outputs" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Status"


# ------------------------------------------------------------
# 2. LOAD TRAIN AND TEST DATA
# ------------------------------------------------------------

train = pd.read_csv(INPUT_DIR / "train_woe.csv")
test = pd.read_csv(INPUT_DIR / "test_woe.csv")

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET].astype(int)

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET].astype(int)

# Ensure identical feature columns and order

X_test = X_test[X_train.columns]

print("=" * 70)
print("PD MODEL - SPECIFICATION COMPARISON")
print("=" * 70)

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))


# ------------------------------------------------------------
# 3. DEFINE FOUR MODEL SPECIFICATIONS
# ------------------------------------------------------------

model_specs = {
    "Model_1_Full": list(X_train.columns),

    "Model_2_Without_loan_type": [
        col for col in X_train.columns
        if col != "loan_type"
    ],

    "Model_3_Without_business": [
        col for col in X_train.columns
        if col != "business_or_commercial"
    ],

    "Model_4_Without_both": [
        col for col in X_train.columns
        if col not in [
            "loan_type",
            "business_or_commercial"
        ]
    ]
}


# ------------------------------------------------------------
# 4. METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(y_true, probabilities):

    auc = roc_auc_score(y_true, probabilities)

    # KS statistic: maximum separation between
    # default and non-default predicted PD distributions

    default_scores = probabilities[
        np.asarray(y_true) == 1
    ]

    non_default_scores = probabilities[
        np.asarray(y_true) == 0
    ]

    ks = ks_2samp(
        default_scores,
        non_default_scores
    ).statistic

    return {
        "ROC_AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks,
        "Brier_Score": brier_score_loss(
            y_true,
            probabilities
        )
    }


# ------------------------------------------------------------
# 5. TRAIN AND EVALUATE ALL MODELS
# ------------------------------------------------------------

performance_results = []
coefficient_results = []
calibration_results = []

prediction_data = pd.DataFrame({
    "Actual_Status": y_test
})

calibration_plot_data = {}

for model_name, selected_features in model_specs.items():

    print("\n" + "-" * 70)
    print("Training:", model_name)
    print("Number of features:", len(selected_features))

    Xtr = X_train[selected_features]
    Xte = X_test[selected_features]

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    model.fit(Xtr, y_train)

    train_pd = model.predict_proba(Xtr)[:, 1]
    test_pd = model.predict_proba(Xte)[:, 1]

    # Train metrics
    train_metrics = calculate_metrics(
        y_train,
        train_pd
    )

    # Test metrics
    test_metrics = calculate_metrics(
        y_test,
        test_pd
    )

    performance_results.append({
        "Model": model_name,
        "Number_of_Features": len(selected_features),

        "Train_ROC_AUC": train_metrics["ROC_AUC"],
        "Test_ROC_AUC": test_metrics["ROC_AUC"],

        "Train_Gini": train_metrics["Gini"],
        "Test_Gini": test_metrics["Gini"],

        "Train_KS": train_metrics["KS"],
        "Test_KS": test_metrics["KS"],

        "Train_Brier": train_metrics["Brier_Score"],
        "Test_Brier": test_metrics["Brier_Score"],

        "AUC_Gap": (
            train_metrics["ROC_AUC"] -
            test_metrics["ROC_AUC"]
        )
    })

    # Save coefficients
    for feature, coefficient in zip(
        selected_features,
        model.coef_[0]
    ):

        coefficient_results.append({
            "Model": model_name,
            "Feature": feature,
            "Coefficient": coefficient,
            "Odds_Ratio": np.exp(coefficient)
        })

    # Save test predictions
    prediction_data[model_name] = test_pd

    # Calibration curve
    observed, predicted = calibration_curve(
        y_test,
        test_pd,
        n_bins=10,
        strategy="quantile"
    )

    calibration_plot_data[model_name] = (
        predicted,
        observed
    )

    for bin_number, (pred, obs) in enumerate(
        zip(predicted, observed), start=1
    ):

        calibration_results.append({
            "Model": model_name,
            "Bin": bin_number,
            "Mean_Predicted_PD": pred,
            "Observed_Default_Rate": obs
        })

    print(
        f"Test AUC: {test_metrics['ROC_AUC']:.4f} | "
        f"Test Gini: {test_metrics['Gini']:.4f} | "
        f"Test KS: {test_metrics['KS']:.4f} | "
        f"Test Brier: {test_metrics['Brier_Score']:.4f}"
    )


# ------------------------------------------------------------
# 6. SAVE PERFORMANCE COMPARISON
# ------------------------------------------------------------

performance_df = pd.DataFrame(performance_results)

performance_df = performance_df.sort_values(
    "Test_ROC_AUC",
    ascending=False
)

performance_df.to_csv(
    OUTPUT_DIR / "model_performance_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 7. SAVE COEFFICIENTS
# ------------------------------------------------------------

coefficients_df = pd.DataFrame(coefficient_results)

coefficients_df.to_csv(
    OUTPUT_DIR / "model_coefficients_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 8. SAVE CALIBRATION RESULTS
# ------------------------------------------------------------

calibration_df = pd.DataFrame(calibration_results)

calibration_df.to_csv(
    OUTPUT_DIR / "calibration_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 9. SAVE TEST PREDICTIONS
# ------------------------------------------------------------

prediction_data.to_csv(
    OUTPUT_DIR / "test_predictions_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 10. PLOT CALIBRATION CURVES
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

for model_name, (predicted, observed) in (
    calibration_plot_data.items()
):

    plt.plot(
        predicted,
        observed,
        marker="o",
        label=model_name
    )

plt.xlabel("Mean Predicted PD")
plt.ylabel("Observed Default Rate")

plt.title("Calibration Comparison - PD Models")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "calibration_comparison.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 11. FINAL RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    performance_df.round(4).to_string(index=False)
)

print("\n" + "=" * 70)
print("COEFFICIENT SIGN COMPARISON")
print("=" * 70)

sign_table = coefficients_df.pivot(
    index="Feature",
    columns="Model",
    values="Coefficient"
)

print(sign_table.round(4).to_string())

print("\n" + "=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)

print("Results saved to:")
print(OUTPUT_DIR)