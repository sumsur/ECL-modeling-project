# 13 - FINAL PD MODEL
# Finalization of the baseline mortgage default prediction model
# ============================================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import ks_2samp


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "outputs" / "data_preparation"
OUTPUT_DIR = BASE_DIR / "outputs" / "final_model"
MODEL_DIR = OUTPUT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Status"
RANDOM_STATE = 42
N_CALIBRATION_BINS = 10

# Same LogisticRegression specification as Script 12
LOGISTIC_REGRESSION_SETTINGS = {
    "max_iter": 2000,
    "solver": "lbfgs"
}


# ------------------------------------------------------------
# 2. LOAD TRAIN AND TEST DATA
# ------------------------------------------------------------

train_path = INPUT_DIR / "train_woe.csv"
test_path = INPUT_DIR / "test_woe.csv"

if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
        "Nisu pronađeni train_woe.csv i/ili test_woe.csv. "
        "Prvo pokreni skriptu 08_data_preparation.py."
    )

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

if TARGET not in train.columns or TARGET not in test.columns:
    raise ValueError(
        f"Target kolona '{TARGET}' nije pronađena u train/test podacima."
    )

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET].astype(int)

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET].astype(int)

# Ensure identical feature columns and order

missing_test_features = sorted(
    set(X_train.columns) - set(X_test.columns)
)

extra_test_features = sorted(
    set(X_test.columns) - set(X_train.columns)
)

if missing_test_features or extra_test_features:
    raise ValueError(
        "Train i test nemaju iste feature kolone. "
        f"Nedostaju u test: {missing_test_features}; "
        f"Dodatne u test: {extra_test_features}"
    )

X_test = X_test[X_train.columns]

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Target mora imati obe klase (0 i 1) u train i test uzorku."
    )

print("=" * 72)
print("FINAL PD MODEL - MODEL FITTING, VALIDATION AND EXPORT")
print("=" * 72)

print(f"Training observations: {len(X_train):,}")
print(f"Testing observations: {len(X_test):,}")
print(f"Training default rate: {y_train.mean():.4%}")
print(f"Testing default rate: {y_test.mean():.4%}")


# ------------------------------------------------------------
# 3. DEFINE FINAL SPECIFICATIONS
# ------------------------------------------------------------

# Model 1 is retained as the full benchmark.
# Model 2 is retained as the more parsimonious candidate because
# loan_type and business_or_commercial were found to be highly redundant.

model_specs = {
    "Model_1_Full": list(X_train.columns),

    "Model_2_Without_loan_type": [
        col for col in X_train.columns
        if col != "loan_type"
    ]
}

for model_name, features in model_specs.items():

    if not features:
        raise ValueError(
            f"{model_name} nema nijednu ulaznu varijablu."
        )


# ------------------------------------------------------------
# 4. METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(y_true, probabilities):

    auc = roc_auc_score(y_true, probabilities)

    probabilities = np.asarray(probabilities)
    y_array = np.asarray(y_true)

    default_scores = probabilities[
        y_array == 1
    ]

    non_default_scores = probabilities[
        y_array == 0
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
# 5. CALIBRATION TABLE FUNCTION
# ------------------------------------------------------------

def create_calibration_table(
    y_true,
    probabilities,
    model_name,
    n_bins=10
):

    """
    Groups test observations into quantile bins based on predicted PD.
    The table includes counts so that each observed default rate can
    be interpreted together with its sample size.
    """

    calibration_data = pd.DataFrame({
        "Actual_Status": np.asarray(y_true),
        "Predicted_PD": np.asarray(probabilities)
    })

    calibration_data["PD_Decile"] = pd.qcut(
        calibration_data["Predicted_PD"],
        q=n_bins,
        duplicates="drop"
    )

    result = (
        calibration_data
        .groupby("PD_Decile", observed=True)
        .agg(
            Number_of_Observations=("Actual_Status", "size"),
            Number_of_Defaults=("Actual_Status", "sum"),
            Mean_Predicted_PD=("Predicted_PD", "mean"),
            Observed_Default_Rate=("Actual_Status", "mean"),
            Min_Predicted_PD=("Predicted_PD", "min"),
            Max_Predicted_PD=("Predicted_PD", "max")
        )
        .reset_index()
    )

    result.insert(0, "Model", model_name)

    result["Number_of_Non_Defaults"] = (
        result["Number_of_Observations"]
        - result["Number_of_Defaults"]
    )

    return result


# ------------------------------------------------------------
# 6. TRAIN, EVALUATE AND SAVE BOTH FINAL MODELS
# ------------------------------------------------------------

performance_results = []
coefficient_results = []
calibration_tables = []

prediction_data = pd.DataFrame({
    "Actual_Status": y_test.reset_index(drop=True)
})

calibration_plot_data = {}

for model_name, selected_features in model_specs.items():

    print("\n" + "-" * 72)
    print(f"Training: {model_name}")
    print(f"Number of features: {len(selected_features)}")

    Xtr = X_train[selected_features]
    Xte = X_test[selected_features]

    model = LogisticRegression(
        **LOGISTIC_REGRESSION_SETTINGS
    )

    model.fit(Xtr, y_train)

    train_pd = model.predict_proba(Xtr)[:, 1]
    test_pd = model.predict_proba(Xte)[:, 1]

    train_metrics = calculate_metrics(
        y_train,
        train_pd
    )

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
            train_metrics["ROC_AUC"]
            - test_metrics["ROC_AUC"]
        )
    })

    # Save coefficients and odds ratios

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

    prediction_data[f"{model_name}_PD"] = test_pd

    # Calibration table by test-set PD quantiles

    calibration_table = create_calibration_table(
        y_test,
        test_pd,
        model_name,
        n_bins=N_CALIBRATION_BINS
    )

    calibration_tables.append(calibration_table)

    calibration_plot_data[model_name] = (
        calibration_table["Mean_Predicted_PD"].to_numpy(),
        calibration_table["Observed_Default_Rate"].to_numpy()
    )

    # Save fitted model together with required feature order.
    # Input data must already be transformed using the WOE mappings
    # from Script 08.

    artifact = {
        "model": model,
        "model_name": model_name,
        "features": selected_features,
        "target": TARGET,
        "input_type": "WOE-transformed features",
        "logistic_regression_settings": LOGISTIC_REGRESSION_SETTINGS
    }

    joblib.dump(
        artifact,
        MODEL_DIR / f"{model_name}.joblib"
    )

    print(
        f"Test AUC: {test_metrics['ROC_AUC']:.4f} | "
        f"Test Gini: {test_metrics['Gini']:.4f} | "
        f"Test KS: {test_metrics['KS']:.4f} | "
        f"Test Brier: {test_metrics['Brier_Score']:.4f}"
    )


# ------------------------------------------------------------
# 7. SAVE RESULTS
# ------------------------------------------------------------

performance_df = pd.DataFrame(
    performance_results
).sort_values(
    "Test_ROC_AUC",
    ascending=False
)

coefficients_df = pd.DataFrame(
    coefficient_results
)

calibration_df = pd.concat(
    calibration_tables,
    ignore_index=True
)

performance_df.to_csv(
    OUTPUT_DIR / "final_model_performance.csv",
    index=False
)

coefficients_df.to_csv(
    OUTPUT_DIR / "final_model_coefficients.csv",
    index=False
)

calibration_df.to_csv(
    OUTPUT_DIR / "final_model_calibration_by_decile.csv",
    index=False
)

prediction_data.to_csv(
    OUTPUT_DIR / "final_test_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 8. CALIBRATION PLOT
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

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

plt.title(
    "Final PD Model - Calibration on Test Sample"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_model_calibration.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 9. SAVE MODEL METADATA
# ------------------------------------------------------------

metadata = {
    "project": "Baseline Mortgage Default Prediction Model",
    "target": TARGET,

    "target_interpretation": (
        "Status = 1: default; Status = 0: non-default"
    ),

    "training_observations": int(len(X_train)),
    "testing_observations": int(len(X_test)),

    "training_default_rate": float(y_train.mean()),
    "testing_default_rate": float(y_test.mean()),

    "model_specifications": {
        name: features
        for name, features in model_specs.items()
    },

    "logistic_regression_settings": (
        LOGISTIC_REGRESSION_SETTINGS
    ),

    "input_requirement": (
        "Prediction inputs must be transformed with the same WOE "
        "mappings generated by 08_data_preparation.py."
    ),

    "scope_limitation": (
        "This is a baseline default prediction model. It is not a complete "
        "IFRS 9 ECL model and does not independently estimate lifetime PD, "
        "LGD or EAD."
    )
}

with open(
    OUTPUT_DIR / "final_model_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )


# ------------------------------------------------------------
# 10. WRITE SHORT FINAL REPORT
# ------------------------------------------------------------

best_row = performance_df.iloc[0]

report_lines = [
    "FINAL PD MODEL - SUMMARY",
    "=" * 60,
    "",

    f"Training observations: {len(X_train):,}",
    f"Testing observations: {len(X_test):,}",

    f"Training default rate: {y_train.mean():.4%}",
    f"Testing default rate: {y_test.mean():.4%}",

    "",
    "MODEL PERFORMANCE",
    "-" * 60,

    performance_df.round(4).to_string(index=False),

    "",
    "INTERPRETATION",
    "-" * 60,

    f"Highest test ROC-AUC: {best_row['Model']} "
    f"({best_row['Test_ROC_AUC']:.4f}).",

    "The full model is retained as the benchmark.",

    "The specification without loan_type is retained as a more "
    "parsimonious candidate because loan_type and "
    "business_or_commercial were highly redundant in the earlier "
    "diagnostic analysis.",

    "",
    "IMPORTANT LIMITATIONS",
    "-" * 60,

    "1. This is a baseline default prediction model, not a complete IFRS 9 ECL model.",

    "2. The model does not independently estimate lifetime PD, LGD or EAD.",

    "3. The saved model expects WOE-transformed input features.",

    "4. Calibration results are diagnostics on the held-out test sample; "
    "no recalibration has been fitted on the test sample.",

    "5. The model comparison does not establish that small metric differences "
    "are statistically significant.",

    "",
    "Saved model artifacts are located in the 'models' subfolder."
]

with open(
    OUTPUT_DIR / "final_model_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


# ------------------------------------------------------------
# 11. FINAL CONSOLE OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 72)
print("FINAL MODEL PERFORMANCE")
print("=" * 72)

print(
    performance_df.round(4).to_string(index=False)
)

print("\n" + "=" * 72)
print("CALIBRATION BY PD DECILE")
print("=" * 72)

print(
    calibration_df[
        [
            "Model",
            "PD_Decile",
            "Number_of_Observations",
            "Number_of_Defaults",
            "Mean_Predicted_PD",
            "Observed_Default_Rate"
        ]
    ].round(4).to_string(index=False)
)

print("\n" + "=" * 72)
print("FINALIZATION COMPLETED")
print("=" * 72)

print(f"Outputs saved to: {OUTPUT_DIR}")
print(f"Model artifacts saved to: {MODEL_DIR}")
