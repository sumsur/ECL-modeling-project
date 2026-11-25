import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve
)


# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "pd_cleaning"
    / "pd_modeling_dataset_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "pd_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading PD modeling dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)


# =====================================================
# 2. FEATURES AND TARGET
# =====================================================

numeric_features = [
    "fico",
    "ltv",
    "cltv",
    "dti",
    "orig_upb",
    "orig_int_rt",
    "orig_loan_term",
    "cnt_borr",
    "cnt_units",
    "mi_pct"
]

categorical_features = [
    "flag_fthb",
    "occpy_sts",
    "channel",
    "st",
    "prop_type",
    "loan_purpose"
]

features = numeric_features + categorical_features

target = "target_12m"

required_columns = (
    features + [target, "id_loan", "period"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# =====================================================
# 3. DATA CLEANING
# =====================================================

df["id_loan"] = (
    df["id_loan"].astype(str).str.strip()
)

df["period"] = (
    df["period"].astype(str).str.strip()
)

for col in numeric_features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

for col in categorical_features:
    df[col] = (
        df[col]
        .fillna("MISSING")
        .astype(str)
        .str.strip()
    )

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)

df = df.dropna(
    subset=[target, "id_loan"]
).copy()

df[target] = df[target].astype(int)

if not df[target].isin([0, 1]).all():
    raise ValueError("Target must be binary (0/1).")

X = df[features].copy()
y = df[target].copy()
groups = df["id_loan"].copy()

print("\n===== DATASET SUMMARY =====")
print("Observations:", len(df))
print("Unique loans:", groups.nunique())
print("Positive targets:", int(y.sum()))
print("Default rate:", round(y.mean(), 6))


# =====================================================
# 4. GROUP TRAIN / TEST SPLIT
# =====================================================

print("\nCreating GroupShuffleSplit...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

# Confirm no loan appears in both datasets.

overlap = (
    set(groups_train.unique())
    & set(groups_test.unique())
)

if overlap:
    raise ValueError(
        "Loan leakage detected between train and test."
    )

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both train and test must contain both target classes."
    )

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Training loans:", groups_train.nunique())
print("Testing loans:", groups_test.nunique())
print("Loan overlap:", len(overlap))


# =====================================================
# 5. PREPROCESSING
# =====================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# =====================================================
# 6. TRAIN PD MODEL
# =====================================================

print("\nTraining validation model...")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    ))
])

# Equal total weight per loan in training sample.

loan_counts_train = groups_train.value_counts()

train_weights = (
    1.0 / groups_train.map(loan_counts_train)
).to_numpy()

train_weights = (
    train_weights / train_weights.mean()
)

model.fit(
    X_train,
    y_train,
    model__sample_weight=train_weights
)

print("Model training completed.")


# =====================================================
# 7. PREDICTIONS
# =====================================================

print("\nGenerating test predictions...")

y_pred_proba = model.predict_proba(
    X_test
)[:, 1]

y_pred_class = model.predict(X_test)


# Equal total weight per loan in test sample.

loan_counts_test = groups_test.value_counts()

test_weights = (
    1.0 / groups_test.map(loan_counts_test)
).to_numpy()

test_weights = (
    test_weights / test_weights.mean()
)


# =====================================================
# 8. MODEL PERFORMANCE METRICS
# =====================================================

print("\n===== PD MODEL VALIDATION =====")

roc_auc = roc_auc_score(
    y_test,
    y_pred_proba
)

gini = 2 * roc_auc - 1

pr_auc = average_precision_score(
    y_test,
    y_pred_proba
)

brier = brier_score_loss(
    y_test,
    y_pred_proba
)

logloss = log_loss(
    y_test,
    y_pred_proba,
    labels=[0, 1]
)

weighted_roc_auc = roc_auc_score(
    y_test,
    y_pred_proba,
    sample_weight=test_weights
)

weighted_pr_auc = average_precision_score(
    y_test,
    y_pred_proba,
    sample_weight=test_weights
)

weighted_brier = brier_score_loss(
    y_test,
    y_pred_proba,
    sample_weight=test_weights
)

weighted_logloss = log_loss(
    y_test,
    y_pred_proba,
    sample_weight=test_weights,
    labels=[0, 1]
)

weighted_gini = 2 * weighted_roc_auc - 1

metrics = pd.DataFrame({
    "Metric": [
        "ROC_AUC",
        "GINI",
        "PR_AUC",
        "BRIER_SCORE",
        "LOG_LOSS"
    ],
    "Observation_Weighted": [
        roc_auc,
        gini,
        pr_auc,
        brier,
        logloss
    ],
    "Loan_Weighted": [
        weighted_roc_auc,
        weighted_gini,
        weighted_pr_auc,
        weighted_brier,
        weighted_logloss
    ]
})

print(metrics.to_string(index=False))

metrics.to_csv(
    OUTPUT_DIR / "pd_validation_metrics.csv",
    index=False
)


# =====================================================
# 9. CALIBRATION AND DECILE ANALYSIS
# =====================================================

print("\nCalculating calibration and decile analysis...")

calibration_data = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted_pd": y_pred_proba,
    "loan_weight": test_weights
})

# Rank first to handle duplicate predicted PD values.

calibration_data["rank"] = (
    calibration_data["predicted_pd"]
    .rank(method="first")
)

calibration_data["decile"] = pd.qcut(
    calibration_data["rank"],
    q=10,
    labels=False
) + 1

def weighted_mean(group, value_column):
    return np.average(
        group[value_column],
        weights=group["loan_weight"]
    )

decile_rows = []

for decile, group in calibration_data.groupby(
    "decile",
    observed=True
):

    decile_rows.append({
        "Decile": int(decile),
        "Number_of_Observations": len(group),
        "Number_of_Loans": groups_test.iloc[
            group.index
        ].nunique(),
        "Average_Predicted_PD": weighted_mean(
            group,
            "predicted_pd"
        ),
        "Observed_Default_Rate": weighted_mean(
            group,
            "actual"
        )
    })

decile_table = pd.DataFrame(decile_rows)

print("\n===== DECILE ANALYSIS =====")
print(decile_table.to_string(index=False))

decile_table.to_csv(
    OUTPUT_DIR / "pd_decile_analysis.csv",
    index=False
)


# =====================================================
# 10. CALIBRATION GRAPH
# =====================================================

plt.figure(figsize=(8, 6))

plt.plot(
    decile_table["Average_Predicted_PD"],
    decile_table["Observed_Default_Rate"],
    marker="o",
    label="PD Model"
)

max_value = max(
    decile_table["Average_Predicted_PD"].max(),
    decile_table["Observed_Default_Rate"].max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Average Predicted PD")
plt.ylabel("Observed Default Rate")
plt.title("PD Model Calibration")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pd_calibration.png",
    dpi=300
)

plt.close()


# =====================================================
# 11. ROC CURVE
# =====================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_pred_proba
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pd_roc_curve.png",
    dpi=300
)

plt.close()


# =====================================================
# 12. PRECISION-RECALL CURVE
# =====================================================

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_pred_proba
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"Average Precision = {pr_auc:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pd_precision_recall.png",
    dpi=300
)

plt.close()


# =====================================================
# 13. EXPORT TEST PREDICTIONS
# =====================================================

test_predictions = pd.DataFrame({
    "id_loan": groups_test.to_numpy(),
    "period": df.iloc[test_idx]["period"].to_numpy(),
    "actual_default_12m": y_test.to_numpy(),
    "predicted_pd_12m": y_pred_proba,
    "predicted_class": y_pred_class
})

test_predictions.to_csv(
    OUTPUT_DIR / "pd_test_predictions.csv",
    index=False
)


# =====================================================
# 14. SAVE VALIDATION MODEL
# =====================================================

import joblib

joblib.dump(
    model,
    OUTPUT_DIR / "logistic_pd_model_validation.joblib"
)


# =====================================================
# 15. FINAL SUMMARY
# =====================================================

print("\n===== VALIDATION SUMMARY =====")

print("ROC-AUC:", round(roc_auc, 6))
print("GINI:", round(gini, 6))
print("PR-AUC:", round(pr_auc, 6))
print("Brier Score:", round(brier, 6))
print("Log Loss:", round(logloss, 6))

print("\nLoan-weighted ROC-AUC:", round(weighted_roc_auc, 6))
print("Loan-weighted GINI:", round(weighted_gini, 6))
print("Loan-weighted PR-AUC:", round(weighted_pr_auc, 6))
print("Loan-weighted Brier:", round(weighted_brier, 6))
print("Loan-weighted Log Loss:", round(weighted_logloss, 6))

print("\nSaved files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    if file.is_file():
        print(file)

print("\nScript 27 completed successfully.")

