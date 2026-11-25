import pandas as pd
import numpy as np
import joblib

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
    log_loss
)


# =====================================================
# 1. PATHS AND DATA
# =====================================================

BASE_DIR = Path(r"C:\Python\ECL modeling")

INPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "pd_cleaning"
    / "pd_modeling_dataset_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "pd_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# =====================================================
# 2. TARGET AND FEATURES
# =====================================================

target = "target_12m"
group_column = "id_loan"

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

required_columns = features + [target, group_column]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.dropna(subset=[target, group_column]).copy()

X = df[features].copy()
y = df[target].astype(int)
groups = df[group_column]

# Ensure categorical variables are treated as categories
for col in categorical_features:
    X[col] = X[col].fillna("MISSING").astype(str)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget rate:", round(y.mean(), 5))
print("Unique loans:", groups.nunique())


# =====================================================
# 3. TRAIN / TEST SPLIT BY LOAN
# =====================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))

print("Training loans:", groups_train.nunique())
print("Testing loans:", groups_test.nunique())

print("Training target rate:", round(y_train.mean(), 5))
print("Testing target rate:", round(y_test.mean(), 5))

# Check that no loan appears in both sets
assert set(groups_train).isdisjoint(set(groups_test))

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Training or testing set contains only one target class."
    )


# =====================================================
# 4. PREPROCESSING
# =====================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# =====================================================
# 5. LOGISTIC REGRESSION
# =====================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight=None
            )
        )
    ]
)


# =====================================================
# 6. EQUAL TOTAL WEIGHT PER LOAN
# =====================================================

loan_observation_counts = groups_train.value_counts()

sample_weights = (
    groups_train.map(loan_observation_counts)
    .rdiv(1.0)
    .to_numpy()
)

# Normalize weights to mean 1
sample_weights = sample_weights / sample_weights.mean()


# =====================================================
# 7. MODEL FIT
# =====================================================

print("\nFitting logistic regression...")

model.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weights
)

print("Model training completed.")


# =====================================================
# 8. MODEL EVALUATION
# =====================================================

predictions = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, predictions)
pr_auc = average_precision_score(y_test, predictions)
brier = brier_score_loss(y_test, predictions)
logloss = log_loss(y_test, predictions)

metrics = pd.DataFrame({
    "Metric": [
        "ROC AUC",
        "Gini",
        "PR AUC",
        "Brier Score",
        "Log Loss"
    ],
    "Value": [
        roc_auc,
        2 * roc_auc - 1,
        pr_auc,
        brier,
        logloss
    ]
})

print("\n===== PD MODEL PERFORMANCE =====")
print(metrics.to_string(index=False))


# =====================================================
# 9. SAVE RESULTS
# =====================================================

test_results = pd.DataFrame({
    "id_loan": groups_test.to_numpy(),
    "actual_target_12m": y_test.to_numpy(),
    "predicted_pd_12m": predictions
})

metrics.to_csv(
    OUTPUT_DIR / "pd_model_metrics.csv",
    index=False
)

test_results.to_csv(
    OUTPUT_DIR / "pd_test_predictions.csv",
    index=False
)

joblib.dump(
    model,
    OUTPUT_DIR / "logistic_pd_model.joblib"
)

print("\nFiles saved:")
print(OUTPUT_DIR / "pd_model_metrics.csv")
print(OUTPUT_DIR / "pd_test_predictions.csv")
print(OUTPUT_DIR / "logistic_pd_model.joblib")

print("\nPD model development completed.")

