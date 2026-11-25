from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "data_preparation"
OUTPUT_DIR = BASE_DIR / "outputs" / "region_comparison"
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

train = pd.read_csv(DATA_DIR / "train_woe.csv")
test = pd.read_csv(DATA_DIR / "test_woe.csv")

if "Region" not in train.columns or "Region" not in test.columns:
    raise ValueError("Region nije pronađen u WOE fajlovima.")

y_train = train[TARGET]
y_test = test[TARGET]


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

    return {
        "Model": name,
        "Train_AUC": train_auc,
        "Test_AUC": test_auc,
        "Test_Gini": 2 * test_auc - 1,
        "Test_Brier": brier_score_loss(y_test, test_pd),
        "AUC_Gap": train_auc - test_auc
    }


results = []

results.append(
    evaluate_model(
        "Baseline - without Region",
        BASE_FEATURES
    )
)

results.append(
    evaluate_model(
        "Extended - with Region",
        BASE_FEATURES + ["Region"]
    )
)

results_df = pd.DataFrame(results)

print("\nREGION MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "region_model_comparison.csv",
    index=False
)

print("\nResults saved in:", OUTPUT_DIR)

