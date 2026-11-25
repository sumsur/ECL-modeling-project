import pandas as pd
from pathlib import Path

FILE = Path(
    r"C:\Python\ECL modeling\data_raw\freddie_mac_sample\performance.csv"
)

df = pd.read_csv(
    FILE,
    nrows=10,
    low_memory=False
)

print("\n===== PERFORMANCE DATASET STRUCTURE =====")

for col in df.columns:
    print(col)

print("\n===== DATA TYPES =====")
print(df.dtypes.to_string())

print("\n===== SAMPLE DATA =====")
print(df.head(5).to_string())

print("\n===== POTENTIAL RECOVERY VARIABLES =====")

keywords = [
    "zero_balance",
    "liquidation",
    "proceeds",
    "foreclosure",
    "expense",
    "recovery",
    "repurchase",
    "disposition",
    "credit_enhancement",
    "upb",
    "delq"
]

for col in df.columns:
    if any(k in col.lower() for k in keywords):
        print(col)

