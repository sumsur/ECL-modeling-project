import pandas as pd
from pathlib import Path

# 1. Load dataset
file_path = Path("C:\Python\ECL modeling\data_raw.csv")
df = pd.read_csv(file_path)

# 2. Basic information
print("\n--- DATASET DIMENSIONS ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA TYPES ---")
print(df.info())

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\n--- DUPLICATES ---")
print(f"Duplicate rows: {df.duplicated().sum()}")

# 3. Target variable analysis
if "Status" in df.columns:
    print("\n--- DEFAULT DISTRIBUTION ---")
    print(df["Status"].value_counts(dropna=False))
    print("\n--- DEFAULT RATE ---")
    print(df["Status"].value_counts(normalize=True, dropna=False))

# 4. Descriptive statistics
print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe(include="all").T)

# 5. Save basic report
Path("outputs").mkdir(exist_ok=True)

df.describe(include="all").T.to_csv(
    "outputs/descriptive_statistics.csv"
)

print("\n--- ALL VARIABLES ---")

for col in df.columns:
    print(f"{col}: {df[col].dtype}")

print("\n--- STATUS DISTRIBUTION ---")
print(df["Status"].value_counts(normalize=True) * 100)

