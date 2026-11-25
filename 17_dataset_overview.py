from pathlib import Path
import pandas as pd

DATA_DIR = Path("data_raw/freddie_mac_sample")

files = {
    "Origination": DATA_DIR / "origination.csv",
    "Performance": DATA_DIR / "performance.csv"
}

for name, path in files.items():
    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        continue

    df = pd.read_csv(path, low_memory=False)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values (%):")
    print(
        (df.isnull().mean() * 100)
        .sort_values(ascending=False)
        .round(2)
        .to_string()
    )

    print("\nNumber of unique values per column:")
    print(df.nunique().to_string())

print("\n\nDATASET OVERVIEW COMPLETED")

