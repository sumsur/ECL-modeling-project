import pandas as pd

df = pd.read_csv(r"C:\Python\ECL modeling\data_raw.csv")

print("YEAR DISTRIBUTION")
print(df["year"].value_counts(dropna=False).sort_index())

print("\nREGION DISTRIBUTION")
print(df["Region"].value_counts(dropna=False))

print("\nNUMBER OF UNIQUE YEARS:")
print(df["year"].nunique())

print("\nNUMBER OF UNIQUE REGIONS:")
print(df["Region"].nunique())