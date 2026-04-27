import pandas as pd

CSV_PATH = "data/drowsiness_dataset.csv"

df = pd.read_csv(CSV_PATH)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset shape ---")
print(df.shape)

print("\n--- Label counts ---")
print(df["status"].value_counts())

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Basic statistics ---")
print(df.describe())