
import pandas as pd
from pathlib import Path

file_path = Path("data/schemes/Dataset/gov_myscheme_data.csv")

df = pd.read_csv(file_path)

print("Total schemes:", len(df))

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nFirst 5 schemes:")
print(df[["Scheme Name", "Level", "State / UT / Ministry"]].head())

# Select Central Government and Kerala schemes
selected = df[
    (df["Level"] == "Central Government") |
    (df["State / UT / Ministry"].str.contains(
        "Kerala", case=False, na=False
    ))
]

output_path = Path("data/schemes/kerala_central_schemes.csv")
selected.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\nCentral + Kerala schemes:", len(selected))
print("Saved to:", output_path)
