
import pandas as pd

df = pd.read_csv("data/schemes/kerala_central_schemes.csv")

columns = [
    "Description",
    "Eligibility Criteria",
    "Benefits",
    "Application Process",
    "Documents Required"
]

print("DATASET QUALITY REPORT\n")
print("Total schemes:", len(df))

for column in columns:
    available = df[column].notna() & (
        df[column].astype(str).str.strip() != ""
    )

    print(f"{column}: {available.sum()} / {len(df)}")

print("\nSample scheme records:\n")

for _, row in df.head(5).iterrows():
    print("Scheme:", row["Scheme Name"])
    print("Benefits:", str(row["Benefits"])[:300])
    print("Eligibility:", str(row["Eligibility Criteria"])[:300])
    print("-" * 60)
