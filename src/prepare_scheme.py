
import pandas as pd
import json
from pathlib import Path

INPUT_FILE = Path("data/schemes/kerala_central_schemes.csv")
OUTPUT_FILE = Path("data/schemes/scheme_records.json")

# Load the scheme dataset
df = pd.read_csv(INPUT_FILE)
df = df.fillna("")

scheme_records = []

for _, row in df.iterrows():

    # Clean and format required documents
    documents = str(row["Documents Required"]).strip()

    # Fix PM-KISAN document formatting
    if (
        "Aadhaar CardLandholding papersSavings Bank Account"
        in documents
    ):
        documents = (
            "• Aadhaar Card\n"
            "• Landholding papers\n"
            "• Savings Bank Account"
        )

    scheme = {
        "scheme_name": row["Scheme Name"],
        "government_level": row["Level"],
        "state_ministry": row["State / UT / Ministry"],
        "category": row["Tags / Categories"],
        "description": row["Description"],
        "eligibility": row["Eligibility Criteria"],
        "general_eligibility": row["Eligibility (General)"],
        "exclusions": row["Exclusions / Ineligibility"],
        "benefits": row["Benefits"],
        "application_process": row["Application Process"],
        "documents_required": documents,
        "faqs": row["Frequently Asked Questions (FAQs)"],
        "official_link": row["Official Link"],
        "myscheme_url": row["MyScheme URL"]
    }

    scheme_records.append(scheme)

# Save prepared scheme records
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        scheme_records,
        file,
        indent=4,
        ensure_ascii=False
    )

print("Scheme data preparation completed!")
print("Total records:", len(scheme_records))
print("Saved to:", OUTPUT_FILE)

# Verify PM-KISAN document formatting
pm_kisan = next(
    (
        scheme for scheme in scheme_records
        if "PM-KISAN" in scheme["official_link"].upper()
    ),
    None
)

if pm_kisan:
    print("\nPM-KISAN documents after formatting:")
    print(pm_kisan["documents_required"])
else:
    print("\nPM-KISAN record not found.")
