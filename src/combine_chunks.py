
import json
from pathlib import Path

PDF_FILE = Path("extracted_text/all_chunks.json")
SCHEME_FILE = Path("data/schemes/scheme_chunks.json")
OUTPUT_FILE = Path("data/schemes/combined_chunks.json")

with open(PDF_FILE, "r", encoding="utf-8") as file:
    pdf_chunks = json.load(file)

with open(SCHEME_FILE, "r", encoding="utf-8") as file:
    scheme_chunks = json.load(file)

# Combine both knowledge sources
combined_chunks = pdf_chunks + scheme_chunks

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        combined_chunks,
        file,
        indent=4,
        ensure_ascii=False
    )

print("Knowledge base combination completed!")
print("PDF chunks:", len(pdf_chunks))
print("Scheme chunks:", len(scheme_chunks))
print("Total chunks:", len(combined_chunks))
print("Saved to:", OUTPUT_FILE)
