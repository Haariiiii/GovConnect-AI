
import fitz
from pathlib import Path

# Folder containing the government scheme PDFs
PDF_FOLDER = Path("data/schemes")

# Folder where extracted text will be saved
OUTPUT_FOLDER = Path("extracted_text")

OUTPUT_FOLDER.mkdir(exist_ok=True)

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    all_text = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        all_text.append(
            f"\n--- Page {page_number} ---\n{text}"
        )

    document.close()

    return "\n".join(all_text)


# Read every PDF in the schemes folder
for pdf_path in PDF_FOLDER.glob("*.pdf"):

    print(f"Reading: {pdf_path.name}")

    extracted_text = extract_text_from_pdf(pdf_path)

    # Save the extracted content as a text file
    output_file = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"
    output_file.write_text(extracted_text, encoding="utf-8")

    print(f"Saved: {output_file}")

print("PDF extraction completed!")
