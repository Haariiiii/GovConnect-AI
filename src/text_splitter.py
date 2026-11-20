
import json
import re
from pathlib import Path

INPUT_FOLDER = Path("extracted_text")
OUTPUT_FOLDER = Path("extracted_text")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_into_chunks(text, chunk_size, overlap):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_document(file_path):
    text = file_path.read_text(encoding="utf-8")

    # Separate the document into pages
    pages = re.split(r"--- Page (\d+) ---", text)

    document_chunks = []

    # Page number and page content appear alternately
    for i in range(1, len(pages), 2):
        page_number = int(pages[i])
        page_text = pages[i + 1].strip()

        chunks = split_into_chunks(
            page_text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for chunk_number, chunk in enumerate(chunks, start=1):
            document_chunks.append({
                "source": file_path.stem,
                "page": page_number,
                "chunk_number": chunk_number,
                "text": chunk
            })

    return document_chunks


all_chunks = []

for file_path in INPUT_FOLDER.glob("*.txt"):
    print(f"Processing {file_path.name}...")

    chunks = process_document(file_path)
    all_chunks.extend(chunks)

output_file = OUTPUT_FOLDER / "all_chunks.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(all_chunks, file, indent=4, ensure_ascii=False)

print(f"\nTotal chunks created: {len(all_chunks)}")
print(f"Saved to: {output_file}")
