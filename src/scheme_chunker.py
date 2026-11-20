
import json
import re
from pathlib import Path

INPUT_FILE = Path("data/schemes/scheme_records.json")
OUTPUT_FILE = Path("data/schemes/scheme_chunks.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_text(text):
    """Split text while trying to preserve sentence boundaries."""

    text = str(text).strip()

    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(sentence) > CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            start = 0

            while start < len(sentence):
                end = start + CHUNK_SIZE
                chunks.append(sentence[start:end].strip())
                start += CHUNK_SIZE - CHUNK_OVERLAP

            continue

        if len(current_chunk) + len(sentence) > CHUNK_SIZE:
            chunks.append(current_chunk.strip())

            overlap = current_chunk[-CHUNK_OVERLAP:]

            current_chunk = overlap + " " + sentence

        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def process_scheme(scheme):

    fields = {
        "description": scheme["description"],
        "eligibility": scheme["eligibility"],
        "general_eligibility": scheme["general_eligibility"],
        "exclusions": scheme["exclusions"],
        "benefits": scheme["benefits"],
        "application_process": scheme["application_process"],
        "documents_required": scheme["documents_required"],
        "faqs": scheme["faqs"]
    }

    scheme_chunks = []

    for field_name, field_text in fields.items():

        if not str(field_text).strip():
            continue

        text = (
            f"Scheme Name: {scheme['scheme_name']}\n"
            f"Government Level: {scheme['government_level']}\n"
            f"Information Type: {field_name}\n"
            f"Information: {field_text}"
        )

        chunks = split_text(text)

        for number, chunk in enumerate(chunks, start=1):

            scheme_chunks.append({
                "source": scheme["scheme_name"],
                "government_level": scheme["government_level"],
                "official_link": scheme["official_link"],
                "field": field_name,
                "chunk_number": number,
                "text": chunk
            })

    return scheme_chunks


# Load scheme records

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    schemes = json.load(file)


# Process all schemes

all_chunks = []

for scheme in schemes:
    chunks = process_scheme(scheme)
    all_chunks.extend(chunks)


# Save the new chunks

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(
        all_chunks,
        file,
        indent=4,
        ensure_ascii=False
    )


print("Field-based chunking completed!")
print("Total schemes:", len(schemes))
print("Total chunks:", len(all_chunks))
print("Saved to:", OUTPUT_FILE)
