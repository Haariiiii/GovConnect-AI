
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load the chunks created in Step 6
INPUT_FILE = Path("extracted_text/all_chunks.json")

# Where to save the embeddings
OUTPUT_FILE = Path("extracted_text/embeddings.npy")

# Load the embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Read chunks from JSON
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Extract only the text from each chunk
texts = [chunk["text"] for chunk in chunks]

print(f"Total chunks: {len(texts)}")

# Convert text into embeddings
print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

# Save embeddings
np.save(OUTPUT_FILE, embeddings)

print("\nEmbedding creation completed!")
print("Embedding shape:", embeddings.shape)
print("Saved to:", OUTPUT_FILE)
