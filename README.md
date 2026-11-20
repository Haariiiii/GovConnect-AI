<<<<<<< HEAD
# GovGuide AI – Quick Start Guide

## Prerequisites
- Python 3.13 (already installed)
- All dependencies installed via `pip install -r requirements.txt`
- An OpenAI API key (add it to a `.env` file at the project root).

## .env file example
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
TOP_K=5               # number of chunks to retrieve per query
```

## Running the API
```powershell
# From the project root (GovGuideAi)
uvicorn app.main:app --reload
```
The server will be available at `http://127.0.0.1:8000`.

## Endpoints
| Method | Path | Description |
|--------|------|------------|
| `GET` | `/search?query=...` | Returns top‑k relevant text chunks (metadata + similarity score). |
| `POST` | `/answer` | Body: `{ "question": "...", "top_k": 5 }`. Retrieves chunks and uses OpenAI to generate a concise answer with source citations. |

## Example usage (curl)
```bash
# Search
curl "http://127.0.0.1:8000/search?query=What%20documents%20are%20required%20for%20a%20scheme"

# Answer (POST)
cat <<EOF > payload.json
{
  "question": "What documents are required for a low‑income housing scheme?",
  "top_k": 5
}
EOF
curl -X POST "http://127.0.0.1:8000/answer" -H "Content-Type: application/json" -d @payload.json
```

## Testing the retrieval layer directly (optional)
You can run a quick Python script to ensure the FAISS index loads correctly:
```python
# test_retrieval.py
from app.retrieval import search, load_resources

load_resources()
results = search("What documents are required for a scheme?", top_k=3)
for r in results:
    print(f"Score: {r['score']:.4f}\nText: {r['text'][:200]}...\n")
```
Run with `python test_retrieval.py`.

## Extending the corpus
- Update `data/sources.yaml` with additional HTML, CSV, PDF, or RSS sources.
- Re‑run the pipeline: `python scripts/prepare_data.py`.
- The FAISS index and metadata will be regenerated automatically.

---
*Feel free to ask for any further customisation, such as adding a new source type, changing chunk size, or packaging the service with Docker.*
=======
# GovConnect-AI-
An intelligent RAG-based assistant for government scheme discovery and eligibility guidance.
>>>>>>> fd2c000ce863249811ec93723b54b38217961a56
