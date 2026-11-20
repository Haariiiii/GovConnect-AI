"""FastAPI entrypoint for GovGuide AI.

The service exposes two endpoints:
* ``GET /search?query=...`` – returns the top k relevant text chunks from the FAISS index.
* ``POST /answer`` – takes a natural‑language question, retrieves the most relevant chunks, and uses OpenAI's
  ChatCompletion API to generate a concise answer with source references.

Configuration (via a ``.env`` file at the project root):
```
OPENAI_API_KEY=sk-...
TOP_K=5            # number of chunks to retrieve for each query
```
If ``OPENAI_API_KEY`` is not set the ``/answer`` endpoint will return an error.
"""

import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOP_K = int(os.getenv("TOP_K", "5"))

if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not set – /answer endpoint will be unavailable.")

# Initialise FastAPI app
app = FastAPI(title="GovGuide AI Retrieval Service")

# Import our retrieval utilities
from app.retrieval import search, load_resources

# Ensure resources are loaded at startup
@app.on_event("startup")
async def startup_event():
    load_resources()
    logging.info("Resources loaded – FAISS index ready.")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AnswerRequest(BaseModel):
    question: str
    top_k: int = TOP_K

class ChunkResult(BaseModel):
    id: int
    text: str
    score: float
    # any extra metadata stored in the JSON will be passed through
    source: Dict | None = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[ChunkResult]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/search", response_model=List[ChunkResult])
async def search_endpoint(query: str = Query(..., min_length=1), top_k: int = TOP_K):
    """Retrieve the most relevant chunks for *query*.
    Returns a list of ``ChunkResult`` objects sorted by similarity.
    """
    results = search(query, top_k=top_k)
    # Convert raw dicts to ChunkResult objects
    return [ChunkResult(**r) for r in results]

@app.post("/answer", response_model=AnswerResponse)
async def answer_endpoint(payload: AnswerRequest):
    """RAG‑style answer generation.
    1. Retrieve ``top_k`` chunks.
    2. Build a prompt that supplies the chunks as context.
    3. Call OpenAI ChatCompletion (gpt‑3.5‑turbo or gpt‑4) and return the answer.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    # 1️⃣ Retrieval
    chunks = search(payload.question, top_k=payload.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # 2️⃣ Prompt construction
    context_text = "\n\n---\n\n".join([c["text"] for c in chunks])
    system_prompt = (
        "You are an assistant that answers questions about Indian government schemes. "
        "Use ONLY the provided context snippets to formulate your answer. "
        "If the answer is not present in the context, respond with a brief statement that the information is unavailable."
    )
    user_prompt = (
        f"Context:\n{context_text}\n\nQuestion: {payload.question}\n\nProvide a concise answer and cite the source snippets (e.g., "
        f""Snippet 1, Snippet 2, ...")."
    )

    # 3️⃣ OpenAI call
    import openai
    openai.api_key = OPENAI_API_KEY
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        logging.exception("OpenAI request failed")
        raise HTTPException(status_code=502, detail=str(e))

    # Prepare source list for the response
    source_objs = [ChunkResult(**c) for c in chunks]
    return AnswerResponse(answer=answer_text, sources=source_objs)

# ---------------------------------------------------------------------------
# Run with: uvicorn app.main:app --reload
# ---------------------------------------------------------------------------
