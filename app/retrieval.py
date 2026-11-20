"""Utility functions for retrieval using the FAISS index.

The pipeline builds a FAISS index stored at ``data/faiss.index`` and a JSON
metadata file ``data/chunks_metadata.json`` that maps each vector ID to the
original text chunk and its source information.

This module loads the index, prepares an embedding function (SentenceTransformer
if available, otherwise a TF‑IDF fallback), and provides a ``search`` function
that returns the most relevant chunks for a given query.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

# Embedding model – try SentenceTransformer first, fall back to TF‑IDF
try:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("BAAI/bge-large")
    def embed(texts: List[str]):
        """Return L2‑normalized embeddings for a list of texts."""
        return _model.encode(texts, normalize_embeddings=True)
except Exception as e:
    logging.warning(f"SentenceTransformer not available ({e}); using TF‑IDF fallback.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    _vectorizer = TfidfVectorizer(max_features=5000)
    _fitted = False
    def embed(texts: List[str]):
        global _fitted, _vectorizer
        if not _fitted:
            # Fit on the whole corpus the first time – we will replace this with the
            # actual corpus embeddings later in ``load_resources``.
            _vectorizer.fit(texts)
            _fitted = True
        arr = _vectorizer.transform(texts).toarray().astype('float32')
        # L2‑normalize rows
        norm = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        return arr / norm

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FAISS_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunks_metadata.json"

# Global objects – loaded lazily
_index: faiss.IndexFlatIP | None = None
_metadata: List[Dict] = []

def load_resources():
    global _index, metadata
    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")
    _index = faiss.read_index(str(FAISS_PATH))
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {META_PATH}")
    metadata = json.load(open(META_PATH, "r", encoding="utf-8"))
    logging.info(f"Loaded FAISS index with {_index.ntotal} vectors and {len(metadata)} metadata entries.")

def search(query: str, top_k: int = 5) -> List[Dict]:
    """Return the top‑k most similar chunks for *query*.

    Each result dict contains:
    - ``id``   : internal chunk ID (matches ``metadata`` index)
    - ``text`` : the raw chunk text
    - ``score``: similarity score (inner product, i.e. cosine similarity after normalisation)
    - any additional fields stored in the metadata JSON.
    """
    if _index is None:
        load_resources()
    q_vec = embed([query])[0].astype('float32')
    D, I = _index.search(np.expand_dims(q_vec, 0), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx].copy()
        entry.update({"score": float(score), "id": idx})
        results.append(entry)
    return results
