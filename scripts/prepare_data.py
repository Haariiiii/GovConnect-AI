# -*- coding: utf-8 -*-
"""GovGuide AI – Data extraction & preparation pipeline

This script performs a simple end‑to‑end workflow:
1. Load the source catalogue (sources.yaml).
2. Download raw artefacts (HTML pages, PDFs, CSV/JSON, RSS feeds).
3. Convert each artefact to plain‑text + minimal metadata.
4. Chunk the text into ~250‑word pieces.
5. Compute embeddings using a sentence‑transformer model.
6. Store embeddings in a local FAISS index and keep a SQLite metadata table.

The pipeline is intentionally modular so you can replace components (e.g. use Chroma instead of FAISS).
"""

import os
import yaml
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser as date_parser

# PDF extraction – we use pdfminer.six (fallback to PyMuPDF if available)
from io import BytesIO
from pdfminer.high_level import extract_text

# Embeddings
# Attempt to import SentenceTransformer; if unavailable fall back to TF-IDF
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None
    fallback_error = e
    # We'll use TF-IDF later

import numpy as np
import faiss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project root (GovGuideAi)
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CHUNK_DIR = DATA_DIR / "chunks"
METADATA_DB = DATA_DIR / "metadata.db"  # SQLite DB for metadata (optional)
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"

EMBEDDING_MODEL_NAME = "BAAI/bge-large"  # open‑source English model, good for Indian context
CHUNK_SIZE_WORDS = 250

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def hash_content(content: str) -> str:
    """Return a deterministic SHA‑256 hash for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def load_sources() -> List[Dict[str, Any]]:
    src_path = BASE_DIR / "data" / "sources.yaml"
    with open(src_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("sources", [])

# ---------------------------------------------------------------------------
# Robust download helpers that log errors instead of raising
def safe_get(url: str):
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def download_html(url: str) -> str:
    resp = safe_get(url)
    return resp.text if resp else ""

def download_pdf(url: str) -> bytes:
    resp = safe_get(url)
    return resp.content if resp else b""

def download_csv(url: str) -> pd.DataFrame:
    resp = safe_get(url)
    if not resp:
        return pd.DataFrame()
    try:
        # Let pandas infer separator; if it fails we fallback to csv module
        return pd.read_csv(pd.compat.StringIO(resp.text))
    except Exception as e:
        logging.warning(f"pandas failed to parse CSV from {url}: {e}. Trying fallback.")
        import csv, io
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows[1:], columns=rows[0])

def download_rss(url: str) -> List[Dict[str, str]]:
    resp = safe_get(url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.content, "xml")
    items = []
    for item in soup.find_all("item"):
        items.append({
            "title": item.title.text if item.title else "",
            "link": item.link.text if item.link else "",
            "pubDate": item.pubDate.text if item.pubDate else "",
            "description": item.description.text if item.description else "",
        })
    return items
# ---------------------------------------------------------------------------
def download_html(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text

def download_pdf(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content

def download_csv(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

def download_rss(url: str) -> List[Dict[str, str]]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "xml")
    items = []
    for item in soup.find_all("item"):
        items.append({
            "title": item.title.text if item.title else "",
            "link": item.link.text if item.link else "",
            "pubDate": item.pubDate.text if item.pubDate else "",
            "description": item.description.text if item.description else "",
        })
    return items

# ---------------------------------------------------------------------------
# Extraction / cleaning
# ---------------------------------------------------------------------------
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style elements
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    return extract_text(BytesIO(pdf_bytes))

def process_source(source: Dict[str, Any]):
    name = source["name"]
    src_type = source["type"]
    url = source["url"]
    logging.info(f"Processing source {name} ({src_type})")

    if src_type == "html":
        html = download_html(url)
        raw_path = RAW_DIR / f"{name}.html"
        raw_path.write_text(html, encoding="utf-8")
        text = html_to_text(html)
        clean_path = CLEAN_DIR / f"{name}.txt"
        write_text(clean_path, text)
        # Attempt to discover linked PDFs on the same page (simple heuristic)
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith('.pdf'):
                pdf_url = href if href.startswith('http') else requests.compat.urljoin(url, href)
                try:
                    pdf_bytes = download_pdf(pdf_url)
                    pdf_hash = hash_content(pdf_url)
                    pdf_raw_path = RAW_DIR / f"{name}_{pdf_hash}.pdf"
                    pdf_raw_path.write_bytes(pdf_bytes)
                    pdf_text = pdf_bytes_to_text(pdf_bytes)
                    pdf_clean_path = CLEAN_DIR / f"{name}_{pdf_hash}.txt"
                    write_text(pdf_clean_path, pdf_text)
                except Exception as e:
                    logging.warning(f"Failed to fetch PDF {pdf_url}: {e}")

    elif src_type == "pdf":
        pdf_bytes = download_pdf(url)
        pdf_hash = hash_content(url)
        raw_path = RAW_DIR / f"{name}_{pdf_hash}.pdf"
        raw_path.write_bytes(pdf_bytes)
        text = pdf_bytes_to_text(pdf_bytes)
        clean_path = CLEAN_DIR / f"{name}_{pdf_hash}.txt"
        write_text(clean_path, text)

    elif src_type == "csv":
        df = download_csv(url)
        raw_path = RAW_DIR / f"{name}.csv"
        df.to_csv(raw_path, index=False)
        # Convert each row to a plain‑text record (you may refine this later)
        records = []
        for _, row in df.iterrows():
            rec = ", ".join([f"{col}: {row[col]}" for col in df.columns])
            records.append(rec)
        clean_path = CLEAN_DIR / f"{name}.txt"
        write_text(clean_path, "\n\n".join(records))

    elif src_type == "rss":
        items = download_rss(url)
        raw_path = RAW_DIR / f"{name}.json"
        raw_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        # Flatten each item into text
        texts = []
        for it in items:
            parts = [it.get('title',''), it.get('description','')]
            if it.get('pubDate'):
                # Normalise date for later filtering
                try:
                    dt = date_parser.parse(it['pubDate'])
                    parts.append(str(dt.date()))
                except Exception:
                    pass
            texts.append(" \n ".join(filter(None, parts)))
        clean_path = CLEAN_DIR / f"{name}.txt"
        write_text(clean_path, "\n\n".join(texts))

    else:
        logging.warning(f"Unsupported source type: {src_type}")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size_words: int = CHUNK_SIZE_WORDS) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunk = " ".join(words[i:i + chunk_size_words])
        chunks.append(chunk)
    return chunks

def prepare_chunks():
    for txt_file in CLEAN_DIR.glob("*.txt"):
        content = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(content)
        for idx, chunk in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            out_path = CHUNK_DIR / f"{txt_file.stem}_{idx}_{chunk_hash[:8]}.txt"
            write_text(out_path, chunk)

# ---------------------------------------------------------------------------
# Embedding & vector store
# ---------------------------------------------------------------------------
def build_faiss_index():
    """Create a FAISS index for the chunk vectors.
    Uses a SentenceTransformer model if available; otherwise falls back to a
    TF‑IDF representation (dense via .toarray()). Both pathways produce
    normalized vectors suitable for inner‑product search.
    """
    meta_records = []
    # Collect texts first
    texts = []
    for chunk_file in CHUNK_DIR.glob("*.txt"):
        txt = chunk_file.read_text(encoding="utf-8")
        texts.append(txt)
        meta_records.append({
            "id": chunk_file.stem,
            "path": str(chunk_file),
            "text": txt,
        })

    if not texts:
        logging.error("No chunks found – aborting index creation")
        return

    if SentenceTransformer is not None:
        # Use the BGE large model
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = model.encode(texts, normalize_embeddings=True)
    else:
        # Fallback: TF‑IDF (dense) – good enough for small corpora
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        sparse = vectorizer.fit_transform(texts)
        # Convert to dense float32 and L2‑normalize rows
        embeddings = sparse.toarray().astype('float32')
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings = embeddings / norm

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    # Persist metadata for lookup
    meta_path = DATA_DIR / "chunks_metadata.json"
    meta_path.write_text(json.dumps(meta_records, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(f"FAISS index built with {len(embeddings)} vectors (dim={dim}). Saved to {FAISS_INDEX_PATH}")

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sources = load_sources()
    for src in sources:
        try:
            process_source(src)
        except Exception as exc:
            logging.exception(f"Failed processing source {src.get('name')}: {exc}")
    logging.info("Completed raw download & cleaning.")
    prepare_chunks()
    logging.info("Text chunking finished.")
    build_faiss_index()
    logging.info("Embedding & index creation completed.")
