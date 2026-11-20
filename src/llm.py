
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.8-flash"


def generate_rag_answer(question, retrieved_chunks):

    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        source = chunk.get("source", "Unknown source")
        field = chunk.get("field", "General information")
        page = chunk.get("page", "Not available")
        text = chunk.get("text", "")
        official_link = chunk.get("official_link", "")

        context = f"""
        Source {index}: {source}
        Information type: {field}
        Page: {page}
        Information:
        {text}
        Official link: {official_link}
        """

        context_parts.append(context)

    context = "\n".join(context_parts)

    prompt = f"""
You are GovConnect AI, an assistant that helps citizens
understand Central Government and Kerala Government schemes.

Answer the user's question using ONLY the retrieved context below.

IMPORTANT RULES:
1. Do not invent facts, amounts, eligibility conditions,
   documents, or application procedures.
2. If the context does not contain the requested information,
   clearly say that the available sources do not provide it.
3. Explain the information in simple, clear language.
4. Organize the answer using headings or bullet points when useful.
5. Include source names and page numbers where available.
6. Include official links when they are provided.
7. Do not treat retrieval relevance scores as facts.
8. Do not claim that a person is officially eligible for a scheme.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

Provide a helpful answer based only on the retrieved context.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text
