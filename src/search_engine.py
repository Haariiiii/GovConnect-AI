
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm import generate_rag_answer
import time
# ==========================================
# 1. LOAD KNOWLEDGE BASE
# ==========================================

INPUT_FILE = Path("data/schemes/combined_chunks.json")

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

documents = [chunk.get("text", "") for chunk in chunks]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

document_vectors = vectorizer.fit_transform(documents)

print("Knowledge base loaded successfully!")
print("Total chunks:", len(chunks))

MIN_SIMILARITY = 0.025


# ==========================================
# 2. SCHEME ALIASES
# ==========================================

SCHEME_ALIASES = {

    "pm-kisan": [
        "pm-kisan",
        "pm kisan",
        "pmkisan",
        "pradhan mantri kisan samman nidhi"
    ],

    "kasp": [
        "kasp",
        "karunya aarogya suraksha padhathi",
        "karunya arogy suraksha"
    ],

    "rainfed area development": [
        "rainfed area development",
        "rad scheme"
    ],

    "women scientist scheme-a": [
        "women scientist scheme-a",
        "women scientist scheme a",
        "wosa"
    ],

    "a2k+": [
        "a2k+",
        "a2k",
        "access to knowledge for technology development and dissemination"
    ],

    "agri-clinics and agri-business centres": [
        "agri-clinics and agri-business centres",
        "agri clinics and agri business centres",
        "agri-clinics",
        "agri-business centres",
        "acabc"
    ],

    "paramparagat krishi vikas yojana": [
        "paramparagat krishi vikas yojana",
        "pkvy"
    ]
}


# ==========================================
# 3. TEXT NORMALIZATION
# ==========================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z0-9]+", " ", text)

    return " ".join(text.split())


# ==========================================
# 4. DETECT QUESTION INTENT
# ==========================================

def detect_intent(question):

    question = normalize_text(question)

    intents = {

        "benefits": [
            "benefit",
            "benefits",
            "amount",
            "financial support",
            "financial assistance",
            "assistance",
            "subsidy",
            "loan",
            "funding",
            "grant",
            "fellowship",
            "scholarship",
            "how much"
        ],

        "eligibility": [
            "eligibility",
            "eligible",
            "qualify",
            "qualification",
            "who can apply",
            "age limit"
        ],

        "documents_required": [
            "document",
            "documents",
            "certificate",
            "proof",
            "paperwork"
        ],

        "application_process": [
            "apply",
            "application",
            "how to apply",
            "process",
            "registration",
            "register"
        ],

        "description": [
            "what is",
            "about",
            "purpose",
            "objective",
            "objectives",
            "aim",
            "explain"
        ]
    }

    matched_fields = []

    for field, keywords in intents.items():

        if any(keyword in question for keyword in keywords):
            matched_fields.append(field)

    return matched_fields


# ==========================================
# 5. DETECT QUERY FOCUS
# ==========================================

QUERY_FOCUS = {

    "financial": [
        "financial assistance",
        "financial support",
        "loan",
        "subsidy",
        "grant",
        "funding",
        "credit",
        "financial aid",
        "project cost",
        "training support",
        "composite subsidy"
    ],

    "eligibility": [
        "age",
        "income",
        "qualification",
        "occupation",
        "eligible",
        "residence",
        "farmer",
        "applicant",
        "belong",
        "category"
    ],

    "application": [
        "application",
        "apply",
        "registration",
        "register",
        "submit",
        "proposal",
        "online",
        "offline",
        "procedure"
    ],

    "documents": [
        "aadhaar",
        "certificate",
        "identity proof",
        "bank account",
        "land documents",
        "documents required",
        "proof"
    ],

    "objectives": [
        "objective",
        "objectives",
        "aim",
        "purpose",
        "develop",
        "support",
        "promote"
    ],

    "general_benefits": [
        "monthly",
        "amount",
        "free treatment",
        "coverage",
        "pension",
        "scholarship",
        "fellowship",
        "benefit"
    ]
}


def detect_query_focus(question):

    question = normalize_text(question)

    if any(word in question for word in [
        "financial assistance",
        "financial support",
        "loan",
        "subsidy",
        "funding",
        "grant"
    ]):
        return "financial"

    if any(word in question for word in [
        "eligibility",
        "eligible",
        "qualify",
        "who can apply"
    ]):
        return "eligibility"

    if any(word in question for word in [
        "how to apply",
        "application process",
        "registration",
        "register"
    ]):
        return "application"

    if any(word in question for word in [
        "documents",
        "document",
        "certificate",
        "proof"
    ]):
        return "documents"

    if any(word in question for word in [
        "objective",
        "objectives",
        "aim",
        "purpose"
    ]):
        return "objectives"

    return "general_benefits"


# ==========================================
# 6. IDENTIFY REQUESTED SCHEME
# ==========================================

def identify_scheme(question):

    normalized_question = normalize_text(question)

    requested_sources = set()

    # Match known scheme aliases
    for alias_group, aliases in SCHEME_ALIASES.items():

        alias_found = any(
            normalize_text(alias) in normalized_question
            for alias in aliases
        )

        if not alias_found:
            continue

        for chunk in chunks:

            source = chunk.get("source", "")

            normalized_source = normalize_text(source)

            if any(
                normalize_text(alias) in normalized_source
                for alias in aliases
            ):
                requested_sources.add(source)

    # Match complete scheme names directly
    for chunk in chunks:

        source = chunk.get("source", "")

        normalized_source = normalize_text(source)

        if (
            len(normalized_source) >= 8
            and normalized_source in normalized_question
        ):
            requested_sources.add(source)

    return requested_sources


# ==========================================
# 7. SEARCH KNOWLEDGE BASE
# ==========================================

def search_schemes(question, top_k=5):

    query_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    matched_fields = detect_intent(question)

    requested_sources = identify_scheme(question)

    candidates = []

    # --------------------------------------
    # STEP 1: STRICT SCHEME FILTERING
    # --------------------------------------

    for index, chunk in enumerate(chunks):

        source = chunk.get("source", "Unknown")

        # If a scheme is identified, exclude all other schemes
        if requested_sources and source not in requested_sources:
            continue

        similarity = float(similarities[index])

        if similarity < MIN_SIMILARITY:
            continue

        candidates.append({

            "index": index,

            "chunk": chunk,

            "similarity": similarity,

            "field": chunk.get("field", "PDF information")

        })

    if not candidates:
        return []

    # --------------------------------------
    # STEP 2: INFORMATION-TYPE FILTERING
    # --------------------------------------

    if matched_fields:

        exact_matches = [
            item for item in candidates
            if item["field"] in matched_fields
        ]

        # Preserve PDF chunks without field metadata
        pdf_matches = [
            item for item in candidates
            if item["field"] == "PDF information"
        ]

        if exact_matches:

            candidates = exact_matches + pdf_matches

        else:

            # FAQ/PDF fallback if requested field is absent
            candidates = [
                item for item in candidates
                if item["field"] in (
                    "faqs",
                    "PDF information"
                )
            ]

    if not candidates:
        return []

    # --------------------------------------
    # STEP 3: QUERY-SPECIFIC KEYWORD BOOST
    # --------------------------------------

    focus = detect_query_focus(question)

    focus_keywords = QUERY_FOCUS[focus]

    for item in candidates:

        score = item["similarity"]

        field = item["field"]

        content = normalize_text(
            item["chunk"].get("text", "")
        )

        # Matching information type gets priority
        if field in matched_fields:
            score += 0.5

        elif field == "PDF information":
            score += 0.1

        # Boost chunks containing relevant keywords
        keyword_matches = sum(
            1 for keyword in focus_keywords
            if normalize_text(keyword) in content
        )

        score += min(keyword_matches * 0.15, 0.75)

        item["score"] = score

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # --------------------------------------
    # STEP 4: REMOVE DUPLICATES
    # --------------------------------------

    results = []

    seen_chunks = set()

    for item in candidates:

        chunk = item["chunk"]

        source = chunk.get("source", "Unknown")

        field = item["field"]

        page = chunk.get("page", "Not available")

        chunk_number = chunk.get(
            "chunk_number",
            item["index"]
        )

        text_content = chunk.get("text", "")

        if not text_content.strip():
            continue

        chunk_key = (
            source,
            field,
            chunk_number,
            page
        )

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)

        results.append({

            "source": source,

            "field": field,

            "page": page,

            "official_link": chunk.get(
                "official_link", ""
            ),

            "score": item["score"],

            "similarity": item["similarity"],

            "text": text_content

        })

        if len(results) >= top_k:
            break

    return results


# ==========================================
# 8. DISPLAY RETRIEVED INFORMATION
# ==========================================

# ==========================================
# 8. GENERATE ANSWER USING GEMINI
# ==========================================

def generate_answer(question):

    results = search_schemes(question)

    print("\n========== GOVCONNECT AI ==========\n")

    if not results:

        print(
            "Sorry, the available documents do not "
            "contain sufficient relevant information "
            "to answer this question."
        )

        return

    print("Generating answer using Gemini...\n")

    try:

        answer = generate_rag_answer(
            question,
            results
        )

        print("ANSWER:\n")
        print(answer)

    except Exception as error:

        print("Unable to generate the answer using Gemini.")
        print("Error:", error)

    print("\n" + "=" * 60)


# ==========================================
# 9. INTERACTIVE CHATBOT
# ==========================================

if __name__ == "__main__":

    print("\nWelcome to GovGuide AI!")

    print("Ask questions about government schemes.")

    while True:

        question = input(
            "\nAsk about a government scheme (type exit to stop): "
        )

        if question.lower().strip() == "exit":

            print("\nThank you for using GovGuide AI!")

            break

        if not question.strip():

            print("Please enter a question.")

            continue

        generate_answer(question)
