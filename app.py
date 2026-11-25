
import time
import streamlit as st

from src.search_engine import search_schemes
from src.llm import generate_rag_answer


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="GovConnect AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: #f5f8f6;
    color: #17251e;
}

[data-testid="stAppViewContainer"] {
    background-color: #f5f8f6;
}

/* Main content width and spacing */

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* General text visibility */

.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp strong,
.stMarkdown {
    color: #17251e !important;
}

/* Header */

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #176b52 !important;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #64756b !important;
    margin-bottom: 20px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #eaf2ed;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #17251e !important;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #e1e9e3;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Chat message text */

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: #17251e !important;
}

/* Chat input */

[data-testid="stChatInput"] {
    background-color: #ffffff !important;
}

[data-testid="stChatInput"] textarea {
    color: #17251e !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: #17251e !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #718078 !important;
    -webkit-text-fill-color: #718078 !important;
}

/* Buttons */

.stButton > button {
    background-color: #176b52;
    color: #ffffff !important;
    border: none;
    border-radius: 9px;
    padding: 9px 14px;
    font-weight: 600;
}

.stButton > button p,
.stButton > button span {
    color: #ffffff !important;
}

.stButton > button:hover {
    background-color: #10543f;
    border: none;
}

/* Links */

.stApp a {
    color: #087b54 !important;
    font-weight: 600;
}

/* Source containers */

[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #dce7df;
    border-radius: 10px;
}

/* Divider */

hr {
    border-color: #dce7df !important;
}

/* Footer */

.footer {
    text-align: center;
    color: #718078 !important;
    font-size: 13px;
    padding-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🏛️ GovConnect AI")

    st.caption("Government Scheme Discovery Assistant")

    st.divider()

    st.markdown("### About the Project")

    st.write(
        "GovConnect AI is an intelligent assistant that "
        "helps citizens discover and understand Central "
        "Government and Kerala Government schemes."
    )

    st.write(
        "It uses Retrieval-Augmented Generation (RAG) "
        "to retrieve information from available scheme "
        "documents and generate helpful answers."
    )

    st.divider()

    st.markdown("### 💡 Try asking")

    example_questions = [
        "Who is eligible for PM-KISAN?",
        "What are the benefits of KASP?",
        "What documents are required for WOS-A?",
        "What is the objective of A2K+?",
        "Explain the Rainfed Area Development scheme."
    ]

    for question in example_questions:

        if st.button(
            question,
            key=f"example_{question}",
            use_container_width=True
        ):
            st.session_state.pending_question = question

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.divider()

    st.caption("GovConnect AI | Final Year Project")


# =====================================================
# MAIN HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🏛️ GovConnect AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Connecting Citizens with the Right Government Schemes.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# WELCOME SECTION
# =====================================================

if not st.session_state.messages:

    st.markdown("### Welcome! 👋")

    st.write(
        "I can help you understand government schemes, "
        "their benefits, eligibility criteria, required "
        "documents and application procedures."
    )

    st.info(
        "Ask your question below to get started. "
        "You can also choose an example question from the sidebar."
    )

    st.markdown("#### What can I help you with?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🎯 Eligibility**")
        st.caption("Understand who can apply for a scheme.")

    with col2:
        st.markdown("**💰 Benefits**")
        st.caption("Explore financial and other benefits.")

    with col3:
        st.markdown("**📄 Documents**")
        st.caption("Find required documents and procedures.")


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for index, message in enumerate(st.session_state.messages):

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("📚 View Sources and References"):

                for source_index, source in enumerate(
                    message["sources"], start=1
                ):

                    st.markdown(
                        f"**Source {source_index}: "
                        f"{source.get('source', 'Unknown source')}**"
                    )

                    st.caption(
                        f"Information type: "
                        f"{source.get('field', 'General information')}"
                    )

                    st.caption(
                        f"Page: {source.get('page', 'Not available')}"
                    )

                    official_link = source.get("official_link")

                    if official_link:
                        st.markdown(
                            f"[🔗 Visit Official Source]({official_link})"
                        )

                    with st.expander("View Retrieved Text"):
                        st.write(source.get("text", ""))

                    st.divider()


# =====================================================
# QUESTION INPUT
# =====================================================

user_question = st.chat_input(
    "Ask anything about Central or Kerala Government schemes..."
)

# Handle sidebar example questions

if st.session_state.pending_question:

    user_question = st.session_state.pending_question
    st.session_state.pending_question = None


# =====================================================
# RAG PIPELINE
# =====================================================

if user_question:

    # Display user message

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate assistant response

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "🔍 Searching government scheme information..."
            ):

                start_time = time.time()

                # STEP 1: RETRIEVAL

                results = search_schemes(user_question)

                retrieval_time = time.time() - start_time

            if not results:

                answer = (
                    "I could not find relevant information in "
                    "the available government scheme documents. "
                    "Please try asking about a specific government scheme."
                )

            else:

                # STEP 2: GENERATION

                with st.spinner(
                    "🤖 Generating your answer using Gemini..."
                ):

                    llm_start = time.time()

                    answer = generate_rag_answer(
                        user_question,
                        results
                    )

                    generation_time = time.time() - llm_start

            # DISPLAY ANSWER

            st.markdown(answer)

            st.caption(
                f"Retrieval: {retrieval_time:.2f}s"
            )

            if results:
                st.caption(
                    f"Answer generation: {generation_time:.2f}s"
                )

            # SAVE MESSAGE

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": results
            })

            # SOURCE DISPLAY

            if results:

                with st.expander("📚 View Sources and References"):

                    for source_index, source in enumerate(
                        results, start=1
                    ):

                        st.markdown(
                            f"**Source {source_index}: "
                            f"{source.get('source', 'Unknown source')}**"
                        )

                        st.caption(
                            f"Information type: "
                            f"{source.get('field', 'General information')}"
                        )

                        st.caption(
                            f"Page: {source.get('page', 'Not available')}"
                        )

                        official_link = source.get("official_link")

                        if official_link:
                            st.markdown(
                                f"[🔗 Visit Official Source]({official_link})"
                            )

                        with st.expander("View Retrieved Text"):
                            st.write(source.get("text", ""))

                        st.divider()

        except Exception as error:

            error_message = (
                "I am currently unable to generate your answer. "
                "This may be due to a temporary API or connection issue. "
                "Please try again in a moment."
            )

            st.error(error_message)

            # Show technical details for debugging

            with st.expander("Technical Error Details"):
                st.code(str(error))

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'GovConnect AI | RAG-Based Government Scheme Discovery '
    'and Eligibility Guidance<br>'
    'Information is provided for guidance and should be '
    'verified with official government sources.'
    '</div>',
    unsafe_allow_html=True
)
