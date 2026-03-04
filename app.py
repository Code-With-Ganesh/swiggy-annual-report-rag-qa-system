# app.py - streamlit UI for Swiggy report QA
# run with: streamlit run app.py

import os, sys, time
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.rag_pipeline import index_exists
from src.embeddings import load_model
from src.vector_store import load_from_disk
from src.query_engine import init_gemini, ask

COOLDOWN = 3

st.set_page_config(page_title="Swiggy Report QA", page_icon="📄", layout="centered")

# some custom css
st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .answer-box {
        background-color: #f0f2f6; border-radius: 10px;
        padding: 1.2rem; margin: 0.5rem 0; border-left: 4px solid #ff5200;
    }
    .context-box {
        background-color: #fafafa; border-radius: 8px;
        padding: 0.8rem; margin: 0.3rem 0;
        border: 1px solid #e0e0e0; font-size: 0.85rem;
    }
    .source-tag {
        background-color: #ff5200; color: white;
        padding: 2px 8px; border-radius: 12px;
        font-size: 0.75rem; font-weight: bold;
    }
    .score-tag {
        background-color: #e8e8e8; padding: 2px 8px;
        border-radius: 12px; font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


# cached resource loaders
@st.cache_resource(show_spinner="Loading FAISS index...")
def get_index():
    return load_from_disk("vector_db")

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed_model():
    return load_model()

@st.cache_resource(show_spinner="Connecting to Gemini...")
def get_gemini(_key=None):
    return init_gemini()


# header
st.markdown("## 📄 Swiggy Annual Report QA System")
st.markdown(
    "Ask questions about the **Swiggy Annual Report FY 2023-24** "
    "and get grounded answers with page citations."
)
st.divider()

# sidebar
with st.sidebar:
    st.markdown("### 🔍 About This System")
    st.markdown(
        "A **RAG-based QA system** that answers questions from the "
        "Swiggy Annual Report FY 2023-24 (170 pages). "
        "It extracts text using OCR, builds a semantic search index, "
        "and generates grounded answers with page citations."
    )
    st.divider()

    st.markdown("### 🛠️ Tech Stack")
    st.markdown(
        "- **PDF Extraction:** PyMuPDF + Tesseract OCR\n"
        "- **Embeddings:** sentence-transformers (MiniLM)\n"
        "- **Vector DB:** FAISS\n"
        "- **LLM:** Google Gemini (auto-fallback)\n"
        "- **Frontend:** Streamlit"
    )

    if index_exists("vector_db"):
        st.success("Vector DB: Ready ✓ (319 chunks)")
    else:
        st.error("Vector DB: Not found — run `python main.py` first")

    st.divider()
    top_k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=5)
    show_context = st.checkbox("Show supporting context", value=True)

    st.divider()
    st.markdown("### 💡 Sample Questions")
    samples = [
        "What was Swiggy's total revenue in FY 2023-24?",
        "Who is the CEO of Swiggy?",
        "Who are the board members of Swiggy?",
        "What are Swiggy's key subsidiaries?",
        "What is Swiggy Instamart?",
    ]
    for q in samples:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill_question"] = q

# check if index is built
if not index_exists("vector_db"):
    st.warning("⚠️ Vector database not found. Run `python main.py` first to build it.")
    st.stop()

# load everything
index, chunks = get_index()
embed_model = get_embed_model()
gemini = get_gemini(None)

# chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prefill = st.session_state.pop("prefill_question", None)
question = st.chat_input("Ask a question about the Swiggy Annual Report...")
if prefill:
    question = prefill

# render past messages
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant", avatar="📄"):
        st.markdown(entry["answer"])
        if entry.get("sources"):
            st.caption(f"📌 Source pages: {', '.join(str(p) for p in entry['sources'])}")
        if show_context and entry.get("context"):
            with st.expander("📚 Supporting Context", expanded=False):
                for i, c in enumerate(entry["context"], 1):
                    snippet = c["text"][:300].replace("\n", " ")
                    st.markdown(
                        f'<div class="context-box">'
                        f'<span class="source-tag">Page {c["page_num"]}</span> '
                        f'<span class="score-tag">Score: {c["score"]:.2f}</span>'
                        f'<br><br>{snippet}...</div>',
                        unsafe_allow_html=True
                    )

if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0.0

# handle new question
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Searching the report..."):
            try:
                result = ask(question, gemini, index, chunks, embed_model, top_k=top_k)
                st.session_state.last_query_time = time.time()

                st.markdown(result["answer"])
                if result["sources"]:
                    st.caption(f"📌 Source pages: {', '.join(str(p) for p in result['sources'])}")

                if show_context and result["context"]:
                    with st.expander("📚 Supporting Context", expanded=False):
                        for i, c in enumerate(result["context"], 1):
                            snippet = c["text"][:300].replace("\n", " ")
                            st.markdown(
                                f'<div class="context-box">'
                                f'<span class="source-tag">Page {c["page_num"]}</span> '
                                f'<span class="score-tag">Score: {c["score"]:.2f}</span>'
                                f'<br><br>{snippet}...</div>',
                                unsafe_allow_html=True
                            )

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "context": result["context"]
                })
            except Exception as e:
                st.error(f"Error: {e}")
