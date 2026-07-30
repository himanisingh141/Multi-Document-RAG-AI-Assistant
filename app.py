import os
import streamlit as st
from typing import List
from rag_pipeline import load_documents, create_or_load_vectorstore, ask_question
import logging

log = logging.getLogger(__name__)
st.set_page_config(page_title="Multi-Document RAG AI Assistant", layout="wide")
st.title("Multi-Document RAG AI Assistant")
st.write("Upload documents (PDF/DOCX/TXT). Index once; ask multiple questions. All local.")

# persistent path for vectorstore
PERSIST_DIR = "vectorstore"

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDFs, Texts, or DOCX files (multiple)", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True
)

# Keep a cached vectorstore / embeddings resource so we don't rebuild on every run
@st.cache_resource
def build_vectorstore_from_uploads(uploaded_files) -> dict:
    """
    Returns a dict with keys:
      - vectorstore
      - created (bool)
      - num_docs (int)
    Cached across runs until server restart or file change triggers new run.
    """
    if not uploaded_files:
        return {"vectorstore": None, "created": False, "num_docs": 0}
    docs = load_documents(uploaded_files)
    vs, created = create_or_load_vectorstore(docs, persist_dir=PERSIST_DIR)
    return {"vectorstore": vs, "created": created, "num_docs": len(docs)}

# If previously indexed content exists, load it to allow queries without re-upload
if os.path.exists(PERSIST_DIR):
    st.info("Persistent index found — loading. If you want to re-index, remove the 'vectorstore' folder.")
    # lazily load (cached)
    vs_info = build_vectorstore_from_uploads(None)
    if vs_info["vectorstore"] is None:
        # build from any uploaded files (none) will trigger load in pipeline (it checks disk)
        vs_info = build_vectorstore_from_uploads(uploaded_files)
else:
    vs_info = build_vectorstore_from_uploads(uploaded_files)

if vs_info["vectorstore"]:
    st.success(f"Vector store ready (indexed {vs_info['num_docs']} document objects).")

    query = st.text_input("Ask a question about your documents:")
    k = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=8, value=4)

    if query:
        with st.spinner("Generating answer..."):
            try:
                answer, citations = ask_question(vs_info["vectorstore"], query, k=k)
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
            else:
                st.subheader("Answer")
                st.success(answer.content)
                #st.write(answer)

                st.subheader("Sources")
                for d in citations:
                    src = d.metadata.get("source", "Unknown")
                    # Show a snippet and source
                    snippet = d.page_content.replace("\n", " ")
                    st.markdown(f"- **{src}** — _{snippet}..._")
else:
    st.info("Upload files and click 'Process' to index them.")
    if uploaded_files:
        if st.button("Process uploads and build index"):
            with st.spinner("Processing... this may take a while on first run"):
                vs_info = build_vectorstore_from_uploads(uploaded_files)
                st.experimental_rerun()

st.markdown("---")
st.caption("Tips: smaller embedding models are faster (all-MiniLM-L6-v2). To reduce hallucinations, the system uses a strict 'Use ONLY the provided context' prompt.")