import os
import streamlit as st
import logging
from rag_pipeline import load_documents, create_or_load_vectorstore, ask_question

log = logging.getLogger(__name__)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Card Container Styling */
    .metric-card {
        background-color: var(--background-secondary-color, #f0f2f6);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 10px;
    }
    /* Citation styling */
    .citation-title {
        font-weight: 600;
        color: var(--primary-color, #ff4b4b);
    }
</style>
""", unsafe_allow_html=True)

PERSIST_DIR = "vectorstore"

# --- CACHED VECTORSTORE BUILDING ---
@st.cache_resource
def build_vectorstore_from_uploads(uploaded_files) -> dict:
    if not uploaded_files:
        return {"vectorstore": None, "created": False, "num_docs": 0}
    docs = load_documents(uploaded_files)
    vs, created = create_or_load_vectorstore(docs, persist_dir=PERSIST_DIR)
    return {"vectorstore": vs, "created": created, "num_docs": len(docs)}

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Document Upload Section
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt", "md", "docx"], 
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, MD, DOCX"
    )
    
    # Retrieval Tuning
    k_chunks = st.slider("Retrieved Chunks (k)", min_value=1, max_value=6, value=4)
    
    st.markdown("---")
    st.subheader("Index Status")
    
    # Manage Index Loading / Clearing
    if os.path.exists(PERSIST_DIR):
        st.success("🟢 Persistent Index Active")
        vs_info = build_vectorstore_from_uploads(None)
        if vs_info["vectorstore"] is None:
            vs_info = build_vectorstore_from_uploads(uploaded_files)
    else:
        st.warning("🔴 No Index Found")
        vs_info = build_vectorstore_from_uploads(uploaded_files)

    if uploaded_files and st.button("🔄 Index Uploaded Files", use_container_width=True):
        with st.spinner("Processing documents & updating embeddings..."):
            vs_info = build_vectorstore_from_uploads(uploaded_files)
            st.toast("Documents indexed successfully!", icon="✅")

# --- MAIN PAGE HEADER ---
st.title("🤖 Modern RAG Knowledge Assistant")
st.caption("Ask questions strictly from your uploaded documentation.")

# --- TABS INTERFACE ---
tab_chat, tab_docs = st.tabs(["💬 Assistant Chat", "📚 Document Workspace"])

# --- TAB 1: CHAT INTERFACE ---
with tab_chat:
    if not vs_info["vectorstore"]:
        st.info("👈 Upload documents in the sidebar to begin querying your knowledge base.")
    else:
        # Render historical chat context
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "citations" in message and message["citations"]:
                    with st.expander("🔍 View Context Sources"):
                        for idx, doc in enumerate(message["citations"], 1):
                            src = doc.metadata.get("source", "Unknown Document")
                            st.markdown(f"**Source {idx}:** `{src}`")
                            st.caption(doc.page_content[:300] + "...")

        # Process standard input prompt
        if prompt := st.chat_input("What would you like to know from your documents?"):
            # Render User Input
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate LLM Response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing context..."):
                    try:
                        answer, citations = ask_question(vs_info["vectorstore"], prompt, k=k_chunks)
                        response_text = answer.content
                        
                        st.markdown(response_text)
                        
                        # Collapsible Sources Section
                        if citations:
                            with st.expander("🔍 View Context Sources"):
                                for idx, d in enumerate(citations, 1):
                                    src = d.metadata.get("source", "Unknown")
                                    st.markdown(f"**Source {idx}:** `{src}`")
                                    st.caption(d.page_content.replace("\n", " ")[:300] + "...")

                        # Store in Chat History
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "citations": citations
                        })
                    except Exception as e:
                        st.error(f"Execution Error: {e}")

# --- TAB 2: DOCUMENT & METRICS WORKSPACE ---
with tab_docs:
    st.subheader("Document Index Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Vector Store Metrics</h4>
            <p><b>Indexed Objects:</b> {vs_info['num_docs']}</p>
            <p><b>Storage Path:</b> <code>./{PERSIST_DIR}</code></p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Pipeline Guardrails</h4>
            <p><b>Embedding Model:</b> <code>all-MiniLM-L6-v2</code></p>
            <p><b>Mode:</b> Strict Zero-Hallucination Prompting</p>
        </div>
        """, unsafe_allow_html=True)