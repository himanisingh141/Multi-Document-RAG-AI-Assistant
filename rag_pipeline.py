import os
import tempfile
from typing import List, Tuple
import logging

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modern standalone packages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

# New core components needed for structured chat prompts
from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)



def _save_tempfile(uploaded_file) -> str:
    """
    Save an in-memory uploaded_file (with .read() or .getvalue()) to a temporary file.
    Returns the temp file path (string).
    Caller should not delete the file immediately; keep until loaders finish.
    """
    suffix = ""
    name = getattr(uploaded_file, "name", None)
    if name and "." in name:
        suffix = "." + name.rsplit(".", 1)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    data = None
    # streamlit's UploadedFile supports getvalue() and read()
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        raise ValueError("Unsupported uploaded_file type")
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def load_documents(files: List, allowed_exts=("pdf", "txt", "md", "docx")) -> List[Document]:
    """
    Accepts a list of file-like/uploaded files (e.g., Streamlit UploadedFile).
    Returns a list of LangChain Document objects with 'source' metadata set.
    """
    docs = []
    for f in files:
        # Determine extension (be robust)
        name = getattr(f, "name", None) or ""
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        # Save to temp file when loader expects a path
        tmp_path = _save_tempfile(f)

        try:
            if ext == "pdf":
                # prefer pypdf loader (LangChain's PyPDFLoader) which expects a path
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(tmp_path)
            elif ext in ("docx", "doc"):
                from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader

                # try a couple of docx loaders depending on availability
                try:
                    loader = Docx2txtLoader(tmp_path)
                except Exception:
                    loader = UnstructuredWordDocumentLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf8")
            loaded = loader.load()
            # Ensure metadata includes "source" so we can show citations
            for d in loaded:
                d.metadata = dict(d.metadata or {})
                d.metadata["source"] = name or tmp_path
            docs.extend(loaded)
        except Exception as e:
            log.exception("Failed to load %s: %s", name, e)
        finally:
            # Remove temp file after loader finished reading it
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return docs


def create_or_load_vectorstore(
    docs: List[Document],
    persist_dir: str = "vectorstore",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
):
    """
    Create a FAISS vectorstore or load it from disk if already persisted.
    Returns (vectorstore, created_flag: bool)
    """
    os.makedirs(persist_dir, exist_ok=True)
    # If persisted index exists, load it
    try:
        # FAISS load API varies; many LangChain wrappers provide load_local()
        if os.path.exists(os.path.join(persist_dir, "index")) or os.path.exists(persist_dir):
            # Try `FAISS.load_local` if available
            if hasattr(FAISS, "load_local"):
                embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
                vs = FAISS.load_local(persist_dir, embeddings)
                return vs, False
    except Exception:
        # ignore and rebuild
        log.info("No usable persisted vectorstore found, rebuilding...")

    # Split documents into chunks
    print("-----docs ----------------------------------")
    print(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print("---------------------------------------")
    print(chunks)

    # Create embeddings (be mindful of batching in real code)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print("---------------------------------------")
    print(embeddings)
    # Build FAISS
    vs = FAISS.from_documents(chunks, embeddings)

    # Persist to disk if API available
    try:
        if hasattr(vs, "save_local"):
            vs.save_local(persist_dir)
    except Exception:
        log.exception("Failed to persist vectorstore to %s", persist_dir)

    return vs, True


def ask_question(vectorstore, query: str, llm_model: str = "llama3", k: int = 4) -> Tuple[str, List[Document]]:
    """
    Retrieve relevant docs and call LLM to answer using only the retrieved context.
    Returns (answer_text, list_of_docs_used)
    """
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(query)

    # Build context from retrieved docs
    context_pieces = []
    for i, d in enumerate(docs):
        source = d.metadata.get("source", f"doc_{i}")
        # include a small header so answer can reference the source
        context_pieces.append(f"--- Source: {source}\n{d.page_content}")
    context = "\n\n".join(context_pieces)

    prompt = (
        "You are an assistant. Use ONLY the provided context (do NOT hallucinate). "
        "If the answer cannot be found in the context, reply 'Not found'.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    if ChatOllama is None:
        raise RuntimeError("Ollama LLM wrapper not available in this environment")

    llm = ChatOllama(model=llm_model)
    # invoke() is supported on many LangChain wrappers; may vary by version.
    # If your version exposes `__call__`, you can use: llm(prompt)
    try:
        message = HumanMessage(content=prompt)
        
        # Invoke the model
        response = llm.invoke([message])
    except Exception:
      
        raise

    return response, docs