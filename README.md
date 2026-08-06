# Multi-Document-RAG-AI-Assistant
Local RAG AI assistant for private doc Q&amp;A built with Streamlit, LangChain, FAISS, and Ollama. Users upload PDFs, Word docs, or text files, which are chunked and converted into vector embeddings. The system searches relevant contexts to deliver fast, accurate answers from local LLMs (like Llama 3) complete with source citations and zero data leaks.
## Flow Chart:
┌─────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐
│ Uploaded Files  │ ───► │  Recursive Text Splitter│ ───► │ HuggingFace Embeddings │
│ (PDF/DOCX/TXT)  │      │ (800 chars, 150 overlap)│      │  (all-MiniLM-L6-v2)    │
└─────────────────┘      └─────────────────────────┘      └───────────┬────────────┘
                                                                      │
                                                                      ▼ 
┌─────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐
│ Verified Answer │ ◄─── │      Ollama (Llama 3)   │ ◄─── │      FAISS Vector      │
│  + Citations    │      │  (Strict Context Prompt)│      │      Database Index    │
└─────────────────┘      └─────────────────────────┘      └────────────────────────┘
<img width="1606" height="839" alt="Screenshot 2026-07-30 at 8 41 11 PM" src="https://github.com/user-attachments/assets/730a337b-a136-467f-8132-ba35caa69c75" />
