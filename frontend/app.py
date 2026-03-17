"""
Streamlit Frontend for the RAG System

Features
- Upload documents
- Ask questions
- Retrieve answers using RAG
- Display retrieved chunks
- Store conversation history
"""

import sys
import os

# Allow Streamlit to access the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.ingestion.document_loader import DocumentLoader
from src.utils.chunker import TextChunker
from src.memory.vector_store import VectorStore
from src.models.llm import LLM
from src.memory.conversation_history import ConversationHistory


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Page title
# -----------------------------
st.markdown(
"""
# 🤖 Retrieval-Augmented Generation Assistant

Upload a document and ask questions about its content using a local AI model.
"""
)


# -----------------------------
# Initialize system components
# -----------------------------
loader = DocumentLoader()

chunker = TextChunker(
    chunk_size=150,
    chunk_overlap=20
)

# Use session_state to avoid DB issues
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(
        persist_directory="data/vector_store",
        embedding_model="nomic-embed-text"
    )

vector_store = st.session_state.vector_store


llm = LLM(
    model_name="phi3:mini",
    system_prompt="You are an assistant that analyzes documents. The user uploaded a document. The following text chunks come from that document. Use this context to answer the question. You may summarize, analyze, or comment on the content. If the answer is not contained in the context, say you don't know. Context: {context} Question:{question}Answer:"
)

history_manager = ConversationHistory()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("📂 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "md", "docx"]
    )

    st.markdown("---")

    st.title("⚙️ System")

    st.write("Model: phi3:mini")
    st.write("Embedding: nomic-embed-text")
    st.write("Vector DB: ChromaDB")

    st.markdown("---")
    
    # -----------------------------
    # Clear conversation
    # -----------------------------
    if st.button("Clear Conversation"):

        open("data/chat_history.json", "w").write("[]")

        st.success("Conversation cleared")


# -----------------------------
# Document processing
# -----------------------------
if uploaded_file:

    # Prevent re-processing same file
    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    if uploaded_file.name != st.session_state.last_file:

        st.session_state.last_file = uploaded_file.name

        with st.spinner("Processing document..."):

            file_path = os.path.join("data/raw", uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            text = loader.load(file_path)

            chunks = chunker.split(text)

            # Prevent re-processing same file
            vector_store.reset()

            ids = [f"chunk_{i}" for i in range(len(chunks))]

            vector_store.add_documents(
                documents=chunks,
                ids=ids
            )

        st.success(f"Document processed successfully ({len(chunks)} chunks stored).")


# -----------------------------
# Chat Interface
# -----------------------------
st.markdown("## 💬 Ask a Question")

query = st.chat_input("Ask something about the document...")


if query:

    # Show user message
    with st.chat_message("user"):
        st.write(query)

    # Retrieve relevant chunks
    results = vector_store.query(query)

    documents = []

    if results and "documents" in results and results["documents"]:
        documents = results["documents"][0]

    # Load conversation history
    history = history_manager.load_history()

    # Generate answer
    if not documents:
        answer = "I couldn't find relevant information in the document."
    else:
        answer = llm.generate(query, documents, history)

    # Show assistant response
    with st.chat_message("assistant"):
        st.write(answer)

    # Save conversation
    history_manager.save_interaction(query, answer)

    # -----------------------------
    # Retrieved context visualization
    # -----------------------------
    with st.expander("🔎 Retrieved Context"):

        for i, doc in enumerate(documents):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc)
            st.markdown("---")


# -----------------------------
# Conversation history display
# -----------------------------
st.markdown("---")
st.markdown("### 🧠 Conversation History")

history = history_manager.load_history()

for turn in reversed(history):

    st.markdown(f"**User:** {turn['question']}")
    st.markdown(f"**Assistant:** {turn['answer']}")
    st.markdown("---")