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
import json

# Allow Streamlit to access the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.ingestion.document_loader import DocumentLoader
from src.utils.chunker import TextChunker
from src.memory.vector_store import VectorStore
from src.models.llm import LLM
from src.memory.conversation_history import ConversationHistory
from src.utils.config_loader import load_config


# -----------------------------
# Load configuration
# -----------------------------
config = load_config()

MODEL_NAME = config["model"]["name"]
TEMPERATURE = config["model"]["temperature"]

EMBEDDING_MODEL = config["embedding"]["model"]

CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]

VECTOR_STORE_PATH = config["vector_store"]["path"]

SYSTEM_PROMPT = config["prompts"]["system"]


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
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Use session_state to avoid DB reinitialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(
        persist_directory=VECTOR_STORE_PATH,
        embedding_model=EMBEDDING_MODEL
    )

vector_store = st.session_state.vector_store


llm = LLM(
    model_name=MODEL_NAME,
    system_prompt=SYSTEM_PROMPT
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

    st.write(f"Model: {MODEL_NAME}")
    st.write(f"Embedding: {EMBEDDING_MODEL}")
    st.write("Vector DB: ChromaDB")

    st.markdown("---")

    # -----------------------------
    # Chat selection
    # -----------------------------
    history_manager = ConversationHistory()

    chat_ids = history_manager.get_chat_ids()

    if not chat_ids:
        chat_ids = ["chat_1"]
        history_manager.create_new_chat()

    selected_chat = st.selectbox("💬 Select conversation", chat_ids)

    # Create new chat
    if st.button("➕ New Chat"):
        new_chat = history_manager.create_new_chat()
        st.session_state.chat_id = new_chat
        st.rerun()

    # Store selected chat
    st.session_state.chat_id = selected_chat
    
    # Clear conversation
    if st.button("Clear Current Chat"):

        chat_id = st.session_state.chat_id

        with open("data/chat_history.json", "r") as f:
            data = json.load(f)

        # If file was corrupted (old format), fix it
        if isinstance(data, list):
            data = {}

        # Clear ONLY current chat
        data[chat_id] = []

        with open("data/chat_history.json", "w") as f:
            json.dump(data, f, indent=2)

        st.success(f"{chat_id} cleared")
        st.rerun()
        
    if st.button("🗑 Delete Chat"):

        chat_id = st.session_state.chat_id

        with open("data/chat_history.json", "r") as f:
            data = json.load(f)

        if chat_id in data:
            del data[chat_id]

        with open("data/chat_history.json", "w") as f:
            json.dump(data, f, indent=2)

        st.success(f"{chat_id} deleted")
        st.rerun()


# -----------------------------
# Document processing
# -----------------------------
if uploaded_file:

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

    with st.chat_message("user"):
        st.write(query)

    # Retrieve relevant chunks
    results = vector_store.query(query)

    documents = results if results else []

    # Load conversation history
    chat_id = st.session_state.chat_id
    history = history_manager.load_history(chat_id)

    # Generate answer
    if not documents:
        answer = "I couldn't find relevant information in the document."
    else:
        answer = llm.generate(query, documents, history)

    # Show assistant response
    with st.chat_message("assistant"):
        st.write(answer)

    # Save conversation
    history_manager.save_interaction(chat_id, query, answer)

    # Retrieved context
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

chat_id = st.session_state.chat_id
history = history_manager.load_history(chat_id)

for turn in reversed(history):

    st.markdown(f"**User:** {turn['question']}")
    st.markdown(f"**Assistant:** {turn['answer']}")
    st.markdown("---")