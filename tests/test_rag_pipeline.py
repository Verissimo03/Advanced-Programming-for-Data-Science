"""
Simple test to validate the RAG pipeline components.

This script tests the full RAG workflow:

1. Load a document
2. Split it into chunks
3. Store chunks in a vector database
4. Ask a question
5. Retrieve relevant chunks
6. Generate an answer using the LLM
7. Use conversation history for context
8. Save the interaction
"""

from src.ingestion.document_loader import DocumentLoader
from src.memory.vector_store import VectorStore
from src.utils.chunker import TextChunker
from src.models.llm import LLM
from src.memory.conversation_history import ConversationHistory


# -----------------------------
# Load document
# -----------------------------
# The DocumentLoader reads the file from disk
# and returns the full text content.
loader = DocumentLoader()
text = loader.load("data/raw/test.pdf")

print("Document loaded:")
print(text)


# -----------------------------
# Chunk the document
# -----------------------------
# The TextChunker splits the document into
# smaller pieces so they can be embedded
# and stored in the vector database.
chunker = TextChunker(chunk_size=200, chunk_overlap=20)

chunks = chunker.split(text)

print("\nChunks created:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")


# -----------------------------
# Initialize vector store
# -----------------------------
# VectorStore wraps ChromaDB and manages:
# - document embeddings
# - storage
# - similarity search
vector_store = VectorStore(
    persist_directory="data/vector_store",
    embedding_model="nomic-embed-text"
)


# -----------------------------
# Store chunks
# -----------------------------
# Each chunk must have a unique ID.
ids = [f"chunk_{i}" for i in range(len(chunks))]

vector_store.add_documents(
    documents=chunks,
    ids=ids
)

print("\nChunks stored in vector database.")


# -----------------------------
# Ask a question
# -----------------------------
# This simulates a user asking something.
query = "Where is it used?"

print("\nUser question:")
print(query)


# -----------------------------
# Retrieve relevant chunks
# -----------------------------
# The query is embedded and compared
# with stored embeddings to find
# the most similar document chunks.
results = vector_store.query(query)

documents = results["documents"][0]

print("\nRetrieved chunks:")
for doc in documents:
    print("-", doc)


# -----------------------------
# Load conversation history
# -----------------------------
# ConversationHistory loads previous
# interactions from the JSON file.
history_manager = ConversationHistory()
history = history_manager.load_history()


# -----------------------------
# Initialize LLM
# -----------------------------
# The LLM wrapper connects to Ollama
# and sends prompts to the local model.
llm = LLM(
    model_name="phi3:mini",
    system_prompt="You answer questions using the provided context. If the answer is not in the context, say you don't know."
)


# -----------------------------
# Generate answer
# -----------------------------
# The model receives:
# - conversation history
# - retrieved document chunks
# - current user question
answer = llm.generate(query, documents, history)

print("\nLLM Answer:")
print(answer)


# -----------------------------
# Save conversation
# -----------------------------
# After generating the answer,
# the interaction is saved so it can
# be used in future conversations.
history_manager.save_interaction(query, answer)

print("\nConversation saved.")


# -----------------------------
# Debug output
# -----------------------------
# This prints the raw retrieval results
# from the vector database for inspection.
print("\nQuery results:")
print(results)