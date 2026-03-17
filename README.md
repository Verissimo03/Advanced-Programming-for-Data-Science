# Advanced Programming for Data Science  
## Retrieval-Augmented Generation (RAG) System

This project implements a **Retrieval-Augmented Generation (RAG) system** that enables users to interact with documents using **local AI models**. The system combines document processing, semantic search, and language generation to provide accurate, context-aware answers.

---

# Overview

The goal of this project is to build a complete pipeline where a user can:

- Upload a document  
- Ask natural language questions  
- Receive answers grounded in the document’s content  

Unlike standard language models, this system retrieves relevant information from user-provided documents in real time instead of relying only on pre-trained knowledge.

---

# Key Features

- Local AI models using **Ollama**  
- Multi-format document ingestion:
  - TXT  
  - PDF  
  - Markdown  
  - DOCX  
- Semantic search using a **vector database (ChromaDB)**  
- Configurable text chunking  
- Context-aware response generation  
- Persistent storage of embeddings  
- Conversation history tracking  
- Interactive **Streamlit frontend**

---

# RAG Architecture

The system follows a standard Retrieval-Augmented Generation pipeline:

- User Query  
- Query Embedding  
- Vector Search (ChromaDB)  
- Retrieval of relevant document chunks  
- Context injection into the LLM  
- Generated answer  

This ensures responses are grounded in the uploaded document.

---

# Project Structure

- `data/`  
  - `raw/` → uploaded original documents  

- `frontend/`  
  - `app.py` → Streamlit user interface  

- `src/`  
  - `ingestion/`  
    - `document_loader.py` → loads and extracts text from files  

  - `utils/`  
    - `chunker.py` → splits text into smaller chunks  

  - `memory/`  
    - `vector_store.py` → handles embedding storage and retrieval (ChromaDB)  
    - `conversation_history.py` → stores past interactions  

  - `models/`  
    - `llm.py` → interface to the local language model (Ollama)  

- `config/` → configuration files (optional)  
- `tests/` → unit tests  

---

# Module Description

## DocumentLoader (`src/ingestion/document_loader.py`)
- Loads documents from different formats (TXT, PDF, DOCX, etc.)  
- Converts them into plain text  

## TextChunker (`src/utils/chunker.py`)
- Splits long text into smaller chunks  
- Improves retrieval accuracy and respects model limits  

## VectorStore (`src/memory/vector_store.py`)
- Converts text chunks into embeddings  
- Stores them in ChromaDB  
- Retrieves relevant chunks using similarity search  
- Acts as the system’s semantic memory  

## LLM (`src/models/llm.py`)
- Interfaces with the local language model  
- Generates answers using retrieved context  
- Ensures responses are grounded in the document  

## ConversationHistory (`src/memory/conversation_history.py`)
- Stores user interactions  
- Enables contextual conversations  

## Streamlit App (`frontend/app.py`)
- Provides the user interface  
- Handles file upload, queries, and display of results  

---

# Data Flow

- User uploads a document → stored in `data/raw/`  
- Document is loaded and converted to text  
- Text is split into chunks  
- Chunks are converted into embeddings  
- Embeddings are stored in `data/vector_store/`  
- User asks a question  
- Relevant chunks are retrieved  
- LLM generates an answer using that context  

---

# Technologies Used

- Python  
- Ollama (local LLM and embeddings)  
- ChromaDB (vector database)  
- Streamlit (frontend)  
- Document processing libraries  

---

# How to Run


## 1. Clone the repository

git clone https://github.com/Verissimo03/Advanced-Programming-for-Data-Science.git

cd Advanced-Programming-for-Data-Science

## 2. Create a virtual environment

python3 -m venv venv

## 3. Activate the virtual environment

source venv/bin/activate

## 4. Install dependencies

pip install -r requirements.txt

## 5. Run the Streamlit interface

streamlit run frontend/app.py


# Conclusion

This project demonstrates how to combine:
- Document processing
- Vector databases
- Language models

to build an intelligent system capable of retrieving and generating answers based on user-provided documents.

The architecture is modular and can be extended with features such as:
- Multi-document support
- Improved user interface
- Advanced retrieval techniques
- More powerful language models