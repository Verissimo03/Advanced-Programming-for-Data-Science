# Advanced Programming for Data Science

This project implements a **Retrieval-Augmented Generation (RAG) system** that allows users to interact with documents using local AI models.

The system supports ingestion and querying of multiple document formats and uses a persistent memory store for efficient retrieval.

## Features

- Local AI models using Ollama
- Document ingestion
  - TXT
  - Markdown
  - PDF
  - DOCX
- Persistent vector memory store
- Configurable chunking and prompts
- Conversation history
- Streamlit frontend interface

## Architecture

The system follows a typical RAG pipeline:

User Query -> Embedding generation -> Vector retrieval -> Context injection -> LLM response

## Project Structure

data/ # Raw and processed documents
src/ # Core system modules
config/ # Configuration files
frontend/ # Streamlit interface
tests/ # Unit tests


## Technologies

- Python
- Ollama (local LLMs)
- Streamlit
- Vector databases
- Document processing libraries

## Goal

The objective is to build a fully functional RAG pipeline capable of retrieving knowledge from documents and generating accurate answers using local AI models.

## How to Run

1. Clone the repository

git clone https://github.com/Verissimo03/Advanced-Programming-for-Data-Science.git
cd Advanced-Programming-for-Data-Science

2. Create a virtual environment

python3 -m venv venv

3. Activate the virtual environment

source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

5. Run the Streamlit interface

streamlit run frontend/app.py
