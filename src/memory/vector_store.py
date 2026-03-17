"""
Vector store module.

This module manages the persistent vector database used for
document storage and retrieval in the RAG system.
"""

import chromadb
from chromadb.utils import embedding_functions


class VectorStore:
    """
    Wrapper around ChromaDB to manage persistent document embeddings.
    """

    def __init__(self, persist_directory: str, embedding_model: str):
        """
        Initialize the vector store.

        Parameters
        ----------
        persist_directory : str
            Directory where the vector database will be stored.

        embedding_model : str
            Name of the embedding model used for generating embeddings.
        """

        self.persist_directory = persist_directory

        # Embedding function
        self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
            model_name=embedding_model
        )

        # Client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Collection
        self.collection_name = "documents"

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, documents, ids):
        """
        Add documents to the vector store.

        Parameters
        ----------
        documents : list[str]
            List of document text chunks.

        ids : list[str]
            Unique identifiers for each chunk.
        """

        self.collection.add(
            documents=documents,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 3):
        """
        Retrieve relevant documents from the vector store.

        Parameters
        ----------
        query_text : str
            User query.

        n_results : int
            Number of results to retrieve.

        Returns
        -------
        dict
            Retrieved documents and metadata.
        """

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        return results
    
