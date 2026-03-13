"""
Conversation History Module

Stores and retrieves chat history for the RAG assistant.
"""

import json
import os


class ConversationHistory:
    """
    Handles storage and retrieval of chat conversations.
    """

    def __init__(self, file_path: str = "data/chat_history.json"):
        self.file_path = file_path

        # Create history file if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)

    def load_history(self):
        """Load conversation history."""

        with open(self.file_path, "r") as f:
            return json.load(f)

    def save_interaction(self, question: str, answer: str):
        """Save a question-answer pair."""

        history = self.load_history()

        history.append({
            "question": question,
            "answer": answer
        })

        with open(self.file_path, "w") as f:
            json.dump(history, f, indent=4)