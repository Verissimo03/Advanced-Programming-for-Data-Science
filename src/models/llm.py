"""
LLM Module

Handles interaction with the local language model via Ollama.
"""

import ollama


class LLM:
    """
    Wrapper for the local Ollama language model.
    """

    def __init__(self, model_name: str, system_prompt: str):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def generate(self, question: str, context: list[str], history: list = None) -> str:
        """
        Generate an answer using retrieved context and conversation history.
        """

        context_text = "\n\n".join(context)

        history_text = ""
        if history:
            for turn in history[-5:]:   # keep last 5 interactions
                history_text += f"User: {turn['question']}\n"
                history_text += f"Assistant: {turn['answer']}\n"

        prompt = f"""
    {self.system_prompt}

    Conversation history:
    {history_text}

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]