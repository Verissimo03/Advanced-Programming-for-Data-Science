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

        messages = []

        # system prompt from config.yaml
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })

        # conversation history
        if history:
            for turn in history[-5:]:
                messages.append({"role": "user", "content": turn["question"]})
                messages.append({"role": "assistant", "content": turn["answer"]})

        # user query with context ONLY (no extra rules here)
        messages.append({
            "role": "user",
            "content": f"""
Context:
{context_text}

Question:
{question}
"""
        })

        response = ollama.chat(
            model=self.model_name,
            messages=messages
        )

        return response["message"]["content"]