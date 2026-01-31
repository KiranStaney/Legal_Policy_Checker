# app/services/llm.py
import os
import json
from google import genai
from typing import Dict, Any


class LLMClient:
    """
    Gemini Client Wrapper for Summary, Structured Output & Chat
    """

    def __init__(self):
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY not found in environment variables.")

        # Initialize Google Gemini client
        self.client = genai.Client(api_key=api_key)

        # Default model
        self.model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    # -----------------------
    # SUMMARY FUNCTION
    # -----------------------
    def get_summary(self, document_text: str) -> str:
        prompt_text = document_text[:6000]

        prompt = (
            "Analyze the legal document and produce a clear, readable summary "
            "with bullet points. Avoid overly long sentences.\n\n"
            f"DOCUMENT:\n{prompt_text}"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"LLM Summary Error: {e}"

    # -----------------------
    # STRUCTURED OUTPUT
    # -----------------------
    def get_structured_response(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema
                }
            )
            return json.loads(response.text)
        except Exception as e:
            raise RuntimeError(f"Failed to get structured response: {e}")

    # -----------------------
    # CHAT FUNCTION
    # -----------------------
    def chat(self, query: str, context: str = "") -> str:
        prompt = (
            "You are a legal contract AI assistant. Answer clearly.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUERY:\n{query}\n"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Chat Error: {e}"

    # -----------------------
    # Text-only LLM call
    # -----------------------
    def get_text_response(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"LLM Text Error: {e}"


# GLOBAL LLM CLIENT
llm_client = LLMClient()
