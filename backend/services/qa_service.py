from backend.services.llm_service import (
    LLMService
)


class QAService:

    @staticmethod
    def answer_question(
        document_text,
        question
    ):

        prompt = f"""
You are a document assistant.

Answer ONLY from the provided document.

If information is not available,
respond with:

"Information not found in document"

DOCUMENT:

{document_text}

QUESTION:

{question}

ANSWER:
"""

        return LLMService.generate(
            prompt
        )