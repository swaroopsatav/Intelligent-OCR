from fastapi import APIRouter, HTTPException

from backend.models.qwen_loader import ModelLoadingError
from backend.services.qa_service import (
    QAService
)

router = APIRouter()


@router.post("/qa")
async def ask_question(
    document_text: str,
    question: str
):

    try:

        answer = QAService.answer_question(
            document_text,
            question
        )

    except ModelLoadingError as e:

        raise HTTPException(
            status_code=503,
            detail=f"Model loading failed: {e}"
        )

    return {
        "question": question,
        "answer": answer
    }
