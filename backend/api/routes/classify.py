from fastapi import APIRouter, HTTPException

from backend.models.qwen_loader import ModelLoadingError
from backend.services.classification_service import (
    ClassificationService
)

router = APIRouter()


@router.post("/classify")
async def classify_document(
    text: str
):

    try:

        result = (
            ClassificationService.classify(
                text
            )
        )

        return result

    except ModelLoadingError as e:

        raise HTTPException(
            status_code=503,
            detail=f"Model loading failed: {e}"
        )
