from fastapi import APIRouter

from backend.services.correction_service import (
    CorrectionService
)

router = APIRouter()


@router.post("/correct")
async def correct_text(
    ocr_text: str
):

    result = (
        CorrectionService.correct_text(
            ocr_text
        )
    )

    return result
