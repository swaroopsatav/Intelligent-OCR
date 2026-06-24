from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from backend.models.qwen_loader import ModelLoadingError
from backend.services.extraction_service import (
    ExtractionService
)

router = APIRouter()


@router.post("/extract")
async def extract_information(
    request: Request,
    text: Optional[str] = None
):

    if text is None:
        try:
            body = await request.json()
        except Exception:
            body = {}

        if isinstance(
            body,
            dict
        ):
            text = body.get(
                "text",
                ""
            )
        elif isinstance(
            body,
            str
        ):
            text = body
        else:
            text = ""

    try:

        result = (
            ExtractionService.extract_data(
                text or ""
            )
        )

        return result

    except ModelLoadingError as e:

        raise HTTPException(
            status_code=503,
            detail=f"Model loading failed: {e}"
        )
