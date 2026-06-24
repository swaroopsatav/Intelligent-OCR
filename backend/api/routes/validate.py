from fastapi import APIRouter

from backend.services.validation_service import (
    ValidationService
)

router = APIRouter()


@router.post("/validate")
async def validate_document(
    data: dict
):

    result = (
        ValidationService
        .validate_document(data)
    )

    return result