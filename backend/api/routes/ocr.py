from fastapi import (
    APIRouter,
    HTTPException
)

from backend.services.ocr_service import (
    OCRService
)

router = APIRouter(
    tags=["OCR"]
)


@router.post("/ocr")
async def run_ocr(
    image_path: str
):

    try:

        image_path = (
            image_path
            .strip()
            .strip('"')
            .replace("\\", "/")
        )

        print(
            "OCR Image Path:",
            repr(image_path)
        )

        result = (
            OCRService.extract_text(
                image_path
            )
        )

        if (
            result["status"]
            == "error"
        ):

            raise HTTPException(
                status_code=500,
                detail=(
                    "OCR failed: "
                    f"{result.get('message', 'Unable to extract text.')}"
                )
            )

        return result

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {e}"
        )
