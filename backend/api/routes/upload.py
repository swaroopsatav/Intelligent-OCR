from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from backend.services.upload_service import (
    UploadService
)

router = APIRouter(
    tags=["Upload"]
)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    try:

        result = await UploadService.save_file(
            file
        )

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )