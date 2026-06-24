from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.services.pdf_service import PDFService
from backend.services.image_service import ImageService

router = APIRouter(
    tags=["Preprocessing"]
)


@router.post("/preprocess")
async def preprocess_document(
    file_path: str
):

    try:

        file_path = file_path.strip('"')

        print(
            f"Received file path: {file_path}"
        )

        if not Path(file_path).exists():

            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )

        extension = (
            Path(file_path)
            .suffix
            .lower()
        )

        processed_images = []
        original_images = []
        preprocessing_results = []

        if extension == ".pdf":

            image_paths = (
                PDFService.pdf_to_images(
                    file_path
                )
            )

        else:

            image_paths = [file_path]

        for image_path in image_paths:

            result = (
                ImageService.preprocess_image(
                    image_path
                )
            )

            original_images.append(
                result[
                    "original_image"
                ]
            )

            processed_images.append(
                result[
                    "processed_image"
                ]
            )

            preprocessing_results.append(
                result
            )

        return {
            "original_images":
                original_images,
            "processed_images":
                processed_images,
            "preprocessing_results":
                preprocessing_results
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
