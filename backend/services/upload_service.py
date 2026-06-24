import os
from pathlib import Path

from backend.config.settings import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    UPLOAD_DIR
)


class UploadService:

    @staticmethod
    def validate_extension(filename: str):
        extension = Path(filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid file format. Please upload one of: "
                f"{', '.join(ALLOWED_EXTENSIONS)}"
            )

    @staticmethod
    def validate_size(file_size: int):
        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File size exceeds "
                f"{MAX_FILE_SIZE // (1024 * 1024)} MB limit"
            )

    @staticmethod
    async def save_file(file):

        os.makedirs(
            UPLOAD_DIR,
            exist_ok=True
        )

        contents = await file.read()

        UploadService.validate_extension(
            file.filename
        )

        UploadService.validate_size(
            len(contents)
        )

        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        with open(file_path, "wb") as f:
            f.write(contents)

        return {
            "filename": file.filename,
            "file_path": file_path,
            "file_size": len(contents),
            "message": "Upload successful"
        }
