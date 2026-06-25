import hashlib
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

        UploadService.validate_extension(
            file.filename
        )

        hasher = hashlib.sha256()
        total_size = 0
        
        temp_file_path = os.path.join(
            UPLOAD_DIR, 
            f"temp_{os.urandom(8).hex()}.tmp"
        )

        try:
            with open(temp_file_path, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    total_size += len(chunk)
                    UploadService.validate_size(total_size)
                    hasher.update(chunk)
                    f.write(chunk)

            content_hash = hasher.hexdigest()[:12]
            original_path = Path(file.filename)
            safe_stem = original_path.stem.replace(
                " ",
                "_"
            )
            stored_filename = (
                f"{safe_stem}_{content_hash}"
                f"{original_path.suffix.lower()}"
            )

            file_path = os.path.join(
                UPLOAD_DIR,
                stored_filename
            )

            os.replace(temp_file_path, file_path)

        except Exception:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise

        return {
            "filename": file.filename,
            "stored_filename": stored_filename,
            "file_path": file_path,
            "file_size": total_size,
            "message": "Upload successful"
        }
